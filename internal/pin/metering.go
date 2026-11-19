package pin

import (
	"context"
	"errors"

	"github.com/MarioGK/AI-Router/internal/usage"
)

// RepairMetering repairs the requests metered before commit 5433f61
// ("meter translated streams right"), which went live on (the cutoff, 1790726400000 ms, is that day's start in UTC), and reports
// how many it repaired.
//
// Until then the usage meter kept the largest report of every field. A
// stream CLIProxy translated from another API (Codex, DeepSeek) opens with
// an estimate of the whole prompt as input_tokens and splits it into fresh
// input and cache reads only in its final message_delta, so the log kept
// the estimate, which is about input plus cache reads plus cache writes,
// as input_tokens next to the real cache_read_tokens: the prompt was billed
// twice, and the cache hit rate read about 0.58 where it was about 0.98.
// The real fresh input was overwritten and cannot be read back.
//
// A row is repaired when all of these hold: it was logged before the
// cutoff; its model is not a Claude model, whose streams are Anthropic's
// own and were metered right; it succeeded; its cache reads are at least
// 10000 tokens; and its input_tokens is at least half of its cache reads
// and writes. The last two are the bug's shape: the opening estimate counts
// the whole prompt, cached part included, give or take the estimator's
// error (the live rows read about 100000 against 108000 cached), so the
// stored input is close to what was cached. Rows the fix metered keep their
// fresh input at a few percent of their cache reads; a correct turn with
// 10000 or more cached and fresh input over half of it is rare, and a
// smaller cache threshold would catch small turns (1000 fresh beside 1000
// cached). Since a correct row can still match, the repair keeps the logged
// values to restore.
//
// The corrected input_tokens is the estimate less the cache reads and
// writes, never below zero: an approximation of the fresh input, which was overwritten and
// cannot be read back.
//
// The repair is opt-in (ai-router repair-metering --apply), since the log
// cannot tell a translated stream from a plain JSON response and a
// correct row can match; without apply it only counts the rows it would
// repair. Nothing is lost: the repaired rows are flagged (repaired = 1) and keep
// their logged input_tokens and cost_usd in original_input_tokens and
// original_cost_usd, which stay NULL on every other row. cost_usd, priced
// on the estimate, is cleared for RepriceRepaired to price again.
func (s *Store) RepairMetering(ctx context.Context, apply bool) (int, error) {
	if !apply {
		var n int
		err := s.write.QueryRowContext(ctx, `SELECT count(*) FROM requests WHERE `+mismetered).Scan(&n)
		return n, err
	}
	result, err := s.write.ExecContext(ctx, `UPDATE requests SET
	repaired = 1,
	original_input_tokens = input_tokens,
	original_cost_usd = cost_usd,
	input_tokens = MAX(input_tokens - cache_read_tokens - cache_write_tokens, 0),
	cost_usd = NULL
WHERE `+mismetered)
	if err != nil {
		return 0, err
	}
	n, err := result.RowsAffected()
	return int(n), err
}

// mismetered selects the unrepaired rows with the bug's shape.
const mismetered = `repaired = 0
	AND at < 1790726400000
	AND model NOT LIKE 'claude-%'
	AND status BETWEEN 200 AND 299
	AND cache_read_tokens >= 10000
	AND input_tokens * 2 >= cache_read_tokens + cache_write_tokens`

// meteringRepairMigration adds the columns RepairMetering fills.
const meteringRepairMigration = `
ALTER TABLE requests ADD COLUMN repaired INTEGER NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN original_input_tokens INTEGER;
ALTER TABLE requests ADD COLUMN original_cost_usd REAL;
`

// RepriceRepaired prices the requests the metering repair corrected, with
// the price of their model; a model without one stays unpriced. It reports
// how many requests it priced.
func (s *Store) RepriceRepaired(ctx context.Context, price func(model string) (usage.Price, bool)) (int, error) {
	rows, err := s.write.QueryContext(ctx, `SELECT id, model, input_tokens, output_tokens, cache_read_tokens,
		cache_write_tokens FROM requests WHERE repaired = 1 AND cost_usd IS NULL`)
	if err != nil {
		return 0, err
	}
	type priced struct {
		id   int64
		cost float64
	}
	var batch []priced
	for rows.Next() {
		var id int64
		var model string
		var t usage.Tokens
		if err := rows.Scan(&id, &model, &t.Input, &t.Output, &t.CacheRead, &t.CacheWrite); err != nil {
			rows.Close()
			return 0, err
		}
		if p, ok := price(model); ok {
			batch = append(batch, priced{id, p.Cost(t)})
		}
	}
	if err := errors.Join(rows.Err(), rows.Close()); err != nil {
		return 0, err
	}
	tx, err := s.write.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	for _, p := range batch {
		if _, err := tx.ExecContext(ctx, `UPDATE requests SET cost_usd = ? WHERE id = ?`, p.cost, p.id); err != nil {
			return 0, err
		}
	}
	return len(batch), tx.Commit()
}
