package pin

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
)

// DecisionFilter selects decisions, newest first. Review is Unreviewed,
// Right or Wrong to select by review.
type DecisionFilter struct {
	Search  string
	Model   string
	Decider string
	Review  string
	Limit   int
	Offset  int
}

// DecisionPage is one page of decisions plus the total matching the filter.
type DecisionPage struct {
	Decisions []Event
	Total     int
}

// ModelDecisions is how many decisions pinned one model, with the latest
// recorded prompts that went to it.
type ModelDecisions struct {
	Model    string
	Count    int
	Examples []Event
}

const (
	// Decision listing limits.
	decisionLimit    = 25
	maxDecisionLimit = 200
	// examplesPerModel is how many recent prompts each model shows.
	examplesPerModel = 3
	// recorded matches the events that still hold a prompt or a decider's
	// raw reply.
	recorded = `(prompt != '' OR first_prompt != '' OR EXISTS (SELECT 1 FROM json_each(attempts)
		WHERE json_type(value, '$.response') IS NOT NULL))`
)

// Decisions lists decisions, newest first. The limit defaults to 25 and is
// capped at 200.
func (s *Store) Decisions(ctx context.Context, filter DecisionFilter) (DecisionPage, error) {
	var where conditions
	where.add("kind = ?", Decision)
	where.equal("model", filter.Model)
	where.equal("decider", filter.Decider)
	if clause, ok := reviewed[filter.Review]; ok {
		where.clauses = append(where.clauses, clause)
	}
	if err := where.search("id", filter.Search); err != nil {
		return DecisionPage{}, err
	}
	var page DecisionPage
	if err := s.read.QueryRowContext(ctx, `SELECT COUNT(*) FROM events`+where.String(), where.args...).Scan(&page.Total); err != nil {
		return DecisionPage{}, err
	}
	decisions, err := events(ctx, s.read, where.String()+` ORDER BY id DESC LIMIT ? OFFSET ?`,
		append(where.args, bounded(filter.Limit, decisionLimit, maxDecisionLimit), max(filter.Offset, 0))...)
	if err != nil {
		return DecisionPage{}, err
	}
	page.Decisions = decisions
	return page, nil
}

// DecisionMix counts the decisions since the given time per pinned model,
// most first, each with its latest three recorded prompts.
func (s *Store) DecisionMix(ctx context.Context, since time.Time) ([]ModelDecisions, error) {
	// One read transaction counts and picks the examples from one snapshot.
	tx, err := s.read.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	rows, err := tx.QueryContext(ctx, `SELECT model, COUNT(*) FROM events WHERE kind = ? AND at >= ?
		GROUP BY model ORDER BY COUNT(*) DESC, model`, Decision, since.UnixMilli())
	if err != nil {
		return nil, err
	}
	var mix []ModelDecisions
	index := map[string]int{}
	for rows.Next() {
		var m ModelDecisions
		if err := rows.Scan(&m.Model, &m.Count); err != nil {
			rows.Close()
			return nil, err
		}
		index[m.Model] = len(mix)
		mix = append(mix, m)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	examples, err := events(ctx, tx, ` WHERE id IN (SELECT id FROM (SELECT id,
			ROW_NUMBER() OVER (PARTITION BY model ORDER BY id DESC) AS rank
		FROM events WHERE kind = ? AND prompt != '' AND at >= ?) WHERE rank <= ?) ORDER BY id DESC`,
		Decision, since.UnixMilli(), examplesPerModel)
	if err != nil {
		return nil, err
	}
	for _, example := range examples {
		if i, ok := index[example.Model]; ok {
			mix[i].Examples = append(mix[i].Examples, example)
		}
	}
	return mix, nil
}

// PrunePrompts blanks the prompts recorded with each decision and
// redecision older than before and drops its deciders' raw replies, in bounded batches, and
// reports how many decisions it changed. The decisions themselves are
// kept, and a reviewed decision keeps everything its review rests on.
func (s *Store) PrunePrompts(ctx context.Context, before time.Time) (int64, error) {
	cutoff := before.UnixMilli()
	// A reader finds the ids still to thin, so the writer walks only those.
	var first, last sql.NullInt64
	if err := s.read.QueryRowContext(ctx, `SELECT MIN(id), MAX(id) FROM events WHERE kind IN (?, ?) AND at < ? AND `+
		recorded+` AND `+unreviewed, Decision, Redecision, cutoff).Scan(&first, &last); err != nil || !last.Valid {
		return 0, err
	}
	var total int64
	// Each batch covers one range of ids, so it holds the writer for at most
	// pruneBatch events.
	for from := first.Int64 - 1; from < last.Int64; from += pruneBatch {
		result, err := s.write.ExecContext(ctx, `UPDATE events SET prompt = '', first_prompt = '',
			attempts = (SELECT json_group_array(json_remove(value, '$.response')) FROM json_each(events.attempts))
			WHERE id > ? AND id <= ? AND kind IN (?, ?) AND at < ? AND `+recorded+` AND `+unreviewed,
			from, from+pruneBatch, Decision, Redecision, cutoff)
		if err != nil {
			return total, err
		}
		changed, err := result.RowsAffected()
		if err != nil {
			return total, err
		}
		total += changed
	}
	return total, nil
}

// SavePolicy records a policy under its fingerprint, once.
func (s *Store) SavePolicy(ctx context.Context, policy decide.Policy) error {
	encoded, err := json.Marshal(policy)
	if err != nil {
		return err
	}
	_, err = s.write.ExecContext(ctx, `INSERT OR IGNORE INTO policies (fingerprint, at, policy) VALUES (?, ?, ?)`,
		policy.Fingerprint(), s.now().UnixMilli(), string(encoded))
	return err
}

// Policy returns the policy recorded under a fingerprint; nil when unknown.
func (s *Store) Policy(ctx context.Context, fingerprint string) (*decide.Policy, error) {
	var encoded string
	err := s.read.QueryRowContext(ctx, `SELECT policy FROM policies WHERE fingerprint = ?`, fingerprint).Scan(&encoded)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var policy decide.Policy
	if err := json.Unmarshal([]byte(encoded), &policy); err != nil {
		return nil, err
	}
	return &policy, nil
}
