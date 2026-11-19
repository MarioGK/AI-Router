package pin

import (
	"context"
	"time"

	"github.com/MarioGK/AI-Router/internal/usage"
)

// ThreadActivity is how a settled, decided thread went: its pin, the
// decision that made it, what its logged requests took and every change
// to its pin since.
type ThreadActivity struct {
	Key string
	// Decided is the model the thread was pinned to first, Level the
	// reasoning level decided, and Model the model it is pinned to now.
	Decided string
	Level   string
	Model   string
	// Decision is the id of the decision that made the pin, and Reviewed
	// whether an operator judged it.
	Decision int64
	Reviewed bool
	// Explored is set when the decision routed the thread one ladder step
	// cheaper than decided, to learn whether that step would do; Propensity
	// is the probability the decision's action had, 0 when the decision
	// was not borderline enough to be explored.
	Explored   bool
	Propensity float64
	// Requests counts the thread's regular turns; First and Last are when
	// its first and last logged request arrived, and Active the sum of
	// every logged request's latency less its wait for a concurrency slot:
	// queueing is a matter of capacity, which spill handles, not of how
	// fast the model works.
	Requests    int
	First, Last time.Time
	Active      time.Duration
	Tokens      usage.Tokens
	Cost        float64
	// Changes are the thread's events after its decision, oldest first:
	// failovers, capability upgrades, migrations, overrides and any other
	// kind a later build records.
	Changes []Event
}

// activitiesQuery reads the settled, decided threads pinned since ?1 whose
// last request is older than ?2. It reads pins through pins_created (named,
// and joined first, so the planner never walks every pin in key order),
// their requests through requests_key and their decision through
// events_key, so it never scans a whole table.
const activitiesQuery = `WITH threads AS (
		SELECT pins.key, pins.decided, pins.decided_level, pins.model,
			COUNT(*) FILTER (WHERE requests.kind = 'messages' AND requests.reason NOT IN ('side_request', 'auxiliary'))
				AS turns,
			MIN(requests.at) AS first, MAX(requests.at) AS last, SUM(MAX(requests.latency_ms - requests.queue_ms, 0)) AS active,
			SUM(requests.input_tokens) AS input, SUM(requests.output_tokens) AS output,
			SUM(requests.cache_read_tokens) AS cache_read, SUM(requests.cache_write_tokens) AS cache_write,
			COALESCE(SUM(requests.cost_usd), 0) AS cost
		FROM pins INDEXED BY pins_created CROSS JOIN requests ON requests.key = pins.key
		WHERE pins.created_at >= ?1
		GROUP BY pins.key HAVING MAX(requests.at) < ?2 AND MAX(requests.reason = 'fixed') = 0),
	decided AS (
		SELECT threads.*, (SELECT MIN(events.id) FROM events WHERE events.key = threads.key AND events.kind = ?3)
			AS decision
		FROM threads)
	SELECT key, decided, decided_level, model, decision,
		EXISTS (SELECT 1 FROM reviews WHERE reviews.event = decided.decision),
		COALESCE((SELECT basis->>'explore' FROM events WHERE events.id = decided.decision), 0),
		COALESCE((SELECT basis->>'propensity' FROM events WHERE events.id = decided.decision), 0),
		turns, first, last, active, input, output, cache_read, cache_write, cost
	FROM decided WHERE decision IS NOT NULL ORDER BY key`

// changesQuery reads the events other than decisions of the threads pinned
// since ?2, through pins_created and events_key.
const changesQuery = `SELECT events.key, events.at, events.kind, events.model, events.previous, events.level, events.detail
	FROM pins INDEXED BY pins_created CROSS JOIN events ON events.key = pins.key
	WHERE pins.created_at >= ?2 AND events.kind != ?1 ORDER BY events.id`

// ThreadActivities reads the threads pinned at or after since by a decision
// whose last logged request is older than settled, the same threads the
// cost per task is learned from, except that re-pinned threads are kept:
// how a thread was moved is part of how it went. Threads containing
// fixed-mode requests are left out. Requests counts only the thread's
// regular turns: not token counts, compactions or background calls.
func (s *Store) ThreadActivities(ctx context.Context, since, settled time.Time) ([]ThreadActivity, error) {
	rows, err := s.read.QueryContext(ctx, activitiesQuery, since.UnixMilli(), settled.UnixMilli(), Decision)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	threads := []ThreadActivity{}
	index := map[string]int{}
	for rows.Next() {
		var t ThreadActivity
		var first, last, active int64
		if err := rows.Scan(&t.Key, &t.Decided, &t.Level, &t.Model, &t.Decision, &t.Reviewed, &t.Explored, &t.Propensity, &t.Requests,
			&first, &last, &active, &t.Tokens.Input, &t.Tokens.Output, &t.Tokens.CacheRead, &t.Tokens.CacheWrite,
			&t.Cost); err != nil {
			return nil, err
		}
		t.First, t.Last = time.UnixMilli(first), time.UnixMilli(last)
		t.Active = time.Duration(active) * time.Millisecond
		index[t.Key] = len(threads)
		threads = append(threads, t)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(threads) == 0 {
		return threads, nil
	}
	changes, err := s.read.QueryContext(ctx, changesQuery, Decision, since.UnixMilli())
	if err != nil {
		return nil, err
	}
	defer changes.Close()
	for changes.Next() {
		var e Event
		var at int64
		if err := changes.Scan(&e.Key, &at, &e.Kind, &e.Model, &e.Previous, &e.Level, &e.Detail); err != nil {
			return nil, err
		}
		if i, ok := index[e.Key]; ok {
			e.At = time.UnixMilli(at)
			threads[i].Changes = append(threads[i].Changes, e)
		}
	}
	return threads, changes.Err()
}
