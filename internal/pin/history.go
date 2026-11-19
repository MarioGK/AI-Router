package pin

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/usage"
)

// ThreadFilter selects pinned threads, newest activity first.
type ThreadFilter struct {
	Model  string
	Limit  int
	Offset int
}

// Thread is a pinned thread with its activity and the decision that
// created the pin (nil when none was recorded). Requests, Tokens and Cost
// cover the requests still in the log.
type Thread struct {
	Pin
	LastUsedAt time.Time
	Requests   int
	Tokens     usage.Tokens
	Cost       float64
	Decision   *Event
}

// ThreadPage is one page of threads plus the total matching the filter.
type ThreadPage struct {
	Threads []Thread
	Total   int
}

// ThreadDetail is a thread with its whole event log, its latest requests,
// newest first, the policy its decision was made under (nil when not
// recorded) and the pin keys of its subagents' threads, oldest first.
type ThreadDetail struct {
	Thread
	Events   []Event
	Requests []RequestLog
	Policy   *decide.Policy
	Agents   []string
}

// Bucket counts requests and decisions per model in one time slot.
type Bucket struct {
	Start     time.Time
	Requests  map[string]int
	Decisions map[string]int
}

// Overview is the dashboard's headline view of a period.
type Overview struct {
	Summary  Summary
	Threads  int
	Requests int
	Errors   int
	// Cost is the priced cost of the period's requests in US dollars;
	// Models breaks the period down per model, costliest first.
	Cost   float64
	Models []ModelCost
	// Queue is the period's queue wait per model.
	Queue    map[string]QueueWait
	Timeline []Bucket
}

// ModelCost is what one model's requests cost in a period. Threads counts
// the pinned threads they belong to, so Cost / Threads is the cost per task.
type ModelCost struct {
	Model    string
	Threads  int
	Requests int
	Tokens   usage.Tokens
	Cost     float64
}

// PerThread is the model's average cost per pinned thread.
func (m ModelCost) PerThread() float64 {
	if m.Threads == 0 {
		return 0
	}
	return m.Cost / float64(m.Threads)
}

// TaskUsage is what the settled threads decided for one model at one
// reasoning level used in all: how many there were, their tokens and
// their priced cost.
type TaskUsage struct {
	// Model is the decided model, and Level the pin's reasoning level.
	Model   string
	Level   string
	Threads int
	Tokens  usage.Tokens
	Cost    float64
}

const (
	// Thread listing limits.
	threadLimit    = 50
	maxThreadLimit = 500
	// threadRequests is how many requests a thread detail carries.
	threadRequests = 200
	// maxBuckets bounds an overview timeline.
	maxBuckets = 10000
)

// Threads lists pinned threads, most recently used first, each with the
// number of requests still in the log. The limit defaults to 50 and is
// capped at 500.
func (s *Store) Threads(ctx context.Context, filter ThreadFilter) (ThreadPage, error) {
	var where conditions
	where.equal("model", filter.Model)
	var page ThreadPage
	if err := s.read.QueryRowContext(ctx, `SELECT COUNT(*) FROM pins`+where.String(), where.args...).Scan(&page.Total); err != nil {
		return ThreadPage{}, err
	}
	threads, err := s.threads(ctx, where.String(), ` ORDER BY last_used_at DESC, key`,
		append(where.args, bounded(filter.Limit, threadLimit, maxThreadLimit), max(filter.Offset, 0))...)
	if err != nil {
		return ThreadPage{}, err
	}
	if err := s.attachDecisions(ctx, threads); err != nil {
		return ThreadPage{}, err
	}
	page.Threads = threads
	return page, nil
}

// Thread returns one thread with its whole event log, oldest first, and its
// latest 200 requests, newest first; nil when the key is unknown.
func (s *Store) Thread(ctx context.Context, key string) (*ThreadDetail, error) {
	threads, err := s.threads(ctx, ` WHERE key = ?`, ``, key, 1, 0)
	if err != nil || len(threads) == 0 {
		return nil, err
	}
	log, err := events(ctx, s.read, ` WHERE key = ? ORDER BY id`, key)
	if err != nil {
		return nil, err
	}
	requests, err := s.requests(ctx, ` WHERE key = ?`, ` ORDER BY at DESC, id DESC`, key, threadRequests)
	if err != nil {
		return nil, err
	}
	agents, err := s.agents(ctx, key)
	if err != nil {
		return nil, err
	}
	detail := &ThreadDetail{Thread: threads[0], Events: log, Requests: requests, Agents: agents}
	if i := slices.IndexFunc(log, func(e Event) bool { return e.Kind == Decision }); i >= 0 {
		decision := log[i]
		detail.Decision = &decision
		if detail.Policy, err = s.Policy(ctx, decision.Basis.Policy); err != nil {
			return nil, err
		}
	}
	return detail, nil
}

// Overview summarizes the period since the given time: the decision log,
// every pinned thread, the requests logged since then and those that failed
// (no upstream status, or status >= 400), and a timeline whose buckets start
// at since, step by bucket and end with the one holding now.
func (s *Store) Overview(ctx context.Context, since time.Time, bucket time.Duration) (Overview, error) {
	from, step := since.UnixMilli(), bucket.Milliseconds()
	if step <= 0 {
		return Overview{}, errors.New("overview bucket must be at least a millisecond")
	}
	count := max(1, (s.now().UnixMilli()-from)/step+1)
	if count > maxBuckets {
		return Overview{}, fmt.Errorf("overview needs %d buckets, more than %d", count, maxBuckets)
	}
	summary, err := s.Summarize(ctx, since)
	if err != nil {
		return Overview{}, err
	}
	overview := Overview{Summary: summary, Timeline: make([]Bucket, count)}
	for i := range overview.Timeline {
		overview.Timeline[i] = Bucket{
			Start: time.UnixMilli(from + int64(i)*step), Requests: map[string]int{}, Decisions: map[string]int{},
		}
	}
	if err := s.read.QueryRowContext(ctx, `SELECT COUNT(*) FROM pins`).Scan(&overview.Threads); err != nil {
		return Overview{}, err
	}
	if err := s.read.QueryRowContext(ctx,
		`SELECT COUNT(*), COUNT(*) FILTER (WHERE status = 0 OR status >= 400), COALESCE(SUM(cost_usd), 0)
		FROM requests WHERE at >= ?`, from).
		Scan(&overview.Requests, &overview.Errors, &overview.Cost); err != nil {
		return Overview{}, err
	}
	if overview.Models, err = s.modelCosts(ctx, from); err != nil {
		return Overview{}, err
	}
	if overview.Queue, err = s.QueueWaits(ctx, since); err != nil {
		return Overview{}, err
	}
	if err := s.tally(ctx, overview.Timeline, func(b *Bucket) map[string]int { return b.Requests },
		`SELECT (at - ?1) / ?2, model, COUNT(*) FROM requests WHERE at >= ?1 GROUP BY 1, 2`, from, step); err != nil {
		return Overview{}, err
	}
	if err := s.tally(ctx, overview.Timeline, func(b *Bucket) map[string]int { return b.Decisions },
		`SELECT (at - ?1) / ?2, model, COUNT(*) FROM events WHERE kind = ?3 AND at >= ?1 GROUP BY 1, 2`,
		from, step, Decision); err != nil {
		return Overview{}, err
	}
	return overview, nil
}

// modelCosts groups the requests since from by model, costliest first.
func (s *Store) modelCosts(ctx context.Context, from int64) ([]ModelCost, error) {
	rows, err := s.read.QueryContext(ctx, `SELECT model, COUNT(DISTINCT key) FILTER (WHERE key != ''), COUNT(*),
		SUM(input_tokens), SUM(output_tokens), SUM(cache_read_tokens), SUM(cache_write_tokens),
		COALESCE(SUM(cost_usd), 0)
		FROM requests WHERE at >= ? GROUP BY model ORDER BY 8 DESC, 3 DESC, model`, from)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	models := []ModelCost{}
	for rows.Next() {
		var m ModelCost
		if err := rows.Scan(&m.Model, &m.Threads, &m.Requests, &m.Tokens.Input, &m.Tokens.Output,
			&m.Tokens.CacheRead, &m.Tokens.CacheWrite, &m.Cost); err != nil {
			return nil, err
		}
		models = append(models, m)
	}
	return models, rows.Err()
}

// TaskUsages sums, per decided model and level, the logged usage of the
// threads pinned at or after since whose last logged request is older
// than settled: each thread's whole usage, wherever its requests went, cut
// down to at most limit tokens so a marathon session weighs like a long
// one. Threads without a logged request, threads an operator re-pinned or
// the router decided again, and threads containing fixed-mode requests are
// left out; their usage does not describe the automatic decision. Unpriced requests add their tokens
// but no cost.
func (s *Store) TaskUsages(ctx context.Context, since, settled time.Time, limit int64) ([]TaskUsage, error) {
	rows, err := s.read.QueryContext(ctx, `WITH threads AS (
			SELECT pins.decided, pins.decided_level AS level, SUM(requests.input_tokens) AS input,
				SUM(requests.output_tokens) AS output, SUM(requests.cache_read_tokens) AS cache_read,
				SUM(requests.cache_write_tokens) AS cache_write, COALESCE(SUM(requests.cost_usd), 0) AS cost
			FROM pins JOIN requests USING (key)
			WHERE pins.created_at >= ?1
				AND NOT EXISTS (SELECT 1 FROM events WHERE events.key = pins.key AND events.kind IN (?4, ?5))
			GROUP BY pins.key HAVING MAX(requests.at) < ?2 AND MAX(requests.reason = 'fixed') = 0),
		cut AS (SELECT *, MIN(1.0, ?3 * 1.0 / MAX(input + output + cache_read + cache_write, 1)) AS share
			FROM threads)
		SELECT decided, level, COUNT(*), CAST(SUM(input * share) AS INTEGER), CAST(SUM(output * share) AS INTEGER),
			CAST(SUM(cache_read * share) AS INTEGER), CAST(SUM(cache_write * share) AS INTEGER), SUM(cost * share)
		FROM cut GROUP BY decided, level ORDER BY decided, level`,
		since.UnixMilli(), settled.UnixMilli(), limit, Override, Redecision)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	usages := []TaskUsage{}
	for rows.Next() {
		var u TaskUsage
		if err := rows.Scan(&u.Model, &u.Level, &u.Threads, &u.Tokens.Input, &u.Tokens.Output, &u.Tokens.CacheRead,
			&u.Tokens.CacheWrite, &u.Cost); err != nil {
			return nil, err
		}
		usages = append(usages, u)
	}
	return usages, rows.Err()
}

// tally adds the per-model counts of a (slot, model, count) query to the
// timeline; rows beyond its last bucket arrived after it was laid out.
func (s *Store) tally(ctx context.Context, timeline []Bucket, counts func(*Bucket) map[string]int, query string, args ...any) error {
	rows, err := s.read.QueryContext(ctx, query, args...)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var slot int64
		var model string
		var count int
		if err := rows.Scan(&slot, &model, &count); err != nil {
			return err
		}
		if slot < int64(len(timeline)) {
			counts(&timeline[slot])[model] += count
		}
	}
	return rows.Err()
}

// threads reads a page of pinned threads with the usage of their logged
// requests: rest follows the FROM clause, order sorts the page and the last
// two arguments are its size and offset.
func (s *Store) threads(ctx context.Context, rest, order string, args ...any) ([]Thread, error) {
	rows, err := s.read.QueryContext(ctx, `WITH page AS (SELECT key, model, level, decided, failed_over, failed_from,
			upgraded, created_at, last_used_at, parent, last_request_at, compacted, turns
			FROM pins`+rest+order+` LIMIT ? OFFSET ?)
		SELECT page.*, COUNT(requests.id), COALESCE(SUM(requests.input_tokens), 0),
			COALESCE(SUM(requests.output_tokens), 0), COALESCE(SUM(requests.cache_read_tokens), 0),
			COALESCE(SUM(requests.cache_write_tokens), 0), COALESCE(SUM(requests.cost_usd), 0)
		FROM page LEFT JOIN requests USING (key) GROUP BY page.key`+order, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	threads := []Thread{}
	for rows.Next() {
		var t Thread
		var created, used, requested int64
		if err := rows.Scan(&t.Key, &t.Model, &t.Level, &t.Decided, &t.FailedOver, &t.FailedFrom, &t.Upgraded,
			&created, &used, &t.Parent, &requested, &t.Compacted, &t.Turns, &t.Requests, &t.Tokens.Input,
			&t.Tokens.Output, &t.Tokens.CacheRead, &t.Tokens.CacheWrite, &t.Cost); err != nil {
			return nil, err
		}
		t.CreatedAt = time.UnixMilli(created)
		t.LastUsedAt = time.UnixMilli(used)
		t.LastRequest = time.UnixMilli(requested)
		threads = append(threads, t)
	}
	return threads, rows.Err()
}

// attachDecisions sets the decision that created each thread's pin.
func (s *Store) attachDecisions(ctx context.Context, threads []Thread) error {
	if len(threads) == 0 {
		return nil
	}
	index := make(map[string]int, len(threads))
	args := []any{Decision}
	for i, t := range threads {
		index[t.Key] = i
		args = append(args, t.Key)
	}
	decisions, err := events(ctx, s.read,
		` WHERE kind = ? AND key IN (?`+strings.Repeat(", ?", len(threads)-1)+`) ORDER BY id`, args...)
	if err != nil {
		return err
	}
	for _, decision := range decisions {
		if t := &threads[index[decision.Key]]; t.Decision == nil {
			t.Decision = &decision
		}
	}
	return nil
}

// events reads decision log entries through q; rest follows the FROM clause.
func events(ctx context.Context, q queryer, rest string, args ...any) ([]Event, error) {
	rows, err := q.QueryContext(ctx, `SELECT id, at, key, kind, model, previous, decider, choice,
		probabilities, escalated, level, levels, latency_ms, attempts, basis, prompt, first_prompt, detail,
		(SELECT json_object('at', reviews.at, 'model', reviews.model, 'level', reviews.level, 'note', reviews.note)
			FROM reviews WHERE reviews.event = events.id)
		FROM events`+rest, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	events := []Event{}
	for rows.Next() {
		var e Event
		var at, latency int64
		var probabilities, levels, attempts, basis string
		var review sql.NullString
		if err := rows.Scan(&e.ID, &at, &e.Key, &e.Kind, &e.Model, &e.Previous, &e.Decider,
			&e.Choice, &probabilities, &e.Escalated, &e.Level, &levels, &latency, &attempts, &basis, &e.Prompt,
			&e.FirstPrompt, &e.Detail, &review); err != nil {
			return nil, err
		}
		if review.Valid {
			if e.Review, err = readReview(e.ID, review.String); err != nil {
				return nil, err
			}
		}
		for _, column := range []struct {
			text  string
			value any
		}{{probabilities, &e.Probabilities}, {levels, &e.Levels}, {attempts, &e.Attempts}, {basis, &e.Basis}} {
			if err := json.Unmarshal([]byte(column.text), column.value); err != nil {
				return nil, err
			}
		}
		e.At = time.UnixMilli(at)
		e.Latency = time.Duration(latency) * time.Millisecond
		events = append(events, e)
	}
	return events, rows.Err()
}
