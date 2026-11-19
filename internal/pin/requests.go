package pin

import (
	"context"
	"database/sql"
	"strings"
	"time"

	"github.com/MarioGK/AI-Router/internal/usage"
)

// RequestLog is one request for the virtual model the router handled.
// Explicit models pass through unlogged. It never holds request or response
// text.
type RequestLog struct {
	ID int64
	At time.Time
	// Key is the pin key; empty when the request was not pinned.
	Key string
	// Kind is the endpoint kind: messages or count_tokens.
	Kind string
	// Requested is the virtual model the client asked for; Model is the one
	// sent to CLIProxy, without the effort suffix, and Effort that suffix
	// (empty when the client's own effort applied).
	Requested string
	Model     string
	Effort    string
	// Reason is how the model was chosen: a route.Reason* value.
	Reason string
	// Status is the upstream status; 0 when CLIProxy was unreachable.
	Status   int
	Failover bool
	// Latency runs from the request's arrival until the response finished
	// streaming, so it includes Queue (and the routing decision before
	// it): the time CLIProxy took is Latency less both.
	Latency time.Duration
	// Queue is the time the request waited for a concurrency slot, summed
	// over every gate it queued at; zero for a model without a cap.
	Queue time.Duration
	// Error is a short error summary, never request text.
	Error string
	// Tokens is the usage the response reported; Cost prices it at the
	// model's list price in US dollars, when Priced (the model has a price).
	Tokens usage.Tokens
	Cost   float64
	Priced bool
	// TaskCost is the pinned thread's cost through this request, read back
	// from the log; zero for a request outside a thread.
	TaskCost float64
}

// RequestFilter selects logged requests. With AfterID the result holds only
// newer requests, oldest first (a live tail); otherwise newest first.
type RequestFilter struct {
	Search  string
	Status  string
	Model   string
	Key     string
	AfterID int64
	Limit   int
}

const (
	// Request listing limits.
	requestLimit    = 100
	maxRequestLimit = 1000
	// pruneBatch bounds each delete or update so the writer is never held
	// long enough to delay routing.
	pruneBatch = 1000
)

// LogRequest records one handled request. A zero At means now.
func (s *Store) LogRequest(ctx context.Context, request RequestLog) error {
	at := request.At
	if at.IsZero() {
		at = s.now()
	}
	cost := sql.NullFloat64{Float64: request.Cost, Valid: request.Priced}
	_, err := s.write.ExecContext(ctx, `INSERT INTO requests
		(at, key, kind, requested, model, effort, reason, status, failover, latency_ms, queue_ms, error,
		 input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		at.UnixMilli(), request.Key, request.Kind, request.Requested, request.Model, request.Effort,
		request.Reason, request.Status, request.Failover, request.Latency.Milliseconds(), request.Queue.Milliseconds(), request.Error,
		request.Tokens.Input, request.Tokens.Output, request.Tokens.CacheRead, request.Tokens.CacheWrite, cost)
	return err
}

// Requests lists logged requests, newest first by arrival, or after
// AfterID in log order. The limit defaults to 100 and is capped at 1000.
func (s *Store) Requests(ctx context.Context, filter RequestFilter) ([]RequestLog, error) {
	var where conditions
	where.equal("model", filter.Model)
	where.equal("key", filter.Key)
	switch filter.Status {
	case "errors":
		where.add("(status = 0 OR status >= 400 OR error != '')")
	case "success":
		where.add("(status != 0 AND status < 400 AND error = '')")
	}
	if strings.TrimSpace(filter.Search) != "" {
		var prompts conditions
		prompts.add("kind = ?", Decision)
		if err := prompts.search("id", filter.Search); err != nil {
			return nil, err
		}
		where.add("key IN (SELECT key FROM events"+prompts.String()+")", prompts.args...)
	}
	source, order := ``, ` ORDER BY at DESC, id DESC`
	if filter.AfterID > 0 {
		// A tail walks the rowids after AfterID; a filter index would visit
		// every older match first.
		source, order = ` NOT INDEXED`, ` ORDER BY id`
		where.add("id > ?", filter.AfterID)
	}
	return s.requests(ctx, source+where.String(), order,
		append(where.args, bounded(filter.Limit, requestLimit, maxRequestLimit))...)
}

// Prune deletes logged requests older than before and reports how many.
// Pins and the decision log are kept; PrunePrompts thins old decisions.
func (s *Store) Prune(ctx context.Context, before time.Time) (int64, error) {
	var total int64
	for {
		result, err := s.write.ExecContext(ctx,
			`DELETE FROM requests WHERE id IN (SELECT id FROM requests WHERE at < ? LIMIT ?)`,
			before.UnixMilli(), pruneBatch)
		if err != nil {
			return total, err
		}
		deleted, err := result.RowsAffected()
		if err != nil {
			return total, err
		}
		total += deleted
		if deleted < pruneBatch {
			return total, nil
		}
	}
}

// requests reads a page of logged requests: rest follows the FROM clause,
// order sorts the page and the last argument is its size. Each request
// carries its thread's cost through it, summed only for the page's threads.
func (s *Store) requests(ctx context.Context, rest, order string, args ...any) ([]RequestLog, error) {
	rows, err := s.read.QueryContext(ctx, `WITH page AS (SELECT id, at, key, kind, requested, model, effort,
			reason, status, failover, latency_ms, queue_ms, error, input_tokens, output_tokens, cache_read_tokens,
			cache_write_tokens, cost_usd FROM requests`+rest+order+` LIMIT ?),
		running AS (SELECT id, SUM(cost_usd) OVER (PARTITION BY key ORDER BY id) AS task FROM requests
			WHERE key IN (SELECT key FROM page WHERE key != ''))
		SELECT page.*, COALESCE(running.task, 0) FROM page LEFT JOIN running USING (id)`+order, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	requests := []RequestLog{}
	for rows.Next() {
		var r RequestLog
		var at, latency, queue int64
		var cost sql.NullFloat64
		if err := rows.Scan(&r.ID, &at, &r.Key, &r.Kind, &r.Requested, &r.Model, &r.Effort, &r.Reason,
			&r.Status, &r.Failover, &latency, &queue, &r.Error, &r.Tokens.Input, &r.Tokens.Output,
			&r.Tokens.CacheRead, &r.Tokens.CacheWrite, &cost, &r.TaskCost); err != nil {
			return nil, err
		}
		r.Cost, r.Priced = cost.Float64, cost.Valid
		r.At = time.UnixMilli(at)
		r.Latency = time.Duration(latency) * time.Millisecond
		r.Queue = time.Duration(queue) * time.Millisecond
		requests = append(requests, r)
	}
	return requests, rows.Err()
}
