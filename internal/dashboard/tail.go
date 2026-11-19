package dashboard

import (
	"context"
	"slices"
	"strconv"

	"github.com/MarioGK/AI-Router/internal/pin"
)

const (
	// tailSize is how many requests a live table keeps.
	tailSize = 50
	// tailBatch caps one poll; a fuller poll means the table fell behind and
	// starts over from the newest requests.
	tailBatch = 200
)

// tail follows the newest requests that match a filter, newest first.
type tail struct {
	source Source
	filter pin.RequestFilter
	// rows are the requests on show.
	rows []pin.RequestLog
	// after is the newest request seen, kept or not.
	after int64
	// fresh is the newest request seen before the latest poll: rows above
	// it are new.
	fresh int64
	err   error
	// revision reloads tables after deletion, including when SQLite reuses IDs.
	revision func() uint64
	seen     uint64
}

// load takes the newest requests.
func (t *tail) load(ctx context.Context) {
	if t.revision != nil {
		t.seen = t.revision()
	}
	filter := t.filter
	filter.Limit = tailSize
	requests, err := t.source.Requests(ctx, filter)
	if t.err = err; err != nil {
		return
	}
	t.rows = t.rows[:0]
	t.after = 0
	for _, request := range requests {
		t.after = max(t.after, request.ID)
		if len(t.rows) < tailSize {
			t.rows = append(t.rows, request)
		}
	}
	t.fresh = t.after
}

// poll adds the requests logged since the last look and reports whether
// the table changed.
func (t *tail) poll(ctx context.Context) bool {
	if t.revision != nil && t.revision() != t.seen {
		t.load(ctx)
		return true
	}
	if t.after == 0 || t.err != nil {
		// Nothing seen yet or the last look failed: look afresh.
		shown, failed := len(t.rows), t.err != nil
		t.load(ctx)
		return len(t.rows) > 0 || shown > 0 || failed != (t.err != nil)
	}
	filter := t.filter
	filter.AfterID, filter.Limit = t.after, tailBatch
	requests, err := t.source.Requests(ctx, filter)
	if err != nil {
		t.err = err
		return true
	}
	if len(requests) == tailBatch {
		t.load(ctx)
		return true
	}
	t.fresh = t.after
	changed := false
	for _, request := range requests {
		t.after = max(t.after, request.ID)
		t.rows = slices.Insert(t.rows, 0, request)
		changed = true
	}
	t.rows = t.rows[:min(len(t.rows), tailSize)]
	return changed
}

// requestTable is what a request table shows.
type requestTable struct {
	Rows []pin.RequestLog
	// Fresh marks the newest request seen before the latest poll; rows
	// above it just arrived. Zero marks none.
	Fresh int64
	Err   error
}

// arrived reports whether a row just arrived.
func (v requestTable) arrived(r pin.RequestLog) bool {
	return v.Fresh > 0 && r.ID > v.Fresh
}

// view is the tail as a table.
func (t *tail) view() requestTable {
	return requestTable{Rows: t.rows, Fresh: t.fresh, Err: t.err}
}

// requestColumns picks the optional columns of a request table.
type requestColumns struct {
	Thread, Kind, Queue, Cost, Task, Error bool
}

// The request tables: the overview's feed, the log and a thread's history.
var (
	feedColumns   = requestColumns{Cost: true}
	logColumns    = requestColumns{Thread: true, Kind: true, Queue: true, Cost: true, Task: true, Error: true}
	threadColumns = requestColumns{Kind: true, Queue: true, Cost: true, Task: true, Error: true}
)

// span is the number of columns, for a cell that fills a row.
func (c requestColumns) span() string {
	columns := 5
	for _, shown := range []bool{c.Thread, c.Kind, c.Queue, c.Cost, c.Task, c.Error} {
		if shown {
			columns++
		}
	}
	return strconv.Itoa(columns)
}
