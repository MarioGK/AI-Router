package pin

import (
	"context"
	"time"
)

// queueWaitMigration adds the time each request waited for a concurrency
// slot, apart from latency_ms, which runs from the request's arrival and so
// includes it. Requests logged before it have no recorded wait: zero.
const queueWaitMigration = `
ALTER TABLE requests ADD COLUMN queue_ms INTEGER NOT NULL DEFAULT 0;
`

// QueueWait is how long the requests of one model waited for a concurrency
// slot in a period. Requests counts all of its logged requests, Queued the
// ones that waited at all; Mean and P95 are over the ones that did.
type QueueWait struct {
	Requests int
	Queued   int
	Mean     time.Duration
	P95      time.Duration
}

// QueueWaits reports, per model, the queue wait of the requests logged
// since the given time.
func (s *Store) QueueWaits(ctx context.Context, since time.Time) (map[string]QueueWait, error) {
	rows, err := s.read.QueryContext(ctx, `SELECT model, queue_ms FROM requests WHERE at >= ? ORDER BY model, queue_ms`,
		since.UnixMilli())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	waits := map[string]QueueWait{}
	queued := map[string][]int64{}
	for rows.Next() {
		var model string
		var ms int64
		if err := rows.Scan(&model, &ms); err != nil {
			return nil, err
		}
		w := waits[model]
		w.Requests++
		waits[model] = w
		if ms > 0 {
			queued[model] = append(queued[model], ms)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for model, list := range queued {
		var sum int64
		for _, ms := range list {
			sum += ms
		}
		w := waits[model]
		w.Queued = len(list)
		w.Mean = time.Duration(sum/int64(len(list))) * time.Millisecond
		// The list is sorted ascending.
		w.P95 = time.Duration(list[len(list)*95/100]) * time.Millisecond
		waits[model] = w
	}
	return waits, nil
}
