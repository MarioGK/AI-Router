package pin

import (
	"context"
	"time"
)

// redecisionMigration keeps, per pin, the session a subagent belongs to and
// what tells whether the thread's prompt cache is cold. Existing pins count
// their last use as their last request and start counting requests anew.
const redecisionMigration = `
ALTER TABLE pins ADD COLUMN parent TEXT NOT NULL DEFAULT '';
ALTER TABLE pins ADD COLUMN last_request_at INTEGER NOT NULL DEFAULT 0;
ALTER TABLE pins ADD COLUMN compacted INTEGER NOT NULL DEFAULT 0;
ALTER TABLE pins ADD COLUMN turns INTEGER NOT NULL DEFAULT 0;
UPDATE pins SET last_request_at = last_used_at;
CREATE INDEX pins_parent ON pins (parent) WHERE parent != '';
`

// Seen records a request of a pinned thread that used its prompt cache: it
// is now the thread's last request, which was no compaction, and counts
// towards the requests since the thread's last decision. The thread's last
// use is updated at most once a minute, as Touch does.
func (s *Store) Seen(ctx context.Context, key string) error {
	now := s.now().UnixMilli()
	_, err := s.write.ExecContext(ctx, `UPDATE pins SET last_request_at = ?1, compacted = 0, turns = turns + 1,
		last_used_at = CASE WHEN last_used_at < ?2 THEN ?1 ELSE last_used_at END WHERE key = ?3`,
		now, now-time.Minute.Milliseconds(), key)
	return err
}

// Compacted records that a thread's last request was a compaction, after
// which its prompt cache is cold.
func (s *Store) Compacted(ctx context.Context, key string) error {
	_, err := s.write.ExecContext(ctx, `UPDATE pins SET last_request_at = ?, compacted = 1 WHERE key = ?`,
		s.now().UnixMilli(), key)
	return err
}

// Redecide moves a pin to model at level on a new decision, unless the pin
// changed since from was read, including another request of the thread
// since then. It spends neither the failover nor the capability upgrade,
// starts the thread's request count anew and marks its cache warm. A thread
// that needs images only moves when supportsImages says the destination can
// read them. The event is logged whenever the pin was decided again, also
// when the decision kept its model and level.
func (s *Store) Redecide(ctx context.Context, from Pin, model, level string, supportsImages bool, event Event) (*Pin, error) {
	return s.change(ctx, event, `UPDATE pins SET model = ?1, level = ?2, updated_at = ?3, last_request_at = ?3,
		compacted = 0, turns = 0
		WHERE key = ?4 AND model = ?5 AND level = ?6 AND last_request_at = ?7 AND compacted = ?8 AND
		(?9 OR (upgraded = 0 AND NOT EXISTS (SELECT 1 FROM thread_image_requirements WHERE key = pins.key)))`,
		func(now int64) []any {
			return []any{model, level, now, from.Key, from.Model, from.Level, from.LastRequest.UnixMilli(),
				from.Compacted, supportsImages}
		})
}

// Overridden reports whether an operator ever re-pinned the thread.
func (s *Store) Overridden(ctx context.Context, key string) (bool, error) {
	var overridden bool
	err := s.write.QueryRowContext(ctx, `SELECT EXISTS (SELECT 1 FROM events WHERE key = ? AND kind = ?)`,
		key, Override).Scan(&overridden)
	return overridden, err
}

// Moves lists the redecisions and operator re-pins of a thread, oldest
// first, with the model each moved from and to.
func (s *Store) Moves(ctx context.Context, key string) ([]Event, error) {
	rows, err := s.write.QueryContext(ctx, `SELECT kind, previous, model FROM events
		WHERE key = ? AND kind IN (?, ?) ORDER BY id`, key, Redecision, Override)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var moves []Event
	for rows.Next() {
		e := Event{Key: key}
		if err := rows.Scan(&e.Kind, &e.Previous, &e.Model); err != nil {
			return nil, err
		}
		moves = append(moves, e)
	}
	return moves, rows.Err()
}

// agents lists the pin keys of a session's subagents, oldest first.
func (s *Store) agents(ctx context.Context, parent string) ([]string, error) {
	rows, err := s.read.QueryContext(ctx, `SELECT key FROM pins WHERE parent = ? AND parent != ''
		ORDER BY created_at, key`, parent)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var keys []string
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err != nil {
			return nil, err
		}
		keys = append(keys, key)
	}
	return keys, rows.Err()
}
