// Package pin persists which model each thread is pinned to, and the
// reasoning level it runs at, plus a log of routing decisions and a log of
// handled requests. A pin is permanent: its model changes at most once
// through failover and at most once through a capability upgrade, and
// otherwise, with its level, only when the router migrates it off a model
// the configuration dropped, decides the thread again where its prompt
// cache is cold, or an operator overrides it. Only the request
// log is pruned, and the prompts and raw replies kept with old decisions.
package pin

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	_ "modernc.org/sqlite"

	"github.com/MarioGK/AI-Router/internal/decide"
)

// Pin is the model a thread is pinned to. FailedOver records its one
// failover; Upgraded records that the thread needs an image-capable model.
// A migration or an override preserves both.
type Pin struct {
	Key   string
	Model string
	// Level is the reasoning level the chooser picked for the thread; empty
	// when it picked none, and the model runs at its fixed effort.
	Level      string
	Decided    string
	FailedOver bool
	// FailedFrom is the model the failover left; empty before one.
	FailedFrom string
	Upgraded   bool
	CreatedAt  time.Time
	// Parent is the pin key of the session whose subagent the thread is;
	// empty for a session's main thread.
	Parent string
	// LastRequest is when the thread last sent a request that used its
	// prompt cache, Compacted whether that request was a compaction, and
	// Turns how many requests the thread made since its last decision.
	LastRequest time.Time
	Compacted   bool
	Turns       int
}

// Event kinds recorded in the decision log. A Migration moved a pin off a
// model the configuration no longer has; a Redecision decided the thread
// again on a new prompt where its prompt cache was cold; an Override is an
// operator re-pinning the thread.
const (
	Decision          = "decision"
	Failover          = "failover"
	CapabilityUpgrade = "capability_upgrade"
	Migration         = "migration"
	Redecision        = "redecision"
	Override          = "override"
)

// Event is one entry of the decision log. ID and At are set when read back
// from the store. A decision keeps every decider's attempt, what it was
// based on and, when prompts are recorded, the decider input; Detail is a
// short note on the other kinds.
type Event struct {
	ID            int64
	At            time.Time
	Key           string
	Kind          string
	Model         string
	Previous      string
	Decider       string
	Choice        string
	Probabilities map[string]float64
	Escalated     bool
	// Level is the reasoning level decided, and Levels the
	// deciding verdict's probability per level.
	Level    string
	Levels   map[string]float64
	Latency  time.Duration
	Attempts []decide.Attempt
	Basis    Basis
	// Prompt is the decider input, the latest user text as judged, and
	// FirstPrompt the thread's first prompt as the user wrote it; both are
	// kept with record_prompts.
	Prompt      string
	FirstPrompt string
	Detail      string
	// Review is an operator's judgement of a decision; nil until one.
	Review *Review
}

// Basis is what a decision was made from, kept to audit decisions and
// improve the routing policy.
type Basis struct {
	// Kind is the endpoint kind of the first request.
	Kind string `json:"kind,omitempty"`
	// UserAgent names the Claude Code version.
	UserAgent string `json:"user_agent,omitempty"`
	// Candidates are the models offered, in ladder order; Withheld are the
	// candidates left out because CLIProxy does not serve them or every
	// account that serves them is at its limit.
	Candidates []string `json:"candidates,omitempty"`
	Withheld   []string `json:"withheld,omitempty"`
	// Spilled is the model decided on when it was saturated and the thread
	// went to its spill model instead.
	Spilled string `json:"spilled,omitempty"`
	// Parent is the pin key of the session a subagent's thread belongs to.
	Parent string `json:"parent,omitempty"`
	// ExploreEligible is set when the decision was borderline enough to be
	// explored one ladder step cheaper; Explore when it was, from
	// ExploredFrom, the step decided. Propensity is the probability the
	// action taken had: the explore share for an explored decision, one
	// less it for one that kept its step.
	ExploreEligible bool    `json:"explore_eligible,omitempty"`
	Explore         bool    `json:"explore,omitempty"`
	Propensity      float64 `json:"propensity,omitempty"`
	ExploredFrom    string  `json:"explored_from,omitempty"`
	// Costs is how the cost of getting the task done was weighed, when it
	// was.
	Costs *decide.Costs `json:"costs,omitempty"`
	// Prior is what the labelled decisions with the most similar first
	// prompts suggested, and how much it weighed against the deciders,
	// when any were found.
	Prior     *decide.Prior `json:"prior,omitempty"`
	Threshold float64       `json:"threshold"`
	Doubt     string        `json:"doubt,omitempty"`
	Images    bool          `json:"images,omitempty"`
	// PromptChars is the length of the user text judged, before it was
	// cut to fit (Truncated).
	PromptChars int  `json:"prompt_chars,omitempty"`
	Truncated   bool `json:"truncated,omitempty"`
	// Policy fingerprints the instructions, deciders and candidates.
	Policy string `json:"policy,omitempty"`
	// Version is the router build.
	Version string `json:"version,omitempty"`
}

// Consulted reports whether a decider in doubt was asked for a second
// opinion.
func (e Event) Consulted() bool {
	for _, attempt := range e.Attempts {
		if attempt.Outcome == decide.Doubted {
			return true
		}
	}
	return false
}

// Store is the SQLite pin store. Writes and the routing path's reads go
// through one connection, which serializes them and reads its own writes;
// dashboard and stats reads go through a pool of read-only connections,
// which WAL runs alongside it, so no dashboard read ever holds up routing.
type Store struct {
	write *sql.DB
	read  *sql.DB
	now   func() time.Time
}

// readers bounds the read-only connections.
const readers = 4

var migrations = []string{`
CREATE TABLE pins (
	key          TEXT PRIMARY KEY,
	client       TEXT NOT NULL,
	model        TEXT NOT NULL,
	decided      TEXT NOT NULL,
	failed_over  INTEGER NOT NULL DEFAULT 0,
	upgraded     INTEGER NOT NULL DEFAULT 0,
	created_at   INTEGER NOT NULL,
	updated_at   INTEGER NOT NULL,
	last_used_at INTEGER NOT NULL
);
CREATE TABLE events (
	id            INTEGER PRIMARY KEY,
	at            INTEGER NOT NULL,
	key           TEXT NOT NULL,
	client        TEXT NOT NULL,
	kind          TEXT NOT NULL,
	model         TEXT NOT NULL,
	previous      TEXT NOT NULL,
	decider       TEXT NOT NULL,
	choice        TEXT NOT NULL,
	probabilities TEXT NOT NULL,
	escalated     INTEGER NOT NULL,
	latency_ms    INTEGER NOT NULL,
	detail        TEXT NOT NULL
);
CREATE INDEX events_at ON events (at);
`, `
CREATE TABLE requests (
	id         INTEGER PRIMARY KEY,
	at         INTEGER NOT NULL,
	client     TEXT NOT NULL,
	key        TEXT NOT NULL,
	kind       TEXT NOT NULL,
	requested  TEXT NOT NULL,
	model      TEXT NOT NULL,
	reason     TEXT NOT NULL,
	status     INTEGER NOT NULL,
	failover   INTEGER NOT NULL,
	latency_ms INTEGER NOT NULL,
	error      TEXT NOT NULL
);
CREATE INDEX requests_at ON requests (at);
CREATE INDEX requests_key ON requests (key, at);
CREATE INDEX requests_client ON requests (client, at);
CREATE INDEX requests_model ON requests (model, at);
CREATE INDEX events_key ON events (key);
CREATE INDEX pins_last_used ON pins (last_used_at);
`, `
ALTER TABLE events ADD COLUMN doubts TEXT NOT NULL DEFAULT '[]';
`, `
ALTER TABLE requests ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN cache_read_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN cache_write_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN cost_usd REAL;
`, `
DELETE FROM requests WHERE reason = 'explicit';
`, `
ALTER TABLE events ADD COLUMN attempts TEXT NOT NULL DEFAULT '[]';
ALTER TABLE events ADD COLUMN basis TEXT NOT NULL DEFAULT '{}';
ALTER TABLE events ADD COLUMN prompt TEXT NOT NULL DEFAULT '';
UPDATE events SET
	attempts = (
		SELECT json_group_array(json(attempt)) FROM (
			SELECT json_object('decider', value->>'decider', 'outcome', 'doubted', 'choice', value->>'choice',
				'probabilities', CASE WHEN value->>'decider' = events.decider THEN json(events.probabilities) END,
				'above', value->>'above', 'latency_ms', 0) AS attempt
			FROM json_each(events.doubts)
			UNION ALL
			SELECT json_object('decider', events.decider, 'outcome', 'decided', 'choice', events.choice,
				'probabilities', json(events.probabilities), 'latency_ms', events.latency_ms)
			WHERE events.decider != 'default'
				AND NOT EXISTS (SELECT 1 FROM json_each(events.doubts) WHERE value->>'decider' = events.decider)
		)),
	basis = json_object('threshold', 0.1)
WHERE kind = 'decision';
ALTER TABLE events DROP COLUMN doubts;
`, `
CREATE TABLE policies (
	fingerprint TEXT PRIMARY KEY,
	at          INTEGER NOT NULL,
	policy      TEXT NOT NULL
);
CREATE INDEX events_kind ON events (kind, id);
`, `
DELETE FROM policies;
`, `
DELETE FROM requests WHERE client != 'claude-code';
DELETE FROM events WHERE client != 'claude-code';
DELETE FROM pins WHERE client != 'claude-code';
ALTER TABLE pins ADD COLUMN level TEXT NOT NULL DEFAULT '';
ALTER TABLE pins ADD COLUMN decided_level TEXT NOT NULL DEFAULT '';
ALTER TABLE pins ADD COLUMN failed_from TEXT NOT NULL DEFAULT '';
UPDATE pins SET failed_from = COALESCE((SELECT previous FROM events
	WHERE events.key = pins.key AND events.kind = 'failover' ORDER BY id DESC LIMIT 1), '')
WHERE failed_over = 1;
ALTER TABLE events ADD COLUMN level TEXT NOT NULL DEFAULT '';
ALTER TABLE events ADD COLUMN levels TEXT NOT NULL DEFAULT '{}';
ALTER TABLE requests ADD COLUMN effort TEXT NOT NULL DEFAULT '';
DROP INDEX requests_client;
ALTER TABLE pins DROP COLUMN client;
ALTER TABLE events DROP COLUMN client;
ALTER TABLE requests DROP COLUMN client;
`, `
ALTER TABLE events ADD COLUMN first_prompt TEXT NOT NULL DEFAULT '';
CREATE TABLE reviews (
	event INTEGER PRIMARY KEY REFERENCES events (id),
	at    INTEGER NOT NULL,
	model TEXT NOT NULL,
	level TEXT NOT NULL,
	note  TEXT NOT NULL
);
`, routingSettingsMigration, promptSearchMigration, meteringRepairMigration, queueWaitMigration, `
CREATE INDEX pins_created ON pins (created_at);
`, redecisionMigration, promptTermsMigration}

// Open opens (and migrates) the store at path.
func Open(path string) (*Store, error) {
	write, err := sql.Open("sqlite", dsn(path, "busy_timeout(5000)", "journal_mode(WAL)", "synchronous(NORMAL)"))
	if err != nil {
		return nil, err
	}
	// One connection serializes writers; every statement is short.
	write.SetMaxOpenConns(1)
	read, err := sql.Open("sqlite", dsn(path, "busy_timeout(5000)", "journal_mode(WAL)", "query_only(1)"))
	if err != nil {
		write.Close()
		return nil, err
	}
	read.SetMaxOpenConns(readers)
	read.SetMaxIdleConns(readers)
	store := &Store{write: write, read: read, now: time.Now}
	if err := store.migrate(); err != nil {
		store.Close()
		return nil, fmt.Errorf("migrate %s: %w", path, err)
	}
	return store, nil
}

// dsn is the connection string of the database at path, running pragmas on
// every connection.
func dsn(path string, pragmas ...string) string {
	return "file:" + path + "?_pragma=" + strings.Join(pragmas, "&_pragma=")
}

// migrate brings the schema up to date, one migration at a time.
func (s *Store) migrate() error {
	var version int
	if err := s.write.QueryRow("PRAGMA user_version").Scan(&version); err != nil {
		return err
	}
	// An older build must not run on a database a newer one migrated: its
	// statements no longer fit the schema.
	if version > len(migrations) {
		return fmt.Errorf("the database is at version %d, newer than this build's %d", version, len(migrations))
	}
	for ; version < len(migrations); version++ {
		if err := s.step(version); err != nil {
			return fmt.Errorf("migration %d: %w", version+1, err)
		}
	}
	return nil
}

// step runs migration n and records it in one transaction, so a failing
// migration leaves the database at the version before it.
func (s *Store) step(n int) error {
	tx, err := s.write.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(migrations[n]); err != nil {
		return err
	}
	if _, err := tx.Exec(fmt.Sprintf("PRAGMA user_version = %d", n+1)); err != nil {
		return err
	}
	return tx.Commit()
}

// Close closes the store.
func (s *Store) Close() error {
	return errors.Join(s.read.Close(), s.write.Close())
}

// Backup writes a consistent snapshot of the database file to path, which
// must not exist, as it stands: it neither migrates the database nor stops a
// router using it.
func Backup(ctx context.Context, database, path string) error {
	// Opening a missing database would create it empty.
	if _, err := os.Stat(database); err != nil {
		return err
	}
	// Claiming path fails when it exists and keeps the copy private; VACUUM
	// INTO fills an empty file.
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	file.Close()
	// A plain connection reads one snapshot alongside a router's writer.
	db, err := sql.Open("sqlite", dsn(database, "busy_timeout(5000)"))
	if err == nil {
		_, err = db.ExecContext(ctx, `VACUUM INTO ?`, path)
		err = errors.Join(err, db.Close())
	}
	if err != nil {
		os.Remove(path)
	}
	return err
}

// Get returns the pin for key, or nil when the thread is not pinned. It reads
// through the writer, so it sees every change and never queues behind the
// dashboard.
func (s *Store) Get(ctx context.Context, key string) (*Pin, error) {
	return get(ctx, s.write, key)
}

// queryer runs reads: a connection pool, or a transaction.
type queryer interface {
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

// get reads the pin for key; nil when the thread is not pinned.
func get(ctx context.Context, q queryer, key string) (*Pin, error) {
	var p Pin
	var created, requested int64
	err := q.QueryRowContext(ctx,
		`SELECT key, model, level, decided, failed_over, failed_from, upgraded, created_at, parent, last_request_at,
			compacted, turns FROM pins WHERE key = ?`, key).
		Scan(&p.Key, &p.Model, &p.Level, &p.Decided, &p.FailedOver, &p.FailedFrom, &p.Upgraded, &created, &p.Parent,
			&requested, &p.Compacted, &p.Turns)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	p.CreatedAt = time.UnixMilli(created)
	p.LastRequest = time.UnixMilli(requested)
	return &p, nil
}

// Create pins a thread unless it is already pinned, and returns the pin that
// holds. The event is logged only when this call created the pin.
func (s *Store) Create(ctx context.Context, p Pin, event Event) (*Pin, error) {
	return s.change(ctx, event, `INSERT OR IGNORE INTO pins
		(key, model, level, decided, decided_level, failed_over, upgraded, created_at, updated_at, last_used_at,
		 parent, last_request_at)
		VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)`,
		func(now int64) []any {
			return []any{p.Key, p.Model, p.Level, p.Model, p.Level, p.Upgraded, now, now, now, p.Parent, now}
		})
}

// Failover moves a pin from one model to another, once. It returns the pin
// that holds afterwards: when another request already failed the thread over
// it carries that request's model. A thread that needs images only moves
// when supportsImages says the destination can read them.
func (s *Store) Failover(ctx context.Context, key, from, to string, supportsImages bool, event Event) (*Pin, error) {
	return s.change(ctx, event, `UPDATE pins SET model = ?, failed_over = 1, failed_from = ?, updated_at = ?
		WHERE key = ? AND model = ? AND failed_over = 0 AND
		(? OR (upgraded = 0 AND NOT EXISTS (SELECT 1 FROM thread_image_requirements WHERE key = pins.key)))`,
		func(now int64) []any { return []any{to, from, now, key, from, supportsImages} })
}

// Upgrade moves a pin to a more capable model, once, when the thread starts
// needing a capability (vision) its model lacks.
func (s *Store) Upgrade(ctx context.Context, key, from, to string, event Event) (*Pin, error) {
	return s.change(ctx, event, `UPDATE pins SET model = ?, upgraded = 1, updated_at = ?
		WHERE key = ? AND model = ? AND upgraded = 0`,
		func(now int64) []any { return []any{to, now, key, from} })
}

// Move moves a pin that is still on from to model at level, spending neither
// its failover nor its capability upgrade: the router migrates pins off
// models the configuration dropped, and operators override them. A thread
// that needs images only moves when supportsImages says the destination
// can read them. The event is logged only when the pin changed.
func (s *Store) Move(ctx context.Context, key, from, model, level string, supportsImages bool, event Event) (*Pin, error) {
	return s.change(ctx, event, `UPDATE pins SET model = ?1, level = ?2, updated_at = ?3
		WHERE key = ?4 AND model = ?5 AND (model != ?1 OR level != ?2) AND
		(?6 OR (upgraded = 0 AND NOT EXISTS (SELECT 1 FROM thread_image_requirements WHERE key = pins.key)))`,
		func(now int64) []any { return []any{model, level, now, key, from, supportsImages} })
}

// change runs statement, which changes at most one pin, in a transaction
// that logs event when it did, and returns the pin that holds afterwards.
func (s *Store) change(ctx context.Context, event Event, statement string, args func(now int64) []any) (*Pin, error) {
	tx, err := s.write.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	now := s.now().UnixMilli()
	result, err := tx.ExecContext(ctx, statement, args(now)...)
	if err != nil {
		return nil, err
	}
	if changed, err := result.RowsAffected(); err != nil {
		return nil, err
	} else if changed == 1 {
		if err := record(ctx, tx, now, event); err != nil {
			return nil, err
		}
	}
	p, err := get(ctx, tx, event.Key)
	if err != nil {
		return nil, err
	}
	if p == nil {
		return nil, fmt.Errorf("pin %s vanished", event.Key)
	}
	return p, tx.Commit()
}

// record appends an event to the decision log.
func record(ctx context.Context, tx *sql.Tx, now int64, e Event) error {
	probabilities, err := json.Marshal(e.Probabilities)
	if err != nil {
		return err
	}
	if e.Levels == nil {
		e.Levels = map[string]float64{}
	}
	levels, err := json.Marshal(e.Levels)
	if err != nil {
		return err
	}
	if e.Attempts == nil {
		e.Attempts = []decide.Attempt{}
	}
	attempts, err := json.Marshal(e.Attempts)
	if err != nil {
		return err
	}
	basis, err := json.Marshal(e.Basis)
	if err != nil {
		return err
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO events
		(at, key, kind, model, previous, decider, choice, probabilities, escalated, level, levels,
		 latency_ms, attempts, basis, prompt, first_prompt, detail)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		now, e.Key, e.Kind, e.Model, e.Previous, e.Decider, e.Choice, string(probabilities),
		e.Escalated, e.Level, string(levels), e.Latency.Milliseconds(), string(attempts), string(basis), e.Prompt,
		e.FirstPrompt, e.Detail)
	return err
}

// Touch records that a thread was used, at most once a minute per pin.
func (s *Store) Touch(ctx context.Context, key string) error {
	now := s.now().UnixMilli()
	_, err := s.write.ExecContext(ctx, `UPDATE pins SET last_used_at = ? WHERE key = ? AND last_used_at < ?`,
		now, key, now-time.Minute.Milliseconds())
	return err
}

// conditions builds a WHERE clause from the filters that are set.
type conditions struct {
	clauses []string
	args    []any
}

// add requires clause with its bound arguments.
func (c *conditions) add(clause string, args ...any) {
	c.clauses = append(c.clauses, clause)
	c.args = append(c.args, args...)
}

// equal matches column against value unless value is empty.
func (c *conditions) equal(column, value string) {
	if value != "" {
		c.add(column+" = ?", value)
	}
}

// String is the WHERE clause; empty when no filter is set.
func (c *conditions) String() string {
	if len(c.clauses) == 0 {
		return ""
	}
	return ` WHERE ` + strings.Join(c.clauses, " AND ")
}

// bounded is limit, or fallback when unset, capped at most.
func bounded(limit, fallback, most int) int {
	if limit <= 0 {
		return fallback
	}
	return min(limit, most)
}
