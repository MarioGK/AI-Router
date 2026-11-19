package pin

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/usage"
)

func open(t *testing.T, path string) *Store {
	t.Helper()
	store, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	return store
}

func TestPinsArePermanentAndSurviveReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	if p, err := store.Get(ctx, "k"); err != nil || p != nil {
		t.Fatalf("Get before create = %+v, %v", p, err)
	}
	created, err := store.Create(ctx, Pin{Key: "k", Model: "glm-5.3"}, Event{Key: "k", Kind: Decision, Model: "glm-5.3", Decider: "jev"})
	if err != nil || created.Model != "glm-5.3" || created.Decided != "glm-5.3" {
		t.Fatalf("Create = %+v, %v", created, err)
	}
	again, err := store.Create(ctx, Pin{Key: "k", Model: "gpt-6.1-sol"}, Event{Key: "k", Kind: Decision, Model: "gpt-6.1-sol"})
	if err != nil || again.Model != "glm-5.3" {
		t.Fatalf("second Create = %+v, %v", again, err)
	}
	store.Close()

	reopened := open(t, path)
	p, err := reopened.Get(ctx, "k")
	if err != nil || p == nil || p.Model != "glm-5.3" {
		t.Fatalf("Get after reopen = %+v, %v", p, err)
	}
	summary, err := reopened.Summarize(ctx, time.Now().Add(-time.Hour))
	if err != nil || summary.Decisions["glm-5.3"] != 1 || summary.Decisions["gpt-6.1-sol"] != 0 || summary.Pins["glm-5.3"] != 1 {
		t.Fatalf("Summarize = %+v, %v", summary, err)
	}
}

func TestFailoverHappensOnceEvenUnderRace(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	if _, err := store.Create(ctx, Pin{Key: "k", Model: "gpt-6.1-sol"}, Event{Key: "k", Kind: Decision, Model: "gpt-6.1-sol"}); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 20 {
		wg.Go(func() {
			p, err := store.Failover(ctx, "k", "gpt-6.1-sol", "glm-5.3", false, Event{Key: "k", Kind: Failover, Model: "glm-5.3", Previous: "gpt-6.1-sol"})
			if err != nil || p.Model != "glm-5.3" || !p.FailedOver || p.FailedFrom != "gpt-6.1-sol" {
				t.Errorf("Failover = %+v, %v", p, err)
			}
		})
	}
	wg.Wait()
	p, err := store.Failover(ctx, "k", "glm-5.3", "deepseek-flash", true, Event{Key: "k", Kind: Failover, Model: "deepseek-flash", Previous: "glm-5.3"})
	if err != nil || p.Model != "glm-5.3" || p.FailedFrom != "gpt-6.1-sol" {
		t.Fatalf("second failover moved the pin: %+v, %v", p, err)
	}
	if p, err := store.Get(ctx, "k"); err != nil || p.FailedFrom != "gpt-6.1-sol" {
		t.Fatalf("Get after the failover = %+v, %v", p, err)
	}
	summary, _ := store.Summarize(ctx, time.Now().Add(-time.Hour))
	if len(summary.Failovers) != 1 || summary.Failovers["gpt-6.1-sol->glm-5.3"] != 1 {
		t.Fatalf("failover events = %+v", summary.Failovers)
	}
}

func TestUpgradeHappensOnceAndIndependentlyOfFailover(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	_, _ = store.Create(ctx, Pin{Key: "k", Model: "glm-5.3"}, Event{Key: "k", Kind: Decision, Model: "glm-5.3"})
	p, err := store.Upgrade(ctx, "k", "glm-5.3", "gpt-6.1-sol", Event{Key: "k", Kind: CapabilityUpgrade, Model: "gpt-6.1-sol", Previous: "glm-5.3"})
	if err != nil || p.Model != "gpt-6.1-sol" || !p.Upgraded || p.FailedOver || p.FailedFrom != "" {
		t.Fatalf("Upgrade = %+v, %v", p, err)
	}
	if p, _ := store.Upgrade(ctx, "k", "gpt-6.1-sol", "other", Event{Key: "k", Kind: CapabilityUpgrade}); p.Model != "gpt-6.1-sol" {
		t.Fatalf("second upgrade moved the pin: %+v", p)
	}
	if p, _ := store.Failover(ctx, "k", "gpt-6.1-sol", "other-vision", true, Event{Key: "k", Kind: Failover}); p.Model != "other-vision" || !p.FailedOver {
		t.Fatalf("failover after upgrade = %+v", p)
	}
}

func TestMoveSpendsNeitherFailoverNorUpgrade(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	if _, err := store.Create(ctx, Pin{Key: "k", Model: "glm-5.3", Level: "high"}, Event{Key: "k", Kind: Decision, Model: "glm-5.3"}); err != nil {
		t.Fatal(err)
	}
	if p, err := store.Move(ctx, "k", "gpt-6.1-sol", "claude-opus-5-5", "", true, Event{Key: "k", Kind: Override}); err != nil || p.Model != "glm-5.3" {
		t.Fatalf("a move from another model = %+v, %v", p, err)
	}
	p, err := store.Move(ctx, "k", "glm-5.3", "gpt-6.1-sol", "xhigh", true, Event{Key: "k", Kind: Migration, Model: "gpt-6.1-sol", Previous: "glm-5.3"})
	if err != nil || p.Model != "gpt-6.1-sol" || p.Level != "xhigh" || p.Decided != "glm-5.3" || p.FailedOver || p.Upgraded {
		t.Fatalf("migration = %+v, %v", p, err)
	}
	if p, err := store.Failover(ctx, "k", "gpt-6.1-sol", "glm-5.3", false, Event{Key: "k", Kind: Failover, Model: "glm-5.3", Previous: "gpt-6.1-sol"}); err != nil || p.Model != "glm-5.3" || p.Level != "xhigh" || !p.FailedOver {
		t.Fatalf("failover after the migration = %+v, %v", p, err)
	}
	for range 2 {
		p, err = store.Move(ctx, "k", "glm-5.3", "claude-opus-5-5", "", true, Event{Key: "k", Kind: Override, Model: "claude-opus-5-5", Previous: "glm-5.3"})
		if err != nil || p.Model != "claude-opus-5-5" || p.Level != "" || !p.FailedOver || p.FailedFrom != "gpt-6.1-sol" {
			t.Fatalf("override = %+v, %v", p, err)
		}
	}
	if p, err := store.Move(ctx, "k", "claude-opus-5-5", "claude-opus-5-5", "", true, Event{Key: "k", Kind: Override}); err != nil || p.Model != "claude-opus-5-5" {
		t.Fatalf("an override to the same pin = %+v, %v", p, err)
	}
	if p, err := store.Upgrade(ctx, "k", "claude-opus-5-5", "gpt-6.1-sol", Event{Key: "k", Kind: CapabilityUpgrade, Model: "gpt-6.1-sol", Previous: "claude-opus-5-5"}); err != nil || p.Model != "gpt-6.1-sol" || !p.Upgraded {
		t.Fatalf("upgrade after the override = %+v, %v", p, err)
	}

	detail, err := store.Thread(ctx, "k")
	if err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, event := range detail.Events {
		kinds = append(kinds, event.Kind)
	}
	if !slices.Equal(kinds, []string{Decision, Migration, Failover, Override, CapabilityUpgrade}) {
		t.Fatalf("event kinds = %v", kinds)
	}
	summary, err := store.Summarize(ctx, time.Now().Add(-time.Hour))
	if err != nil || len(summary.Migrations) != 1 || summary.Migrations["glm-5.3->gpt-6.1-sol"] != 1 || summary.Overrides != 1 {
		t.Fatalf("summary = %+v, %v", summary, err)
	}
}

func TestRoutingNeverWaitsForReaders(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	pinThread(t, store, base, "k", "glm-5.3", false)
	logRequests(t, store, RequestLog{At: base, Key: "k", Model: "glm-5.3"})

	// Long dashboard reads take every reader, each holding its snapshot.
	for range readers {
		tx, err := store.read.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
		if err != nil {
			t.Fatal(err)
		}
		defer tx.Rollback()
		var count int
		if err := tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM requests`).Scan(&count); err != nil {
			t.Fatal(err)
		}
	}
	waiting, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	if _, err := store.Requests(waiting, RequestFilter{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("a dashboard read with every reader busy = %v", err)
	}

	// Were routing to wait for a reader, the deadline would pass.
	quick, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	for i := range 10 {
		key := fmt.Sprintf("thread-%d", i)
		if _, err := store.Create(quick, Pin{Key: key, Model: "gpt-6.1-sol"}, Event{Key: key, Kind: Decision, Model: "gpt-6.1-sol"}); err != nil {
			t.Fatal(err)
		}
		if p, err := store.Get(quick, key); err != nil || p == nil || p.Model != "gpt-6.1-sol" {
			t.Fatalf("Get right after Create = %+v, %v", p, err)
		}
		if err := store.Touch(quick, key); err != nil {
			t.Fatal(err)
		}
		if err := store.LogRequest(quick, RequestLog{Key: key, Model: "gpt-6.1-sol"}); err != nil {
			t.Fatal(err)
		}
	}
}

// atClientSchema creates the database at path at schema version 8, the last
// with clients, holding a thread, a failed over thread with its failover, two
// logged requests and a policy.
func atClientSchema(t *testing.T, path string) {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, statement := range append(migrations[:8:8],
		`INSERT INTO pins VALUES ('solid', 'claude-code', 'glm-5.3', 'glm-5.3', 0, 0, 1000, 1000, 5000),
			('moved', 'claude-code', 'glm-5.3', 'gpt-6.1-sol', 1, 0, 2000, 3000, 6000),
			('codex', 'codex', 'gpt-6.1-sol', 'gpt-6.1-sol', 0, 0, 2500, 2500, 2500)`,
		`INSERT INTO events VALUES
			(1, 1000, 'solid', 'claude-code', 'decision', 'glm-5.3', '', 'jev', 'glm-5.3', '{"glm-5.3":0.9}', 0, 250, '',
			 '[{"decider":"jev","outcome":"decided","choice":"glm-5.3","latency_ms":250}]', '{"threshold":0.1}', 'add a flag'),
			(2, 2000, 'moved', 'claude-code', 'decision', 'gpt-6.1-sol', '', 'jev', 'gpt-6.1-sol', '{"gpt-6.1-sol":1}', 0, 300, '',
			 '[]', '{}', ''),
			(3, 3000, 'moved', 'claude-code', 'failover', 'glm-5.3', 'gpt-6.1-sol', '', '', 'null', 0, 0,
			 'upstream status 429', '[]', '{}', ''),
			(4, 2500, 'codex', 'codex', 'decision', 'gpt-6.1-sol', '', 'jev', 'gpt-6.1-sol', '{"gpt-6.1-sol":1}', 0, 300, '',
			 '[]', '{}', 'fix it')`,
		`INSERT INTO requests VALUES
			(1, 4000, 'claude-code', 'moved', 'messages', 'auto', 'glm-5.3', 'failover', 200, 1, 900, '', 10, 20, 30, 40, 0.5),
			(2, 5000, 'claude-code', '', 'messages', 'auto', 'glm-5.3-flash', 'auxiliary', 200, 0, 100, '', 1, 2, 0, 0, NULL),
			(3, 5500, 'codex', 'codex', 'responses', 'auto', 'gpt-6.1-sol', 'pinned', 200, 0, 100, '', 5, 5, 0, 0, 0.1),
			(4, 5600, 'opencode', '', 'chat', 'auto', 'glm-5.3', 'no_session', 200, 0, 100, '', 5, 5, 0, 0, 0.1)`,
		`INSERT INTO policies VALUES ('f72644462d41', 1, '{}')`,
		`PRAGMA user_version = 8`) {
		if _, err := db.Exec(statement); err != nil {
			t.Fatal(err)
		}
	}
}

// userVersion reads the schema version of the database file at path.
func userVersion(t *testing.T, path string) int {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var version int
	if err := db.QueryRow("PRAGMA user_version").Scan(&version); err != nil {
		t.Fatal(err)
	}
	return version
}

func TestOpenKeepsOnlyClaudeCodesRows(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router.db")
	atClientSchema(t, path)
	store := open(t, path)

	var pins, events, requests, policies, clients int
	if err := store.write.QueryRow(`SELECT (SELECT COUNT(*) FROM pins), (SELECT COUNT(*) FROM events),
		(SELECT COUNT(*) FROM requests), (SELECT COUNT(*) FROM policies)`).Scan(&pins, &events, &requests, &policies); err != nil ||
		pins != 2 || events != 3 || requests != 2 || policies != 1 {
		t.Fatalf("rows = %d pins, %d events, %d requests, %d policies, %v", pins, events, requests, policies, err)
	}
	if err := store.write.QueryRow(`SELECT (SELECT COUNT(*) FROM pragma_table_info('pins') WHERE name = 'client')
		+ (SELECT COUNT(*) FROM pragma_table_info('events') WHERE name = 'client')
		+ (SELECT COUNT(*) FROM pragma_table_info('requests') WHERE name = 'client')
		+ (SELECT COUNT(*) FROM sqlite_schema WHERE name = 'requests_client')`).Scan(&clients); err != nil || clients != 0 {
		t.Fatalf("client columns and index left = %d, %v", clients, err)
	}
	for key, want := range map[string]Pin{
		"solid": {Key: "solid", Model: "glm-5.3", Decided: "glm-5.3", CreatedAt: time.UnixMilli(1000),
			LastRequest: time.UnixMilli(5000)},
		"moved": {Key: "moved", Model: "glm-5.3", Decided: "gpt-6.1-sol", FailedOver: true, FailedFrom: "gpt-6.1-sol",
			CreatedAt: time.UnixMilli(2000), LastRequest: time.UnixMilli(6000)},
	} {
		if p, err := store.Get(ctx, key); err != nil || p == nil || *p != want {
			t.Fatalf("pin %s = %+v, %v; want %+v", key, p, err, want)
		}
	}

	moved, err := store.Thread(ctx, "moved")
	if err != nil || moved == nil || !moved.LastUsedAt.Equal(time.UnixMilli(6000)) || len(moved.Events) != 2 ||
		moved.Events[1].Kind != Failover || moved.Events[1].Previous != "gpt-6.1-sol" || moved.Events[1].Detail != "upstream status 429" {
		t.Fatalf("moved thread = %+v, %v", moved, err)
	}
	if r := moved.Requests; len(r) != 1 || r[0].Tokens != (usage.Tokens{Input: 10, Output: 20, CacheRead: 30, CacheWrite: 40}) ||
		r[0].Cost != 0.5 || r[0].TaskCost != 0.5 || !r[0].Failover || r[0].Latency != 900*time.Millisecond {
		t.Fatalf("moved requests = %+v", r)
	}
	solid, err := store.Thread(ctx, "solid")
	if err != nil || solid.Decision == nil || solid.Decision.Prompt != "add a flag" || solid.Decision.Basis.Threshold != 0.1 ||
		solid.Decision.Probabilities["glm-5.3"] != 0.9 || len(solid.Decision.Attempts) != 1 {
		t.Fatalf("solid decision = %+v, %v", solid.Decision, err)
	}
	if unpinned, err := store.Requests(ctx, RequestFilter{Model: "glm-5.3-flash"}); err != nil || len(unpinned) != 1 ||
		unpinned[0].Priced || unpinned[0].Tokens != (usage.Tokens{Input: 1, Output: 2}) {
		t.Fatalf("unpinned requests = %+v, %v", unpinned, err)
	}
}

func TestAFailingMigrationLeavesThePreviousVersion(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	pinThread(t, store, base, "k", "glm-5.3", false)
	store.Close()

	released := migrations
	t.Cleanup(func() { migrations = released })
	migrations = append(slices.Clip(released), `ALTER TABLE pins ADD COLUMN note TEXT NOT NULL DEFAULT '';
INSERT INTO missing VALUES (1);`)
	if _, err := Open(path); err == nil {
		t.Fatal("Open ran a failing migration")
	}
	// The fixed binary runs the whole migration again.
	migrations[len(released)] = `ALTER TABLE pins ADD COLUMN note TEXT NOT NULL DEFAULT '';`
	fixed := open(t, path)
	var version int
	if err := fixed.write.QueryRow("PRAGMA user_version").Scan(&version); err != nil || version != len(migrations) {
		t.Fatalf("user_version = %d, %v", version, err)
	}
	if p, err := fixed.Get(ctx, "k"); err != nil || p == nil || p.Model != "glm-5.3" {
		t.Fatalf("pin after the fixed migration = %+v, %v", p, err)
	}
}

func TestBackupSnapshotsADatabaseAsItStands(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	database := filepath.Join(dir, "router.db")
	atClientSchema(t, database)

	// A backup neither needs nor runs the migrations.
	old := filepath.Join(dir, "old.db")
	if err := Backup(ctx, database, old); err != nil {
		t.Fatal(err)
	}
	if userVersion(t, database) != 8 || userVersion(t, old) != 8 {
		t.Fatalf("versions after the backup = %d, %d", userVersion(t, database), userVersion(t, old))
	}

	// A backup runs alongside the store.
	store := open(t, database)
	logRequests(t, store, RequestLog{At: base, Key: "solid", Model: "glm-5.3", Cost: 0.25, Priced: true})
	path := filepath.Join(dir, "backup.db")
	if err := Backup(ctx, database, path); err != nil {
		t.Fatal(err)
	}
	if err := Backup(ctx, database, path); !errors.Is(err, fs.ErrExist) {
		t.Fatalf("a backup over an existing file = %v", err)
	}
	pinThread(t, store, base, "later", "gpt-6.1-sol", false)
	backup := open(t, path)
	solid, err := backup.Thread(ctx, "solid")
	if err != nil || solid == nil || len(solid.Requests) != 1 || solid.Requests[0].TaskCost != 0.25 {
		t.Fatalf("backed up thread = %+v, %v", solid, err)
	}
	if later, err := backup.Get(ctx, "later"); err != nil || later != nil {
		t.Fatalf("the backup holds a later pin: %+v, %v", later, err)
	}

	missing, target := filepath.Join(dir, "missing.db"), filepath.Join(dir, "none.db")
	if err := Backup(ctx, missing, target); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("a backup of a missing database = %v", err)
	}
	for _, path := range []string{missing, target} {
		if _, err := os.Stat(path); !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("the failed backup left %s: %v", path, err)
		}
	}
}

func TestAnOlderBuildRefusesANewerDatabase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	open(t, path).Close()
	saved := migrations
	t.Cleanup(func() { migrations = saved })
	migrations = migrations[:len(migrations)-1]
	if store, err := Open(path); err == nil || !strings.Contains(err.Error(), "newer than this build") {
		if store != nil {
			store.Close()
		}
		t.Fatalf("an older build opened a newer database: %v", err)
	}
}
