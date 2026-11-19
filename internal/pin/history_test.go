package pin

import (
	"context"
	"database/sql"
	"maps"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
)

// pinThread pins key to model at the given time through a decision or,
// when undecided is set, through an event of a kind the router no longer
// records, as threads pinned before it served Claude Code alone were.
func pinThread(t *testing.T, store *Store, now time.Time, key, model string, undecided bool) {
	t.Helper()
	at(store, now)
	event := Event{Key: key, Kind: Decision, Model: model, Decider: "jev", Choice: model,
		Probabilities: map[string]float64{model: 0.9, "deepseek-flash": 0.1}, Latency: 250 * time.Millisecond}
	if undecided {
		event = Event{Key: key, Kind: "inherit", Model: model, Previous: "parent"}
	}
	if _, err := store.Create(context.Background(), Pin{Key: key, Model: model}, event); err != nil {
		t.Fatal(err)
	}
}

func keys(threads []Thread) []string {
	result := []string{}
	for _, thread := range threads {
		result = append(result, thread.Key)
	}
	return result
}

func TestThreadsFilterAndPaginate(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	pinThread(t, store, base, "a", "glm-5.3", false)
	pinThread(t, store, base.Add(time.Minute), "b", "gpt-6.1-sol", false)
	pinThread(t, store, base.Add(2*time.Minute), "c", "gpt-6.1-sol", true)
	pinThread(t, store, base.Add(3*time.Minute), "d", "glm-5.3", false)
	at(store, base.Add(10*time.Minute))
	if err := store.Touch(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	logRequests(t, store, RequestLog{Key: "a"}, RequestLog{Key: "a"}, RequestLog{Key: "c"}, RequestLog{})

	cases := []struct {
		name   string
		filter ThreadFilter
		want   []string
		total  int
	}{
		{"most recently used first", ThreadFilter{}, []string{"a", "d", "c", "b"}, 4},
		{"model", ThreadFilter{Model: "gpt-6.1-sol"}, []string{"c", "b"}, 2},
		{"model page", ThreadFilter{Model: "gpt-6.1-sol", Limit: 1, Offset: 1}, []string{"b"}, 2},
		{"first page", ThreadFilter{Limit: 2}, []string{"a", "d"}, 4},
		{"second page", ThreadFilter{Limit: 2, Offset: 2}, []string{"c", "b"}, 4},
		{"past the end", ThreadFilter{Offset: 10}, []string{}, 4},
		{"negative offset", ThreadFilter{Offset: -1}, []string{"a", "d", "c", "b"}, 4},
		{"no match", ThreadFilter{Model: "claude-opus-5-5"}, []string{}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			page, err := store.Threads(ctx, c.filter)
			if err != nil || page.Threads == nil || !slices.Equal(keys(page.Threads), c.want) || page.Total != c.total {
				t.Fatalf("Threads(%+v) = %v of %d, %v; want %v of %d", c.filter, keys(page.Threads), page.Total, err, c.want, c.total)
			}
		})
	}

	page, err := store.Threads(ctx, ThreadFilter{})
	if err != nil {
		t.Fatal(err)
	}
	a, d, c := page.Threads[0], page.Threads[1], page.Threads[2]
	if a.Requests != 2 || c.Requests != 1 || d.Requests != 0 {
		t.Fatalf("request counts = %d, %d, %d", a.Requests, c.Requests, d.Requests)
	}
	if !a.LastUsedAt.Equal(base.Add(10*time.Minute)) || !a.CreatedAt.Equal(base) || a.Decided != "glm-5.3" {
		t.Fatalf("thread a = %+v", a)
	}
	if c.Decision != nil {
		t.Fatalf("inherited thread has a decision: %+v", c.Decision)
	}
	decision := d.Decision
	if decision == nil || decision.ID == 0 || !decision.At.Equal(base.Add(3*time.Minute)) || decision.Key != "d" ||
		decision.Decider != "jev" || decision.Probabilities["glm-5.3"] != 0.9 || decision.Latency != 250*time.Millisecond {
		t.Fatalf("decision of d = %+v", decision)
	}
}

func TestThreadCarriesItsEventsAndLatestRequests(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	pinThread(t, store, base, "k", "glm-5.3", false)
	at(store, base.Add(time.Minute))
	if _, err := store.Upgrade(ctx, "k", "glm-5.3", "gpt-6.1-sol", Event{Key: "k", Kind: CapabilityUpgrade, Model: "gpt-6.1-sol", Previous: "glm-5.3"}); err != nil {
		t.Fatal(err)
	}
	at(store, base.Add(2*time.Minute))
	if _, err := store.Failover(ctx, "k", "gpt-6.1-sol", "deepseek-flash", true, Event{Key: "k", Kind: Failover, Model: "deepseek-flash", Previous: "gpt-6.1-sol"}); err != nil {
		t.Fatal(err)
	}
	pinThread(t, store, base.Add(3*time.Minute), "child", "glm-5.3", true)
	flood(t, store, threadRequests, base, "k", "deepseek-flash")
	logRequests(t, store, RequestLog{At: base.Add(time.Hour), Key: "k", Model: "deepseek-flash", Status: 429}, RequestLog{At: base.Add(2 * time.Hour), Key: "child"})

	detail, err := store.Thread(ctx, "k")
	if err != nil || detail == nil {
		t.Fatalf("Thread = %+v, %v", detail, err)
	}
	if detail.Model != "deepseek-flash" || !detail.FailedOver || detail.FailedFrom != "gpt-6.1-sol" || !detail.Upgraded ||
		detail.Thread.Requests != threadRequests+1 {
		t.Fatalf("thread = %+v", detail.Thread)
	}
	var kinds []string
	for i, event := range detail.Events {
		kinds = append(kinds, event.Kind)
		if event.ID == 0 || !event.At.Equal(base.Add(time.Duration(i)*time.Minute)) {
			t.Fatalf("event %d = %+v", i, event)
		}
	}
	if !slices.Equal(kinds, []string{Decision, CapabilityUpgrade, Failover}) {
		t.Fatalf("event kinds = %v", kinds)
	}
	if detail.Decision == nil || detail.Decision.ID != detail.Events[0].ID || detail.Decision.Probabilities["glm-5.3"] != 0.9 {
		t.Fatalf("decision = %+v", detail.Decision)
	}
	if len(detail.Requests) != threadRequests || detail.Requests[0].Status != 429 || !detail.Requests[0].At.Equal(base.Add(time.Hour)) {
		t.Fatalf("thread requests = %d, newest %+v", len(detail.Requests), detail.Requests[0])
	}

	child, err := store.Thread(ctx, "child")
	if err != nil || child == nil || child.Decision != nil || len(child.Events) != 1 || child.Events[0].Kind != "inherit" || len(child.Requests) != 1 {
		t.Fatalf("undecided thread = %+v, %v", child, err)
	}
	if unknown, err := store.Thread(ctx, "unknown"); err != nil || unknown != nil {
		t.Fatalf("Thread(unknown) = %+v, %v", unknown, err)
	}
}

func TestOverviewLaysOutEveryBucket(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	since := base.Add(-150 * time.Minute)
	pinThread(t, store, since.Add(-time.Hour), "before", "gpt-6.1-sol", false)
	pinThread(t, store, since.Add(10*time.Minute), "x", "glm-5.3", false)
	pinThread(t, store, since.Add(125*time.Minute), "y", "gpt-6.1-sol", false)
	pinThread(t, store, since.Add(126*time.Minute), "z", "gpt-6.1-sol", true)
	logRequests(t, store,
		RequestLog{At: since.Add(-time.Minute), Model: "glm-5.3", Status: 500},
		RequestLog{At: since, Model: "glm-5.3", Status: 200},
		RequestLog{At: since.Add(59 * time.Minute), Model: "gpt-6.1-sol", Status: 500},
		RequestLog{At: since.Add(2 * time.Hour), Model: "glm-5.3", Status: 0},
		RequestLog{At: base, Model: "glm-5.3", Status: 429},
		RequestLog{At: base, Model: "deepseek-flash", Status: 200},
	)
	at(store, base)

	overview, err := store.Overview(ctx, since, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if overview.Threads != 4 || overview.Requests != 5 || overview.Errors != 3 {
		t.Fatalf("threads %d, requests %d, errors %d", overview.Threads, overview.Requests, overview.Errors)
	}
	if overview.Summary.Decisions["glm-5.3"] != 1 || overview.Summary.Decisions["gpt-6.1-sol"] != 1 || len(overview.Summary.Decisions) != 2 {
		t.Fatalf("summary = %+v", overview.Summary)
	}
	want := []Bucket{
		{Start: since, Requests: map[string]int{"glm-5.3": 1, "gpt-6.1-sol": 1}, Decisions: map[string]int{"glm-5.3": 1}},
		{Start: since.Add(time.Hour), Requests: map[string]int{}, Decisions: map[string]int{}},
		{Start: since.Add(2 * time.Hour), Requests: map[string]int{"glm-5.3": 2, "deepseek-flash": 1}, Decisions: map[string]int{"gpt-6.1-sol": 1}},
	}
	if len(overview.Timeline) != len(want) {
		t.Fatalf("timeline has %d buckets, want %d", len(overview.Timeline), len(want))
	}
	for i, bucket := range overview.Timeline {
		if !bucket.Start.Equal(want[i].Start) || bucket.Requests == nil || bucket.Decisions == nil ||
			!maps.Equal(bucket.Requests, want[i].Requests) || !maps.Equal(bucket.Decisions, want[i].Decisions) {
			t.Errorf("bucket %d = %+v, want %+v", i, bucket, want[i])
		}
	}
}

func TestOverviewBucketBounds(t *testing.T) {
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	at(store, base)
	cases := []struct {
		name    string
		since   time.Time
		bucket  time.Duration
		buckets int
	}{
		{"exact window ends with the bucket holding now", base.Add(-24 * time.Hour), time.Hour, 25},
		{"partial last bucket", base.Add(-90 * time.Minute), time.Hour, 2},
		{"since in the future", base.Add(time.Hour), time.Minute, 1},
		{"bucket below a millisecond", base.Add(-time.Hour), time.Microsecond, 0},
		{"too many buckets", time.Time{}, time.Millisecond, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			overview, err := store.Overview(context.Background(), c.since, c.bucket)
			if c.buckets == 0 {
				if err == nil {
					t.Fatal("Overview accepted the bucket")
				}
				return
			}
			if err != nil || len(overview.Timeline) != c.buckets {
				t.Fatalf("Overview = %d buckets, %v; want %d", len(overview.Timeline), err, c.buckets)
			}
		})
	}
}

func TestOpenMigratesAVersionOneDatabase(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	for _, statement := range []string{
		migrations[0],
		`INSERT INTO pins VALUES ('k', 'claude-code', 'glm-5.3', 'glm-5.3', 0, 0, 1, 1, 1)`,
		`INSERT INTO events VALUES (1, 1, 'k', 'claude-code', 'decision', 'glm-5.3', '', 'jev', 'glm-5.3', '{}', 0, 250, '')`,
		`PRAGMA user_version = 1`,
	} {
		if _, err := db.Exec(statement); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()

	store := open(t, path)
	logRequests(t, store, RequestLog{Key: "k", Model: "glm-5.3"})
	detail, err := store.Thread(ctx, "k")
	if err != nil || detail == nil || detail.Model != "glm-5.3" || len(detail.Requests) != 1 {
		t.Fatalf("migrated thread = %+v, %v", detail, err)
	}
	want := []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "glm-5.3", Probabilities: map[string]float64{}, LatencyMS: 250}}
	if detail.Decision == nil || !reflect.DeepEqual(detail.Decision.Attempts, want) || detail.Decision.Basis.Threshold != 0.1 {
		t.Fatalf("migrated decision = %+v", detail.Decision)
	}
	var version int
	if err := store.write.QueryRow("PRAGMA user_version").Scan(&version); err != nil || version != len(migrations) {
		t.Fatalf("user_version = %d, %v", version, err)
	}
}

func TestOpenTurnsDoubtsIntoAttempts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	for _, statement := range append(migrations[:5:5],
		`INSERT INTO events (at, key, client, kind, model, previous, decider, choice, probabilities, escalated, latency_ms, detail, doubts)
		 VALUES (1, 'consulted', 'claude-code', 'decision', 'glm-5.3', '', 'judge', 'glm-5.3', '{"glm-5.3":1}', 0, 3900, '',
		         '[{"decider":"jev","choice":"deepseek-flash","above":0.31}]'),
		        (2, 'stood', 'claude-code', 'decision', 'gpt-6.1-sol', '', 'jev', 'glm-5.3', '{"glm-5.3":0.8,"gpt-6.1-sol":0.2}', 1, 5000, '',
		         '[{"decider":"jev","choice":"glm-5.3","above":0.2}]'),
		        (3, 'default', 'claude-code', 'decision', 'glm-5.3', '', 'default', '', 'null', 0, 0, 'jev: down', '[]'),
		        (4, 'stood', 'claude-code', 'failover', 'glm-5.3', 'gpt-6.1-sol', '', '', 'null', 0, 0, 'upstream status 429', '[]')`,
		`PRAGMA user_version = 5`) {
		if _, err := db.Exec(statement); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()

	store := open(t, path)
	log, err := events(context.Background(), store.read, ` ORDER BY id`)
	if err != nil || len(log) != 4 {
		t.Fatalf("events = %+v, %v", log, err)
	}
	want := [][]decide.Attempt{
		{{Decider: "jev", Outcome: decide.Doubted, Choice: "deepseek-flash", Above: 0.31},
			{Decider: "judge", Outcome: decide.Decided, Choice: "glm-5.3", Probabilities: map[string]float64{"glm-5.3": 1}, LatencyMS: 3900}},
		{{Decider: "jev", Outcome: decide.Doubted, Choice: "glm-5.3", Probabilities: map[string]float64{"glm-5.3": 0.8, "gpt-6.1-sol": 0.2}, Above: 0.2}},
		{},
		{},
	}
	for i, e := range log {
		if !reflect.DeepEqual(e.Attempts, want[i]) {
			t.Errorf("event %d attempts = %+v, want %+v", e.ID, e.Attempts, want[i])
		}
	}
	if log[0].Basis.Threshold != 0.1 || log[3].Basis.Threshold != 0 || log[2].Detail != "jev: down" {
		t.Fatalf("migrated events = %+v", log)
	}
}

func TestOpenDropsLoggedExplicitRequests(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	insert := `INSERT INTO requests (at, client, key, kind, requested, model, reason, status, failover, latency_ms, error)
		VALUES (1, 'claude-code', ?, 'messages', ?, ?, ?, 200, 0, 5, '')`
	for _, statement := range append(migrations[:4:4], `PRAGMA user_version = 4`) {
		if _, err := db.Exec(statement); err != nil {
			t.Fatal(err)
		}
	}
	for _, row := range [][]any{
		{"k", "auto", "glm-5.3", "decided"},
		{"", "claude-opus-5-5", "claude-opus-5-5", "explicit"},
	} {
		if _, err := db.Exec(insert, row...); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()

	requests, err := open(t, path).Requests(context.Background(), RequestFilter{})
	if err != nil || len(requests) != 1 || requests[0].Requested != "auto" {
		t.Fatalf("migrated requests = %+v, %v", requests, err)
	}
}

func TestDecisionsKeepTheirTrailBasisAndPrompt(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	attempts := []decide.Attempt{
		{Decider: "jev", Outcome: decide.Doubted, Model: "jev-1.13.0", Choice: "glm-5.3",
			Probabilities: map[string]float64{"glm-5.3": 0.65, "gpt-6.1-sol": 0.35}, Confidence: 0.6, Above: 0.35,
			InputTokens: 356, OutputTokens: 52, LatencyMS: 280},
		{Decider: "llm", Outcome: decide.Failed, LatencyMS: 20000, Error: "status 503"},
		{Decider: "judge", Outcome: decide.Decided, Model: "gpt-6-luna", Choice: "gpt-6.1-sol",
			Probabilities: map[string]float64{"gpt-6.1-sol": 1}, Reason: "Cross-cutting bug.", LatencyMS: 3500},
	}
	basis := Basis{Kind: "messages", UserAgent: "claude-cli/2.1.284", Candidates: []string{"glm-5.3", "gpt-6.1-sol"},
		Threshold: 0.1, Doubt: "consult", PromptChars: 18000, Truncated: true, Policy: "a1b2c3d4e5f6", Version: "9e9f7a1"}
	at(store, base)
	if _, err := store.Create(ctx, Pin{Key: "consulted", Model: "gpt-6.1-sol"}, Event{
		Key: "consulted", Kind: Decision, Model: "gpt-6.1-sol", Decider: "judge", Choice: "gpt-6.1-sol",
		Attempts: attempts, Basis: basis, Prompt: "client=claude-code images=no\n\nfix the race",
	}); err != nil {
		t.Fatal(err)
	}
	pinThread(t, store, base, "sure", "glm-5.3", false)

	detail, err := store.Thread(ctx, "consulted")
	if err != nil || !reflect.DeepEqual(detail.Decision.Attempts, attempts) || !reflect.DeepEqual(detail.Decision.Basis, basis) ||
		detail.Decision.Prompt != "client=claude-code images=no\n\nfix the race" || !detail.Decision.Consulted() {
		t.Fatalf("decision = %+v, %v", detail.Decision, err)
	}
	if sure, err := store.Thread(ctx, "sure"); err != nil || sure.Decision.Attempts == nil || sure.Decision.Consulted() {
		t.Fatalf("a sure decision = %+v, %v", sure.Decision, err)
	}
	summary, err := store.Summarize(ctx, base.Add(-time.Hour))
	if err != nil || summary.Consulted != 1 || total(summary.Decisions) != 2 {
		t.Fatalf("summary = %+v, %v", summary, err)
	}
}

func TestSummaryLatencyCoversTheDecidersThatRan(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	at(store, base)
	for key, event := range map[string]Event{
		"quick":   {Decider: "jev", Latency: 300 * time.Millisecond, Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, LatencyMS: 300}}},
		"sure":    {Decider: "jev", Latency: 400 * time.Millisecond, Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, LatencyMS: 400}}},
		"slow":    {Decider: "judge", Latency: 900 * time.Millisecond, Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Failed, LatencyMS: 500}, {Decider: "judge", Outcome: decide.Decided, LatencyMS: 400}}},
		"default": {Decider: "default"},
		"resting": {Decider: "default", Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Skipped}}},
	} {
		event.Key, event.Kind, event.Model = key, Decision, "glm-5.3"
		if _, err := store.Create(ctx, Pin{Key: key, Model: "glm-5.3"}, event); err != nil {
			t.Fatal(err)
		}
	}
	summary, err := store.Summarize(ctx, base.Add(-time.Hour))
	if err != nil || summary.Decisions["glm-5.3"] != 5 || summary.Deciders["default"] != 2 ||
		!maps.Equal(summary.LatencyMS, map[string]int64{"p50": 400, "p95": 900, "max": 900}) {
		t.Fatalf("summary = %+v, %v", summary, err)
	}
}

func total(counts map[string]int) int {
	sum := 0
	for count := range maps.Values(counts) {
		sum += count
	}
	return sum
}
