package pin

import (
	"context"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// base is the fixed "now" of the history tests.
var base = time.Date(2026, 9, 28, 12, 0, 0, 0, time.UTC)

// at pins the store's clock to a fixed time.
func at(store *Store, now time.Time) {
	store.now = func() time.Time { return now }
}

func logRequests(t *testing.T, store *Store, requests ...RequestLog) {
	t.Helper()
	for _, request := range requests {
		if err := store.LogRequest(context.Background(), request); err != nil {
			t.Fatal(err)
		}
	}
}

// flood logs n requests of one key and model at the same time, in one
// statement.
func flood(t *testing.T, store *Store, n int, at time.Time, key, model string) {
	t.Helper()
	_, err := store.write.Exec(`WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?)
		INSERT INTO requests (at, key, kind, requested, model, reason, status, failover, latency_ms, error)
		SELECT ?, ?, 'messages', 'auto', ?, 'pinned', 200, 0, 0, '' FROM n`, n, at.UnixMilli(), key, model)
	if err != nil {
		t.Fatal(err)
	}
}

func ids(requests []RequestLog) []int64 {
	result := []int64{}
	for _, r := range requests {
		result = append(result, r.ID)
	}
	return result
}

func TestRequestsFilterOrderAndTail(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	logRequests(t, store,
		RequestLog{At: base, Key: "a", Kind: "messages", Requested: "auto", Model: "glm-5.3", Reason: "decided", Status: 200},
		RequestLog{At: base.Add(time.Minute), Key: "b", Kind: "messages", Requested: "auto", Model: "gpt-6.1-sol", Reason: "pinned", Status: 200},
		RequestLog{At: base.Add(2 * time.Minute), Key: "a", Kind: "messages", Requested: "auto", Model: "glm-5.3", Reason: "pinned", Status: 400},
		RequestLog{At: base.Add(3 * time.Minute), Kind: "messages", Requested: "auto", Model: "glm-5.3-flash", Reason: "auxiliary", Status: 200},
		// A long stream: it arrived before the fourth request but finished,
		// and was logged, after it.
		RequestLog{At: base.Add(90 * time.Second), Key: "a", Kind: "messages", Requested: "auto", Model: "gpt-6.1-sol", Reason: "failover", Failover: true, Status: 200},
	)
	cases := []struct {
		name   string
		filter RequestFilter
		want   []int64
	}{
		{"newest arrival first", RequestFilter{}, []int64{4, 3, 5, 2, 1}},
		{"model", RequestFilter{Model: "gpt-6.1-sol"}, []int64{5, 2}},
		{"key", RequestFilter{Key: "a"}, []int64{3, 5, 1}},
		{"key and model", RequestFilter{Key: "a", Model: "glm-5.3"}, []int64{3, 1}},
		{"limit", RequestFilter{Limit: 2}, []int64{4, 3}},
		{"no match", RequestFilter{Model: "claude-opus-5-5"}, []int64{}},
		{"tail in log order", RequestFilter{AfterID: 2}, []int64{3, 4, 5}},
		{"tail with filter", RequestFilter{AfterID: 1, Key: "a"}, []int64{3, 5}},
		{"tail limit keeps the oldest", RequestFilter{AfterID: 1, Limit: 2}, []int64{2, 3}},
		{"tail at the end", RequestFilter{AfterID: 5}, []int64{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := store.Requests(ctx, c.filter)
			if err != nil || got == nil || !slices.Equal(ids(got), c.want) {
				t.Fatalf("Requests(%+v) = %v, %v; want %v", c.filter, ids(got), err, c.want)
			}
		})
	}
}

func TestRequestLimits(t *testing.T) {
	cases := []struct{ limit, fallback, most, want int }{
		{0, requestLimit, maxRequestLimit, requestLimit},
		{-5, requestLimit, maxRequestLimit, requestLimit},
		{7, requestLimit, maxRequestLimit, 7},
		{5000, requestLimit, maxRequestLimit, maxRequestLimit},
		{0, threadLimit, maxThreadLimit, threadLimit},
		{501, threadLimit, maxThreadLimit, maxThreadLimit},
	}
	for _, c := range cases {
		if got := bounded(c.limit, c.fallback, c.most); got != c.want {
			t.Errorf("bounded(%d, %d, %d) = %d, want %d", c.limit, c.fallback, c.most, got, c.want)
		}
	}

	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	flood(t, store, maxRequestLimit+5, base, "k", "glm-5.3")
	for _, c := range []struct{ limit, want int }{{0, requestLimit}, {maxRequestLimit + 1, maxRequestLimit}} {
		if got, err := store.Requests(context.Background(), RequestFilter{Limit: c.limit}); err != nil || len(got) != c.want {
			t.Fatalf("limit %d returned %d requests, %v; want %d", c.limit, len(got), err, c.want)
		}
	}
}

func TestRequestLogRoundTripsAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	at(store, base)
	want := RequestLog{
		At: base.Add(-time.Minute), Key: "k", Kind: "messages", Requested: "auto", Model: "glm-5.3", Effort: "high",
		Reason: "failover", Status: 0, Failover: true, Latency: 1500 * time.Millisecond, Error: "CLIProxy unreachable: EOF",
	}
	logRequests(t, store, want, RequestLog{Kind: "other"})
	store.Close()

	got, err := open(t, path).Requests(ctx, RequestFilter{})
	if err != nil || len(got) != 2 {
		t.Fatalf("Requests after reopen = %+v, %v", got, err)
	}
	if !got[0].At.Equal(base) {
		t.Fatalf("a request without At was logged at %v, want the store's now", got[0].At)
	}
	stored := got[1]
	if stored.ID != 1 || !stored.At.Equal(want.At) {
		t.Fatalf("stored = %+v", stored)
	}
	stored.ID, stored.At = 0, want.At
	if stored != want {
		t.Fatalf("stored = %+v, want %+v", stored, want)
	}
}

func TestPruneDeletesOnlyOldRequests(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	old := base.Add(-48 * time.Hour)
	at(store, old)
	if _, err := store.Create(ctx, Pin{Key: "k", Model: "glm-5.3"}, Event{Key: "k", Kind: Decision, Model: "glm-5.3"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Failover(ctx, "k", "glm-5.3", "gpt-6.1-sol", true, Event{Key: "k", Kind: Failover, Model: "gpt-6.1-sol", Previous: "glm-5.3"}); err != nil {
		t.Fatal(err)
	}
	stale := pruneBatch + 5
	flood(t, store, stale, old, "k", "glm-5.3")
	logRequests(t, store, RequestLog{At: base.Add(-time.Hour), Key: "k", Model: "gpt-6.1-sol"}, RequestLog{At: base, Model: "glm-5.3"})

	cutoff := base.Add(-24 * time.Hour)
	if deleted, err := store.Prune(ctx, cutoff); err != nil || deleted != int64(stale) {
		t.Fatalf("Prune = %d, %v; want %d", deleted, err, stale)
	}
	if deleted, err := store.Prune(ctx, cutoff); err != nil || deleted != 0 {
		t.Fatalf("second Prune = %d, %v", deleted, err)
	}
	if left, err := store.Requests(ctx, RequestFilter{}); err != nil || len(left) != 2 {
		t.Fatalf("requests left = %d, %v", len(left), err)
	}
	detail, err := store.Thread(ctx, "k")
	if err != nil || detail == nil || detail.Model != "gpt-6.1-sol" || len(detail.Events) != 2 || detail.Thread.Requests != 1 {
		t.Fatalf("thread after prune = %+v, %v", detail, err)
	}
}

func TestRequestStatusFilterFindsOlderMatches(t *testing.T) {
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	logRequests(t, store, RequestLog{Key: "a", Model: "glm-5.3", Status: 503},
		RequestLog{Key: "a", Model: "glm-5.3", Status: 200, Error: "stream interrupted"})
	flood(t, store, 600, time.Now(), "a", "glm-5.3")
	requests, err := store.Requests(t.Context(), RequestFilter{Status: "errors", Model: "glm-5.3", Limit: 50})
	if err != nil || !slices.Equal(ids(requests), []int64{2, 1}) {
		t.Fatalf("old status matches = %v, %v", ids(requests), err)
	}
	requests, err = store.Requests(t.Context(), RequestFilter{Status: "success", AfterID: 1, Limit: 2})
	if err != nil || !slices.Equal(ids(requests), []int64{3, 4}) {
		t.Fatalf("successful live matches = %v, %v", ids(requests), err)
	}
}
