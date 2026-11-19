package pin

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/usage"
)

func TestThreadActivitiesReadHowSettledThreadsWent(t *testing.T) {
	ctx := t.Context()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	at(store, base.Add(-3*time.Hour))
	for _, key := range []string{"moved", "quiet", "fixed", "busy"} {
		if _, err := store.Create(ctx, Pin{Key: key, Model: "glm-5.3", Level: "low"},
			Event{Key: key, Kind: Decision, Model: "glm-5.3", Level: "low"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Move(ctx, "moved", "glm-5.3", "claude-opus-5-5", "high", true,
		Event{Key: "moved", Kind: Override, Model: "claude-opus-5-5", Level: "high", Previous: "glm-5.3", Detail: "re-pinned by an operator"}); err != nil {
		t.Fatal(err)
	}
	early := base.Add(-2 * time.Hour)
	logRequests(t, store,
		RequestLog{At: early, Key: "moved", Kind: "messages", Model: "glm-5.3", Reason: "decided", Status: 200,
			Latency: 2 * time.Second, Tokens: usage.Tokens{Input: 100, Output: 10}, Cost: 0.25, Priced: true},
		RequestLog{At: early.Add(10 * time.Minute), Key: "moved", Kind: "messages", Model: "claude-opus-5-5",
			Reason: "pinned", Status: 200, Latency: 3 * time.Second, Queue: 2 * time.Second, Tokens: usage.Tokens{Input: 50}, Cost: 0.5, Priced: true},
		// Token counts, compactions and background calls are no turns, but
		// their time counts.
		RequestLog{At: early.Add(time.Minute), Key: "moved", Kind: "count_tokens", Model: "claude-opus-5-5",
			Reason: "side_request", Status: 200, Latency: time.Second},
		// A wait longer than the latency logged counts as nothing, not less.
		RequestLog{At: early.Add(4 * time.Minute), Key: "moved", Kind: "count_tokens", Model: "claude-opus-5-5",
			Reason: "side_request", Status: 200, Latency: time.Second, Queue: 5 * time.Second},
		RequestLog{At: early.Add(2 * time.Minute), Key: "moved", Kind: "messages", Model: "claude-opus-5-5",
			Reason: "side_request", Status: 200},
		RequestLog{At: early.Add(3 * time.Minute), Key: "moved", Kind: "messages", Model: "glm-5.3-flash",
			Reason: "auxiliary", Status: 200},
		RequestLog{At: early, Key: "quiet", Kind: "messages", Model: "glm-5.3", Reason: "decided", Status: 200, Latency: time.Second},
		RequestLog{At: early, Key: "fixed", Kind: "messages", Model: "glm-5.3", Reason: "fixed", Status: 200, Latency: time.Second},
		// A thread still running has not settled.
		RequestLog{At: base.Add(-10 * time.Minute), Key: "busy", Kind: "messages", Model: "glm-5.3", Reason: "decided", Status: 200},
	)
	at(store, base)

	threads, err := store.ThreadActivities(ctx, base.Add(-24*time.Hour), base.Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if len(threads) != 2 || threads[0].Key != "moved" || threads[1].Key != "quiet" {
		t.Fatalf("threads = %+v", threads)
	}
	moved := threads[0]
	if moved.Decided != "glm-5.3" || moved.Model != "claude-opus-5-5" || moved.Level != "low" || moved.Requests != 2 ||
		moved.Active != 4*time.Second || moved.Last.Sub(moved.First) != 10*time.Minute || moved.Cost != 0.75 ||
		moved.Tokens != (usage.Tokens{Input: 150, Output: 10}) || moved.Decision == 0 || moved.Reviewed {
		t.Fatalf("moved = %+v", moved)
	}
	if len(moved.Changes) != 1 || moved.Changes[0].Kind != Override || moved.Changes[0].Previous != "glm-5.3" ||
		moved.Changes[0].Model != "claude-opus-5-5" {
		t.Fatalf("changes = %+v", moved.Changes)
	}
	if len(threads[1].Changes) != 0 {
		t.Fatalf("a quiet thread changed: %+v", threads[1].Changes)
	}

	if err := store.SaveReview(ctx, Review{Event: moved.Decision, Model: "claude-opus-5-5", Level: "high"}); err != nil {
		t.Fatal(err)
	}
	if threads, err = store.ThreadActivities(ctx, base.Add(-24*time.Hour), base.Add(-time.Hour)); err != nil || !threads[0].Reviewed {
		t.Fatalf("a reviewed decision = %+v, %v", threads, err)
	}
}

func TestThreadActivitiesNeverScanAWholeTable(t *testing.T) {
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	for name, query := range map[string]string{"activities": activitiesQuery, "changes": changesQuery} {
		rows, err := store.read.QueryContext(t.Context(), "EXPLAIN QUERY PLAN "+query, int64(0), int64(0), Decision)
		if err != nil {
			t.Fatal(err)
		}
		var plan []string
		for rows.Next() {
			var id, parent, unused int
			var detail string
			if err := rows.Scan(&id, &parent, &unused, &detail); err != nil {
				t.Fatal(err)
			}
			plan = append(plan, detail)
		}
		rows.Close()
		text := strings.Join(plan, "\n")
		for _, table := range []string{"pins", "requests", "events", "reviews"} {
			for _, line := range plan {
				if line == "SCAN "+table || strings.HasPrefix(line, "SCAN "+table+" ") && !strings.Contains(line, "INDEX") {
					t.Errorf("the %s query scans %s:\n%s", name, table, text)
				}
			}
		}
		if !strings.Contains(text, "pins_created") {
			t.Errorf("the %s query does not read pins by creation:\n%s", name, text)
		}
	}
}

func TestExplorationReachesTheOutcomesAndStats(t *testing.T) {
	ctx := t.Context()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	at(store, base.Add(-3*time.Hour))
	for key, basis := range map[string]Basis{
		"explored": {ExploreEligible: true, Explore: true, Propensity: 0.05, ExploredFrom: "gpt-6.1-sol"},
		"kept":     {ExploreEligible: true, Propensity: 0.95},
		"plain":    {},
	} {
		if _, err := store.Create(ctx, Pin{Key: key, Model: "glm-5.3"},
			Event{Key: key, Kind: Decision, Model: "glm-5.3", Basis: basis}); err != nil {
			t.Fatal(err)
		}
		logRequests(t, store, RequestLog{At: base.Add(-2 * time.Hour), Key: key, Kind: "messages", Model: "glm-5.3",
			Reason: "decided", Status: 200, Latency: time.Second})
	}
	at(store, base)
	threads, err := store.ThreadActivities(ctx, base.Add(-24*time.Hour), base.Add(-time.Hour))
	if err != nil || len(threads) != 3 {
		t.Fatalf("threads = %+v, %v", threads, err)
	}
	want := map[string]struct {
		explored   bool
		propensity float64
	}{"explored": {true, 0.05}, "kept": {false, 0.95}, "plain": {false, 0}}
	for _, thread := range threads {
		if w := want[thread.Key]; thread.Explored != w.explored || thread.Propensity != w.propensity {
			t.Errorf("%s: explored %v with propensity %v", thread.Key, thread.Explored, thread.Propensity)
		}
	}
	summary, err := store.Summarize(ctx, base.Add(-24*time.Hour))
	if err != nil || summary.Explored != 1 || summary.ExploreEligible != 2 {
		t.Fatalf("summary explored %d of %d eligible, %v", summary.Explored, summary.ExploreEligible, err)
	}
}
