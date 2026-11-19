package pin

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/usage"
)

func TestThreadsAndOverviewCarryCost(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router.db")
	store := open(t, path)
	at(store, base)
	for key, model := range map[string]string{"a": "gpt-6.1-sol", "b": "gpt-6.1-sol", "c": "claude-opus-5-5"} {
		if _, err := store.Create(ctx, Pin{Key: key, Model: model}, Event{Key: key, Kind: Decision, Model: model}); err != nil {
			t.Fatal(err)
		}
	}
	logRequests(t, store,
		RequestLog{At: base.Add(-time.Hour), Key: "a", Model: "gpt-6.1-sol", Status: 200, Tokens: usage.Tokens{Input: 100, Output: 50}, Cost: 0.10, Priced: true},
		RequestLog{At: base.Add(-time.Hour), Key: "a", Model: "gpt-6.1-sol", Status: 200, Tokens: usage.Tokens{Input: 10, CacheRead: 900}, Cost: 0.02, Priced: true},
		RequestLog{At: base.Add(-time.Hour), Key: "b", Model: "gpt-6.1-sol", Status: 200, Tokens: usage.Tokens{Output: 5}, Cost: 0.06, Priced: true},
		RequestLog{At: base.Add(-time.Hour), Key: "c", Model: "claude-opus-5-5", Status: 200, Tokens: usage.Tokens{Output: 70}, Cost: 1.50, Priced: true},
		RequestLog{At: base.Add(-time.Hour), Model: "glm-5.3-flash", Status: 200, Tokens: usage.Tokens{Output: 9}},
	)

	page, err := store.Threads(ctx, ThreadFilter{})
	if err != nil {
		t.Fatal(err)
	}
	byKey := map[string]Thread{}
	for _, thread := range page.Threads {
		byKey[thread.Key] = thread
	}
	if a := byKey["a"]; a.Requests != 2 || a.Tokens != (usage.Tokens{Input: 110, Output: 50, CacheRead: 900}) || math.Abs(a.Cost-0.12) > 1e-9 {
		t.Fatalf("thread a = %+v", a)
	}

	overview, err := store.Overview(ctx, base.Add(-24*time.Hour), time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(overview.Cost-1.68) > 1e-9 || len(overview.Models) != 3 {
		t.Fatalf("overview cost %v, models %+v", overview.Cost, overview.Models)
	}
	opus, sol, flash := overview.Models[0], overview.Models[1], overview.Models[2]
	if opus.Model != "claude-opus-5-5" || opus.Threads != 1 || opus.PerThread() != 1.50 {
		t.Fatalf("opus = %+v", opus)
	}
	if sol.Model != "gpt-6.1-sol" || sol.Threads != 2 || sol.Requests != 3 || math.Abs(sol.PerThread()-0.09) > 1e-9 {
		t.Fatalf("sol = %+v", sol)
	}
	if flash.Model != "glm-5.3-flash" || flash.Threads != 0 || flash.Cost != 0 || flash.PerThread() != 0 {
		t.Fatalf("unpinned, unpriced flash = %+v", flash)
	}

	store.Close()
	reopened := open(t, path)
	requests, err := reopened.Requests(ctx, RequestFilter{Key: "a"})
	if err != nil || len(requests) != 2 || !requests[0].Priced || requests[0].Tokens.Total() == 0 {
		t.Fatalf("requests after reopen = %+v, %v", requests, err)
	}
	unpriced, _ := reopened.Requests(ctx, RequestFilter{Model: "glm-5.3-flash"})
	if len(unpriced) != 1 || unpriced[0].Priced || unpriced[0].Tokens.Output != 9 {
		t.Fatalf("unpriced request = %+v", unpriced)
	}
}

func TestRequestsCarryTheirTaskCostSoFar(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	pinThread(t, store, base, "a", "gpt-6.1-sol", false)
	logRequests(t, store,
		RequestLog{At: base, Key: "a", Model: "gpt-6.1-sol", Cost: 0.10, Priced: true},
		RequestLog{At: base.Add(time.Minute), Key: "b", Model: "glm-5.3", Cost: 0.50, Priced: true},
		RequestLog{At: base.Add(2 * time.Minute), Key: "a", Model: "gpt-6.1-sol"},
		RequestLog{At: base.Add(3 * time.Minute), Key: "a", Model: "gpt-6.1-sol", Cost: 0.25, Priced: true},
		RequestLog{At: base.Add(4 * time.Minute), Model: "glm-5.3-flash", Cost: 0.01, Priced: true},
	)
	want := map[int64]float64{1: 0.10, 2: 0.50, 3: 0.10, 4: 0.35, 5: 0}
	check := func(page string, requests []RequestLog) {
		t.Helper()
		if len(requests) == 0 {
			t.Fatalf("%s is empty", page)
		}
		for _, r := range requests {
			if math.Abs(r.TaskCost-want[r.ID]) > 1e-9 {
				t.Errorf("%s: request %d task cost = %v, want %v", page, r.ID, r.TaskCost, want[r.ID])
			}
		}
	}
	// A page's task costs sum the whole thread, whatever the page holds.
	for _, filter := range []RequestFilter{{Limit: 10}, {Key: "a", Limit: 1}, {Model: "gpt-6.1-sol"}, {AfterID: 2}, {AfterID: 1, Key: "a"}} {
		requests, err := store.Requests(ctx, filter)
		if err != nil {
			t.Fatal(err)
		}
		check(fmt.Sprintf("Requests(%+v)", filter), requests)
	}
	detail, err := store.Thread(ctx, "a")
	if err != nil || detail == nil {
		t.Fatalf("Thread = %+v, %v", detail, err)
	}
	check("thread a", detail.Requests)
}

func TestTaskUsagesSumSettledThreadsByDecision(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	since, settled := base.Add(-24*time.Hour), base.Add(-30*time.Minute)
	for key, pin := range map[string]Pin{
		"failed":  {Key: "failed", Model: "glm-5.3"},
		"steady":  {Key: "steady", Model: "glm-5.3"},
		"deep":    {Key: "deep", Model: "glm-5.3", Level: "high"},
		"hard":    {Key: "hard", Model: "claude-opus-5-5"},
		"running": {Key: "running", Model: "glm-5.3"},
		"idle":    {Key: "idle", Model: "claude-opus-5-5"},
		"tuned":   {Key: "tuned", Model: "glm-5.3", Level: "low"},
		"endless": {Key: "endless", Model: "gpt-6.1-sol", Level: "xhigh"},
	} {
		at(store, base.Add(-2*time.Hour))
		if _, err := store.Create(ctx, pin, Event{Key: key, Kind: Decision, Model: pin.Model}); err != nil {
			t.Fatal(err)
		}
	}
	pinThread(t, store, base.Add(-48*time.Hour), "old", "glm-5.3", false)
	if _, err := store.Failover(ctx, "failed", "glm-5.3", "gpt-6.1-sol", true, Event{Key: "failed", Kind: Failover, Model: "gpt-6.1-sol"}); err != nil {
		t.Fatal(err)
	}
	// An operator's re-pin says the decision was wrong: it teaches nothing.
	if _, err := store.Move(ctx, "tuned", "glm-5.3", "claude-opus-5-5", "xhigh", true, Event{Key: "tuned", Kind: Override, Model: "claude-opus-5-5"}); err != nil {
		t.Fatal(err)
	}
	early := base.Add(-time.Hour)
	logRequests(t, store,
		RequestLog{At: early, Key: "failed", Model: "glm-5.3", Tokens: usage.Tokens{Input: 100, Output: 10}, Cost: 0.10, Priced: true},
		RequestLog{At: early, Key: "failed", Model: "gpt-6.1-sol", Tokens: usage.Tokens{Input: 200, CacheRead: 50}, Cost: 0.20, Priced: true},
		RequestLog{At: early, Key: "failed", Model: "gpt-6.1-sol", Tokens: usage.Tokens{Output: 5, CacheWrite: 7}},
		RequestLog{At: early, Key: "steady", Model: "glm-5.3", Tokens: usage.Tokens{Input: 30}, Cost: 0.05, Priced: true},
		RequestLog{At: early, Key: "deep", Model: "glm-5.3", Tokens: usage.Tokens{Output: 40}, Cost: 1, Priced: true},
		RequestLog{At: early, Key: "hard", Model: "claude-opus-5-5", Tokens: usage.Tokens{Input: 9}, Cost: 1.5, Priced: true},
		RequestLog{At: early, Key: "running", Model: "glm-5.3", Tokens: usage.Tokens{Input: 1000}, Cost: 5, Priced: true},
		RequestLog{At: base.Add(-time.Minute), Key: "running", Model: "glm-5.3", Tokens: usage.Tokens{Input: 1000}, Cost: 5, Priced: true},
		RequestLog{At: early, Key: "old", Model: "glm-5.3", Tokens: usage.Tokens{Input: 1000}, Cost: 5, Priced: true},
		RequestLog{At: early, Key: "tuned", Model: "claude-opus-5-5", Tokens: usage.Tokens{Input: 1000}, Cost: 40, Priced: true},
		RequestLog{At: early, Key: "endless", Model: "gpt-6.1-sol", Tokens: usage.Tokens{Input: 4000, Output: 4000}, Cost: 8, Priced: true},
		RequestLog{At: early, Model: "glm-5.3-flash", Tokens: usage.Tokens{Input: 1000}, Cost: 5, Priced: true},
	)

	usages, err := store.TaskUsages(ctx, since, settled, 2000)
	if err != nil {
		t.Fatal(err)
	}
	// failed counts under its decided model, whatever it failed over to;
	// endless counts as its first 2000 tokens.
	want := []TaskUsage{
		{Model: "claude-opus-5-5", Threads: 1, Tokens: usage.Tokens{Input: 9}, Cost: 1.5},
		{Model: "glm-5.3", Threads: 2, Tokens: usage.Tokens{Input: 330, Output: 15, CacheRead: 50, CacheWrite: 7}, Cost: 0.35},
		{Model: "glm-5.3", Level: "high", Threads: 1, Tokens: usage.Tokens{Output: 40}, Cost: 1},
		{Model: "gpt-6.1-sol", Level: "xhigh", Threads: 1, Tokens: usage.Tokens{Input: 1000, Output: 1000}, Cost: 2},
	}
	if len(usages) != len(want) {
		t.Fatalf("usages = %+v, want %+v", usages, want)
	}
	for i := range want {
		got := usages[i]
		if math.Abs(got.Cost-want[i].Cost) > 1e-9 {
			t.Errorf("usage %d cost = %v, want %v", i, got.Cost, want[i].Cost)
		}
		got.Cost = want[i].Cost
		if !reflect.DeepEqual(got, want[i]) {
			t.Errorf("usage %d = %+v, want %+v", i, usages[i], want[i])
		}
	}
	if none, err := store.TaskUsages(ctx, base, base, 2000); err != nil || none == nil || len(none) != 0 {
		t.Fatalf("TaskUsages of no thread = %+v, %v", none, err)
	}
}
