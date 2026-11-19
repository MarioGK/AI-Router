package pin

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/usage"
)

func TestTaskUsagesExcludeFixedModeThreads(t *testing.T) {
	ctx := t.Context()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	at(store, base.Add(-2*time.Hour))
	for _, key := range []string{"automatic", "mixed"} {
		if _, err := store.Create(ctx, Pin{Key: key, Model: "glm-5.3", Level: "medium"},
			Event{Key: key, Kind: Decision, Model: "glm-5.3", Level: "medium"}); err != nil {
			t.Fatal(err)
		}
	}
	early := base.Add(-time.Hour)
	logRequests(t, store,
		RequestLog{At: early, Key: "automatic", Model: "glm-5.3", Reason: "decided", Status: 200,
			Tokens: usage.Tokens{Input: 100, Output: 10}, Cost: 0.25, Priced: true},
		RequestLog{At: early.Add(time.Minute), Key: "automatic", Model: "glm-5.3", Reason: "pinned", Status: 200,
			Tokens: usage.Tokens{Input: 50, CacheRead: 50}, Cost: 0.125, Priced: true},
		RequestLog{At: early, Key: "mixed", Model: "glm-5.3", Reason: "decided", Status: 200,
			Tokens: usage.Tokens{Input: 100, Output: 10}, Cost: 0.25, Priced: true},
		RequestLog{At: early.Add(time.Minute), Key: "mixed", Model: "claude-opus-5-5", Reason: "fixed", Status: 200,
			Tokens: usage.Tokens{Input: 300, Output: 40}, Cost: 2, Priced: true},
		// Returning to automatic mode keeps the original pin; it must not
		// make this mixed thread eligible for cost learning again.
		RequestLog{At: early.Add(2 * time.Minute), Key: "mixed", Model: "glm-5.3", Reason: "pinned", Status: 200,
			Tokens: usage.Tokens{Input: 50, CacheRead: 50}, Cost: 0.125, Priced: true},
	)
	at(store, base)

	usages, err := store.TaskUsages(ctx, base.Add(-24*time.Hour), base.Add(-30*time.Minute), 2000)
	if err != nil {
		t.Fatal(err)
	}
	want := TaskUsage{Model: "glm-5.3", Level: "medium", Threads: 1,
		Tokens: usage.Tokens{Input: 150, Output: 10, CacheRead: 50}, Cost: 0.375}
	if len(usages) != 1 || usages[0] != want {
		t.Fatalf("cost learning included fixed-mode traffic: %+v, want %+v", usages, want)
	}

	detail, err := store.Thread(ctx, "mixed")
	if err != nil || detail == nil {
		t.Fatalf("mixed thread = %+v, %v", detail, err)
	}
	if detail.Model != "glm-5.3" || detail.Level != "medium" || detail.Thread.Requests != 3 || len(detail.Requests) != 3 ||
		detail.Tokens != (usage.Tokens{Input: 450, Output: 50, CacheRead: 50}) || detail.Cost != 2.375 {
		t.Fatalf("fixed-mode learning exclusion changed thread accounting: %+v", detail)
	}
	if detail.Requests[1].Model != "claude-opus-5-5" || detail.Requests[1].Reason != "fixed" || detail.Requests[1].Cost != 2 {
		t.Fatalf("fixed-mode request disappeared from the log: %+v", detail.Requests)
	}
	overview, err := store.Overview(ctx, base.Add(-24*time.Hour), time.Hour)
	if err != nil || overview.Requests != 5 || overview.Cost != 2.75 {
		t.Fatalf("fixed-mode learning exclusion changed overview accounting: %+v, %v", overview, err)
	}
}
