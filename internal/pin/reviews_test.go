package pin

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// decideThread pins key through a decision for model at level with a recorded
// prompt, and returns the decision's id.
func decideThread(t *testing.T, store *Store, key, model, level string, basis Basis) int64 {
	t.Helper()
	ctx := context.Background()
	if _, err := store.Create(ctx, Pin{Key: key, Model: model, Level: level}, Event{Key: key, Kind: Decision,
		Model: model, Level: level, Basis: basis, Prompt: "images=no\n\n" + key, FirstPrompt: "the whole first prompt of " + key}); err != nil {
		t.Fatal(err)
	}
	detail, err := store.Thread(ctx, key)
	if err != nil || detail.Decision == nil {
		t.Fatalf("thread %s = %+v, %v", key, detail, err)
	}
	return detail.Decision.ID
}

func TestReviewsJudgeDecisions(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	at(store, base)
	right := decideThread(t, store, "right", "glm-5.3", "medium", Basis{})
	wrong := decideThread(t, store, "wrong", "deepseek-flash", "low", Basis{})
	// A spilled decision is judged by the model the deciders chose.
	spilled := decideThread(t, store, "spilled", "gpt-6.1-sol", "high", Basis{Spilled: "glm-5.3"})
	pending := decideThread(t, store, "pending", "glm-5.3", "", Basis{})
	for _, review := range []Review{
		{Event: right, Model: "glm-5.3", Level: "medium"},
		{Event: wrong, Model: "gpt-6.1-sol", Level: "high", Note: "the race needed Sol"},
		{Event: spilled, Model: "glm-5.3", Level: "high"},
	} {
		if err := store.SaveReview(ctx, review); err != nil {
			t.Fatal(err)
		}
	}
	// A second review replaces the first.
	at(store, base.Add(time.Hour))
	if err := store.SaveReview(ctx, Review{Event: wrong, Model: "claude-opus-5-5", Level: "xhigh", Note: "cause unknown"}); err != nil {
		t.Fatal(err)
	}
	if err := store.SaveReview(ctx, Review{Event: 999, Model: "glm-5.3"}); !errors.Is(err, ErrNoDecision) {
		t.Fatalf("a review of no decision = %v", err)
	}

	detail, err := store.Thread(ctx, "wrong")
	if err != nil {
		t.Fatal(err)
	}
	review := detail.Decision.Review
	if review == nil || review.Model != "claude-opus-5-5" || review.Level != "xhigh" || review.Note != "cause unknown" ||
		!review.At.Equal(base.Add(time.Hour)) || detail.Decision.FirstPrompt != "the whole first prompt of wrong" {
		t.Fatalf("wrong decision = %+v, review %+v", detail.Decision, review)
	}
	if model, level := review.Right(*detail.Decision); model || level {
		t.Fatal("a review against the decision agrees with it")
	}

	for filter, want := range map[string][]int64{
		Right: {spilled, right}, Wrong: {wrong}, Unreviewed: {pending}, "": {pending, spilled, wrong, right},
	} {
		page, err := store.Decisions(ctx, DecisionFilter{Review: filter})
		var ids []int64
		for _, e := range page.Decisions {
			ids = append(ids, e.ID)
		}
		if err != nil || !slices.Equal(ids, want) || page.Total != len(want) {
			t.Errorf("decisions reviewed %q = %v (%d), %v; want %v", filter, ids, page.Total, err, want)
		}
	}

	summary, err := store.Reviews(ctx, time.Time{})
	if err != nil || summary.Reviewed != 3 || summary.RightModel != 2 || summary.RightLevel != 2 ||
		summary.Misses["deepseek-flash->claude-opus-5-5"] != 1 || summary.Unreviewed != 1 {
		t.Fatalf("summary = %+v, %v", summary, err)
	}
	all, err := store.Reviewed(ctx)
	if err != nil || len(all) != 3 || all[0].ID != spilled || all[2].Review == nil {
		t.Fatalf("reviewed = %+v, %v", all, err)
	}

	if err := store.DeleteReview(ctx, right); err != nil {
		t.Fatal(err)
	}
	if page, _ := store.Decisions(ctx, DecisionFilter{Review: Unreviewed}); page.Total != 2 {
		t.Fatalf("after forgetting a review %d decisions wait", page.Total)
	}
}

func TestPromptRetentionSparesReviewedDecisions(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	at(store, base)
	kept := decideThread(t, store, "kept", "glm-5.3", "", Basis{})
	decideThread(t, store, "thinned", "glm-5.3", "", Basis{})
	if err := store.SaveReview(ctx, Review{Event: kept, Model: "gpt-6.1-sol"}); err != nil {
		t.Fatal(err)
	}
	if thinned, err := store.PrunePrompts(ctx, base.Add(time.Hour)); err != nil || thinned != 1 {
		t.Fatalf("PrunePrompts = %d, %v", thinned, err)
	}
	for key, want := range map[string]string{"kept": "the whole first prompt of kept", "thinned": ""} {
		detail, err := store.Thread(ctx, key)
		if err != nil || detail.Decision.FirstPrompt != want || (want == "") != (detail.Decision.Prompt == "") {
			t.Fatalf("%s after pruning = %+v, %v", key, detail.Decision, err)
		}
	}
}
