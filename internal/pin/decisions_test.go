package pin

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
)

func TestDecisionsListFilterAndMix(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	for i, d := range []struct{ key, model, decider, prompt string }{
		{"a", "glm-5.3", "jev", "hide the nav entry"},
		{"b", "claude-opus-5-5", "jev", "design the onboarding"},
		{"c", "glm-5.3", "judge", "add a flag"},
		{"d", "glm-5.3", "jev", ""},
		{"e", "glm-5.3", "jev", "write tests"},
		{"f", "glm-5.3", "jev", "migrate the buttons"},
	} {
		at(store, base.Add(time.Duration(i)*time.Minute))
		if _, err := store.Create(ctx, Pin{Key: d.key, Model: d.model}, Event{Key: d.key, Kind: Decision, Model: d.model,
			Decider: d.decider, Choice: d.model, Prompt: d.prompt}); err != nil {
			t.Fatal(err)
		}
	}
	at(store, base.Add(time.Hour))
	if _, err := store.Failover(ctx, "a", "glm-5.3", "gpt-6.1-sol", true, Event{Key: "a", Kind: Failover, Model: "gpt-6.1-sol"}); err != nil {
		t.Fatal(err)
	}

	page, err := store.Decisions(ctx, DecisionFilter{Model: "glm-5.3", Decider: "jev", Limit: 2, Offset: 1})
	if err != nil || page.Total != 4 || len(page.Decisions) != 2 || page.Decisions[0].Key != "e" || page.Decisions[1].Key != "d" {
		t.Fatalf("decisions = %+v, %v", page, err)
	}
	if page, err := store.Decisions(ctx, DecisionFilter{Decider: "judge"}); err != nil || page.Total != 1 || page.Decisions[0].Key != "c" {
		t.Fatalf("judge decisions = %+v, %v", page, err)
	}

	mix, err := store.DecisionMix(ctx, base.Add(time.Minute))
	if err != nil || len(mix) != 2 || mix[0].Model != "glm-5.3" || mix[0].Count != 4 || mix[1].Count != 1 {
		t.Fatalf("mix = %+v, %v", mix, err)
	}
	var examples []string
	for _, e := range mix[0].Examples {
		examples = append(examples, e.Key)
	}
	// The newest three recorded prompts; d has none and a is before since.
	if !slices.Equal(examples, []string{"f", "e", "c"}) || mix[1].Examples[0].Prompt != "design the onboarding" {
		t.Fatalf("examples = %q, %+v", examples, mix[1])
	}
}

func TestDecisionMixKeepsExamplesWithTheirModel(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	// Every decision pins a model the mix has not seen yet.
	var decided atomic.Bool
	var wg sync.WaitGroup
	defer wg.Wait()
	wg.Go(func() {
		defer decided.Store(true)
		for i := range 100 {
			key, model := fmt.Sprintf("k%d", i), fmt.Sprintf("model-%d", i)
			if _, err := store.Create(ctx, Pin{Key: key, Model: model}, Event{Key: key, Kind: Decision, Model: model,
				Prompt: "add a flag"}); err != nil {
				t.Error(err)
				return
			}
		}
	})
	for done := false; !done; {
		done = decided.Load()
		mix, err := store.DecisionMix(ctx, time.Time{})
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range mix {
			if m.Count != 1 || len(m.Examples) != 1 || m.Examples[0].Model != m.Model {
				t.Fatalf("mix of %s = %+v", m.Model, m)
			}
		}
	}
}

// floodDecisions logs n decisions at the same time, each with a prompt and a
// decider's raw reply, in one statement.
func floodDecisions(t *testing.T, store *Store, n int, at time.Time) {
	t.Helper()
	_, err := store.write.Exec(`WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?)
		INSERT INTO events (at, key, kind, model, previous, decider, choice, probabilities, escalated, latency_ms, detail,
			attempts, prompt)
		SELECT ?, 'flood-' || i, 'decision', 'glm-5.3', '', 'jev', 'glm-5.3', '{}', 0, 5, '',
			'[{"decider":"jev","outcome":"decided","choice":"glm-5.3","latency_ms":5,"response":"glm-5.3"}]', 'add a flag'
		FROM n`, n, at.UnixMilli())
	if err != nil {
		t.Fatal(err)
	}
}

func TestPrunePromptsKeepsTheDecisions(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	attempts := []decide.Attempt{
		{Decider: "jev", Outcome: decide.Doubted, Model: "jev-1.13.0", Choice: "glm-5.3",
			Probabilities: map[string]float64{"glm-5.3": 0.65, "gpt-6.1-sol": 0.35}, Above: 0.35, LatencyMS: 280,
			Response: `{"choice":"glm-5.3"}`, Digest: "0a1b2c"},
		{Decider: "llm", Outcome: decide.Failed, LatencyMS: 20000, Error: "status 503"},
		{Decider: "judge", Outcome: decide.Decided, Choice: "gpt-6.1-sol", Probabilities: map[string]float64{"gpt-6.1-sol": 1},
			Reason: "Cross-cutting bug.", LatencyMS: 3500, Response: `{"reason":"Cross-cutting bug."}`},
	}
	old := base.Add(-48 * time.Hour)
	for _, d := range []struct {
		key      string
		at       time.Time
		attempts []decide.Attempt
		prompt   string
	}{
		{"old", old, attempts, "fix the race"},
		{"bare", old, []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "gpt-6.1-sol", LatencyMS: 300}}, ""},
		{"new", base, attempts, "fix the race"},
	} {
		at(store, d.at)
		if _, err := store.Create(ctx, Pin{Key: d.key, Model: "gpt-6.1-sol"}, Event{Key: d.key, Kind: Decision,
			Model: "gpt-6.1-sol", Attempts: d.attempts, Prompt: d.prompt}); err != nil {
			t.Fatal(err)
		}
	}
	// Enough old decisions to take more than one batch.
	floodDecisions(t, store, pruneBatch+5, old)

	cutoff := base.Add(-24 * time.Hour)
	if changed, err := store.PrunePrompts(ctx, cutoff); err != nil || changed != pruneBatch+6 {
		t.Fatalf("PrunePrompts = %d, %v; want %d", changed, err, pruneBatch+6)
	}
	if changed, err := store.PrunePrompts(ctx, cutoff); err != nil || changed != 0 {
		t.Fatalf("second PrunePrompts = %d, %v", changed, err)
	}

	thinned := slices.Clone(attempts)
	for i := range thinned {
		thinned[i].Response = ""
	}
	for key, want := range map[string]struct {
		attempts []decide.Attempt
		prompt   string
	}{
		"old":     {thinned, ""},
		"bare":    {[]decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "gpt-6.1-sol", LatencyMS: 300}}, ""},
		"new":     {attempts, "fix the race"},
		"flood-1": {[]decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "glm-5.3", LatencyMS: 5}}, ""},
	} {
		decisions, err := events(ctx, store.read, ` WHERE key = ?`, key)
		if err != nil || len(decisions) != 1 || decisions[0].Prompt != want.prompt || !reflect.DeepEqual(decisions[0].Attempts, want.attempts) {
			t.Fatalf("decision %s = %+v, %v", key, decisions, err)
		}
	}
	if page, err := store.Decisions(ctx, DecisionFilter{}); err != nil || page.Total != pruneBatch+8 {
		t.Fatalf("decisions left = %d, %v", page.Total, err)
	}
}

func TestPoliciesAreRecordedOnceAndFollowTheirThread(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	policy := decide.Policy{Instructions: decide.Instructions{Question: "be cheap"}, Threshold: 0.1, Doubt: decide.DoubtPair,
		Deciders:   []decide.DeciderInfo{{Name: "jev", Kind: decide.KindSystemOne, Model: "jev-latest"}},
		Candidates: []decide.Candidate{{ID: "glm-5.3", Description: "everyday"}, {ID: "claude-opus-5-5", Description: "hard"}}}
	for range 2 {
		if err := store.SavePolicy(ctx, policy); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Create(ctx, Pin{Key: "k", Model: "glm-5.3"}, Event{Key: "k", Kind: Decision, Model: "glm-5.3",
		Basis: Basis{Policy: policy.Fingerprint()}}); err != nil {
		t.Fatal(err)
	}
	detail, err := store.Thread(ctx, "k")
	if err != nil || detail.Policy == nil || !reflect.DeepEqual(*detail.Policy, policy) {
		t.Fatalf("thread policy = %+v, %v", detail, err)
	}
	if unknown, err := store.Policy(ctx, "000000000000"); err != nil || unknown != nil {
		t.Fatalf("unknown policy = %+v, %v", unknown, err)
	}
}

func TestOpenDropsPoliciesOfTheOldFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	for _, statement := range append(migrations[:7:7],
		`INSERT INTO policies VALUES ('f72644462d41', 1, '{"instructions":"Choose the cheapest model"}')`,
		`PRAGMA user_version = 7`) {
		if _, err := db.Exec(statement); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()
	if policy, err := open(t, path).Policy(context.Background(), "f72644462d41"); err != nil || policy != nil {
		t.Fatalf("old policy = %+v, %v", policy, err)
	}
}
