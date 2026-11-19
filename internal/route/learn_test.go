package route

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/thread"
	"github.com/MarioGK/AI-Router/internal/wire"
)

// fakeLearner suggests the same prior for every prompt and remembers what
// it was asked.
type fakeLearner struct {
	prior              decide.Prior
	prompt, key        string
	candidates, levels []string
}

func (l *fakeLearner) Prior(_ context.Context, prompt, key string, candidates, levels []string) *decide.Prior {
	l.prompt, l.key, l.candidates, l.levels = prompt, key, candidates, levels
	prior := l.prior
	return &prior
}

// slowLearner searches until its time is up.
type slowLearner struct{ waited time.Duration }

func (l *slowLearner) Prior(ctx context.Context, _, _ string, _, _ []string) *decide.Prior {
	start := time.Now()
	<-ctx.Done()
	l.waited = time.Since(start)
	return &decide.Prior{}
}

func TestASlowHistoryLeavesTheDecidersTheirWholeTimeout(t *testing.T) {
	h := exploreResolver(t, 0, 0)
	learner := &slowLearner{}
	h.r.Learn(learner)
	event := h.decide(t, false)
	// decide_timeout is 1s: the search gets a quarter of it, and the
	// deciders still decide.
	if event.Decider != "verdict" || event.Model != "middle" || learner.waited > 400*time.Millisecond {
		t.Fatalf("decision = %s by %s after a %v search", event.Model, event.Decider, learner.waited)
	}
}

func TestFixedModeAdvertisesThePolicyItRunsUnder(t *testing.T) {
	h := exploreResolver(t, 0, 0)
	h.r.Learn(&fakeLearner{prior: decide.Prior{Fit: 0.8, LevelFit: 0.4}})
	if _, err := h.r.SetSettings(t.Context(), Settings{Mode: ModeFixed, FixedModel: "middle"}); err != nil {
		t.Fatal(err)
	}
	advertised := h.r.Policy(false)
	_, tried := h.r.Try(t.Context(), "Fix the flaky retry test.", false)
	if advertised.Learning != nil || advertised.Fingerprint() != tried.Fingerprint() {
		t.Fatalf("fixed mode advertises %+v, runs under %+v", advertised.Learning, tried.Learning)
	}
}

func TestADecisionBlendsInWhatTheHistorySuggests(t *testing.T) {
	h := exploreResolver(t, 0, 0)
	plain := h.decide(t, false)
	learner := &fakeLearner{prior: decide.Prior{Models: map[string]float64{"light": 0, "middle": 0, "heavy": 1},
		Neighbours: 20, Fit: 0.8, Weight: 0.64,
		Similar: []decide.Precedent{{Event: 3, Similarity: 1, Model: "heavy", Weight: 1}}}}
	h.r.Learn(learner)
	learned := h.decide(t, false)
	// 0.36 × 0.6 on middle against 0.64 on heavy: the history moves it up.
	if plain.Model != "middle" || learned.Model != "heavy" || learned.Choice != "heavy" ||
		learned.Attempts[0].Choice != "middle" {
		t.Fatalf("decisions = %s, then %s (choice %s, attempt %s)", plain.Model, learned.Model, learned.Choice,
			learned.Attempts[0].Choice)
	}
	if learned.Basis.Prior == nil || learned.Basis.Prior.Weight != 0.64 || len(learned.Basis.Prior.Similar) != 1 ||
		learned.Basis.Policy == plain.Basis.Policy {
		t.Fatalf("basis = %+v", learned.Basis)
	}
	if learner.prompt != "Fix the flaky retry test." || !slices.Equal(learner.candidates, []string{"light", "middle", "heavy", "special"}) {
		t.Fatalf("the learner was asked about %q among %v", learner.prompt, learner.candidates)
	}
	// A thread's own decisions are left out by its pin key.
	h.r.Resolve(t.Context(), wire.Messages, thread.Identity{Session: "s"}, wire.Conversation{
		LatestUser: "Fix the flaky retry test.", HasTools: true, Prompted: true})
	if learner.key != h.r.key("s", "") {
		t.Fatalf("the learner left out %q", learner.key)
	}
	// Neighbours without weight change nothing and leave the policy as it
	// was; the prior is still recorded.
	learner.prior.Fit, learner.prior.Weight = 0, 0
	weightless := h.decide(t, false)
	if weightless.Model != "middle" || weightless.Basis.Policy != plain.Basis.Policy || weightless.Basis.Prior == nil {
		t.Fatalf("weightless decision = %s under %s, prior %+v", weightless.Model, weightless.Basis.Policy,
			weightless.Basis.Prior)
	}
	// Without neighbours nothing is recorded.
	learner.prior.Similar = nil
	if alone := h.decide(t, false); alone.Basis.Prior != nil {
		t.Fatalf("a prior without neighbours was recorded: %+v", alone.Basis.Prior)
	}
}
