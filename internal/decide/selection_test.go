package decide

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestDeciderSelectionKeepsIndependentFallbacksAndPolicy(t *testing.T) {
	jev := &selectionPairDecider{selectionDecider: &selectionDecider{
		info: DeciderInfo{Name: "jev", Kind: KindSystemOne, Model: "jev-latest"}, verdict: Verdict{Choice: three[0].ID},
	}}
	other := &selectionDecider{info: DeciderInfo{Name: "other-systemone", Kind: KindSystemOne, Model: "other-latest"},
		err: errors.New("unavailable")}
	broken := &selectionDecider{info: DeciderInfo{Name: "first-judge", Kind: KindLLM}, err: errors.New("unavailable")}
	fallback := &selectionPairDecider{
		selectionDecider: &selectionDecider{
			info: DeciderInfo{Name: "second-judge", Kind: KindLLM},
			verdict: Verdict{Choice: three[1].ID, Probabilities: map[string]float64{three[1].ID: 0.8, three[2].ID: 0.2},
				Levels: map[string]float64{"high": 1}},
		},
		pair: Verdict{Choice: three[1].ID},
	}
	chain := NewChain([]Step{
		{Decider: jev, Timeout: time.Second}, {Decider: other, Timeout: time.Second},
		{Decider: broken, Timeout: time.Second}, {Decider: fallback, Timeout: time.Second},
	}, 0.1, DoubtPair, efforts).Rest(1, time.Minute)
	before := chain.Health()[0]
	jevenabled, predicateCalls := false, 0
	selection := chain.Select(func(info DeciderInfo) bool {
		predicateCalls++
		isJev := info.Kind == KindSystemOne && (info.Model == "jev" || strings.HasPrefix(info.Model, "jev-"))
		return jevenabled || !isJev
	})
	// A subsequent runtime setting change cannot change this request's steps.
	jevenabled = true
	decision, ok := selection.Decide(t.Context(), "state", three)
	policy := selection.Policy(three)
	if !ok || decision.Decider != "second-judge" || decision.Model != three[1].ID || decision.Level != "high" ||
		len(decision.Attempts) != 4 || other.calls.Load() != 1 || broken.calls.Load() != 1 || fallback.calls.Load() != 1 || fallback.pairs.Load() != 1 {
		t.Fatalf("filtered fallback decision = %+v, %v", decision, ok)
	}
	if jev.calls.Load() != 0 || jev.pairs.Load() != 0 || chain.Health()[0] != before || predicateCalls != 4 {
		t.Fatalf("disabled decider ran or selection changed: calls %d, pairs %d, health %+v, predicate calls %d",
			jev.calls.Load(), jev.pairs.Load(), chain.Health()[0], predicateCalls)
	}
	want := []DeciderInfo{other.Info(), broken.Info(), fallback.Info()}
	if !slices.Equal(policy.Deciders, want) || policy.Reasoning == nil ||
		policy.Fingerprint() == chain.Policy(three).Fingerprint() || len(selection.Health()) != 3 {
		t.Fatalf("selected policy = %+v, health %+v", policy, selection.Health())
	}
	if _, ok := policy.Request("jev", "state"); ok {
		t.Fatal("selected policy rebuilt a disabled decider request")
	}
	if _, ok := policy.PairRequest("jev", "state", three[0].ID, three[1].ID); ok {
		t.Fatal("selected policy rebuilt a disabled decider pair request")
	}
}

func TestDeciderSelectionsShareCircuitHealth(t *testing.T) {
	decider := &selectionDecider{info: DeciderInfo{Name: "judge", Kind: KindLLM}, err: errors.New("unavailable")}
	chain := NewChain([]Step{{Decider: decider, Timeout: time.Second}}, 0.1, DoubtEscalate, nil).Rest(1, time.Minute)
	first := chain.Select(nil)
	first.Decide(t.Context(), "state", three)
	second := chain.Select(func(DeciderInfo) bool { return true })
	decision, ok := second.Decide(t.Context(), "state", three)
	if ok || decider.calls.Load() != 1 || len(decision.Attempts) != 1 || decision.Attempts[0].Outcome != Skipped {
		t.Fatalf("new selection forgot shared circuit health: %+v, %v, calls %d", decision, ok, decider.calls.Load())
	}
	if !slices.Equal(first.Health(), chain.Health()) || !slices.Equal(second.Health(), chain.Health()) {
		t.Fatalf("selection health differs: first %+v, second %+v, chain %+v", first.Health(), second.Health(), chain.Health())
	}
	if first.Policy(three).Fingerprint() != chain.Policy(three).Fingerprint() {
		t.Fatal("nil selection predicate changed the policy")
	}
}

func TestEmptyDeciderSelectionLeavesHealthAndEconomyIntact(t *testing.T) {
	decider := &selectionDecider{info: DeciderInfo{Name: "jev", Kind: KindSystemOne}, err: errors.New("unavailable")}
	chain := NewChain([]Step{{Decider: decider, Timeout: time.Second}}, 0.1, DoubtEscalate, efforts).
		Rest(1, time.Minute).Price(economy(1))
	chain.Decide(t.Context(), "state", three)
	before := chain.Health()
	selection := chain.Select(func(DeciderInfo) bool { return false })
	decision, ok := selection.Decide(t.Context(), "state", three)
	policy := selection.Policy(three)
	if ok || len(decision.Attempts) != 0 || decider.calls.Load() != 1 || !slices.Equal(before, chain.Health()) ||
		len(selection.Health()) != 0 || len(policy.Deciders) != 0 || policy.Miss == nil || *policy.Miss != 1 || policy.Reasoning == nil {
		t.Fatalf("empty selection changed chain: decision %+v, %v, health %+v, policy %+v", decision, ok, chain.Health(), policy)
	}
}

func TestConcurrentDeciderSelectionsDoNotChangeSharedSteps(t *testing.T) {
	jev := &selectionDecider{info: DeciderInfo{Name: "jev", Kind: KindSystemOne}, verdict: Verdict{Choice: three[0].ID}}
	judge := &selectionDecider{info: DeciderInfo{Name: "judge", Kind: KindLLM}, verdict: Verdict{Choice: three[1].ID}}
	chain := NewChain([]Step{{Decider: jev, Timeout: time.Second}, {Decider: judge, Timeout: time.Second}}, 0.1, DoubtEscalate, nil)
	filtered := chain.Select(func(info DeciderInfo) bool { return info.Kind != KindSystemOne })
	var turns sync.WaitGroup
	for range 20 {
		turns.Go(func() {
			if decision, ok := chain.Decide(t.Context(), "state", three); !ok || decision.Decider != "jev" {
				t.Errorf("full chain decision = %+v, %v", decision, ok)
			}
		})
		turns.Go(func() {
			if decision, ok := filtered.Decide(t.Context(), "state", three); !ok || decision.Decider != "judge" {
				t.Errorf("filtered chain decision = %+v, %v", decision, ok)
			}
		})
	}
	turns.Wait()
	if jev.calls.Load() != 20 || judge.calls.Load() != 20 || len(chain.Health()) != 2 || len(filtered.Health()) != 1 {
		t.Fatalf("shared steps changed: Jev calls %d, judge calls %d, full health %+v, filtered health %+v",
			jev.calls.Load(), judge.calls.Load(), chain.Health(), filtered.Health())
	}
}

type selectionDecider struct {
	info    DeciderInfo
	verdict Verdict
	err     error
	calls   atomic.Int32
}

func (d *selectionDecider) Name() string { return d.info.Name }

func (d *selectionDecider) Info() DeciderInfo { return d.info }

func (d *selectionDecider) Decide(context.Context, string, []Candidate, []Level) (Verdict, error) {
	d.calls.Add(1)
	return d.verdict, d.err
}

type selectionPairDecider struct {
	*selectionDecider
	pair  Verdict
	pairs atomic.Int32
}

func (d *selectionPairDecider) Pair(context.Context, string, Candidate, Candidate) (Verdict, error) {
	d.pairs.Add(1)
	return d.pair, nil
}
