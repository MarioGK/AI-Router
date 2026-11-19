package decide

import (
	"math"
	"reflect"
	"testing"
	"time"
)

// leaning is a verdict leaning towards the lightest step at low reasoning.
var leaning = Verdict{Choice: "deepseek-flash",
	Probabilities: map[string]float64{"deepseek-flash": 0.6, "glm-5.3": 0.3, "gpt-6.1-sol": 0.1},
	Levels:        map[string]float64{"low": 0.6, "medium": 0.3, "high": 0.1}}

func leaningChain() *Chain {
	return NewChain([]Step{{Decider: &fakeDecider{name: "jev", verdict: leaning}, Timeout: time.Second}},
		0.5, DoubtEscalate, efforts)
}

func TestAPriorWithoutWeightChangesNoDecision(t *testing.T) {
	chain := leaningChain()
	alone, ok := chain.Select(nil).Decide(t.Context(), "state", three)
	if !ok {
		t.Fatal("no decision")
	}
	// Neighbours were found, but the history has earned no weight: the
	// decision is exactly the deciders'.
	prior := &Prior{Models: map[string]float64{"gpt-6.1-sol": 1}, Levels: map[string]float64{"high": 1},
		Neighbours: 4, LevelNeighbours: 4, Similar: []Precedent{{Event: 1, Similarity: 1, Model: "gpt-6.1-sol"}}}
	learned, ok := chain.Select(nil).Learned(prior).Decide(t.Context(), "state", three)
	if !ok {
		t.Fatal("no decision")
	}
	alone.Latency, learned.Latency = 0, 0
	alone.Attempts[0].LatencyMS, learned.Attempts[0].LatencyMS = 0, 0
	if !reflect.DeepEqual(alone, learned) {
		t.Fatalf("decision with a weightless prior = %+v, want %+v", learned, alone)
	}
	if chain.Select(nil).Learned(prior).Policy(three).Fingerprint() != chain.Policy(three).Fingerprint() {
		t.Fatal("a weightless prior changed the policy fingerprint")
	}
}

func TestAWeightedPriorBlendsBeforeTheEconomy(t *testing.T) {
	chain := leaningChain()
	prior := &Prior{Models: map[string]float64{"deepseek-flash": 0.1, "glm-5.3": 0.1, "gpt-6.1-sol": 0.8},
		Levels: map[string]float64{"low": 0.1, "medium": 0.1, "high": 0.8}, Fit: 0.8, LevelFit: 0.8,
		Weight: 0.6, LevelWeight: 0.6, Neighbours: 15, LevelNeighbours: 15}
	selection := chain.Select(nil).Learned(prior)
	decision, ok := selection.Decide(t.Context(), "state", three)
	if !ok {
		t.Fatal("no decision")
	}
	// 0.4*0.1 + 0.6*0.8 = 0.52 on gpt-6.1-sol: the blend chooses it.
	if decision.Choice != "gpt-6.1-sol" || decision.Model != "gpt-6.1-sol" ||
		math.Abs(decision.Probabilities["gpt-6.1-sol"]-0.52) > 1e-9 || decision.Level != "high" {
		t.Fatalf("blended decision = %+v", decision)
	}
	// The attempt keeps what the decider said, for replays.
	attempt := decision.Attempts[0]
	if attempt.Choice != "deepseek-flash" || math.Abs(attempt.Probabilities["deepseek-flash"]-0.6) > 1e-9 || attempt.Level != "low" {
		t.Fatalf("attempt = %+v", attempt)
	}
	policy := selection.Policy(three)
	if policy.Learning == nil || policy.Learning.Fit != 0.8 || policy.Fingerprint() == chain.Policy(three).Fingerprint() {
		t.Fatalf("the policy does not record the fitted weights: %+v", policy.Learning)
	}
}

func TestAPairQuestionBlendsTheHistoryToo(t *testing.T) {
	pairer := func() *fakePairer {
		return &fakePairer{fakeDecider: fakeDecider{name: "jev", verdict: Verdict{Choice: "deepseek-flash",
			Probabilities: map[string]float64{"deepseek-flash": 0.8, "glm-5.3": 0.2}}},
			pair: Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.6, "glm-5.3": 0.4}}}
	}
	chain := func(p *fakePairer) *Chain {
		return NewChain([]Step{{Decider: p, Timeout: time.Second}}, 0.2, DoubtPair, nil)
	}
	// Alone, 40% on glm-5.3 is under the one-half cutoff.
	alone, ok := chain(pairer()).Decide(t.Context(), "state", three)
	if !ok || alone.Model != "deepseek-flash" {
		t.Fatalf("decision without history = %+v", alone)
	}
	// The history puts 80% on glm-5.3 at weight 0.4: 0.6 × 40% + 0.4 × 80%
	// is 56%, so the pair moves up.
	prior := &Prior{Models: map[string]float64{"deepseek-flash": 0.2, "glm-5.3": 0.8}, Fit: 0.8, Weight: 0.4,
		Neighbours: 5}
	learned, ok := chain(pairer()).Select(nil).Learned(prior).Decide(t.Context(), "state", three)
	if !ok || learned.Model != "glm-5.3" || len(learned.Attempts) != 2 {
		t.Fatalf("decision with history = %+v", learned)
	}
	pair := learned.Attempts[1]
	if math.Abs(pair.Above-0.4) > 1e-9 || math.Abs(pair.Blended-0.56) > 1e-9 {
		t.Fatalf("pair attempt = %+v", pair)
	}
}
