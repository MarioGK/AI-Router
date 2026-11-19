package learn

import (
	"math"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/outcome"
	"github.com/MarioGK/AI-Router/internal/pin"
)

var ladder = []string{"deepseek-flash", "glm-5.3-flash", "glm-5.3", "gpt-6.1-sol", "claude-opus-5-5"}

var rules = outcome.Rules{Ladder: ladder, Long: 200, Tiny: 3}

var start = time.Date(2026, 9, 1, 9, 0, 0, 0, time.UTC)

// settled is how a thread decided for model at level that made requests
// turns went, after its changes.
func settled(model, level string, requests int, changes ...pin.Event) outcome.Outcome {
	return outcome.Derive(pin.ThreadActivity{Key: "k", Decided: model, Level: level, Model: model, Decision: 7,
		Requests: requests, First: start, Last: start.Add(time.Hour), Changes: changes}, rules)
}

func TestEachOutcomeLeavesItsLabel(t *testing.T) {
	cases := []struct {
		name    string
		outcome outcome.Outcome
		review  *pin.Review
		ok      bool
		want    Label
	}{
		{"a review names where it belonged", settled("glm-5.3", "low", 40,
			pin.Event{Kind: pin.Failover, Previous: "glm-5.3", Model: "gpt-6.1-sol", Detail: "upstream status 529"}),
			&pin.Review{Model: "gpt-6.1-sol", Level: "high"}, true,
			Label{Kind: Reviewed, Model: "gpt-6.1-sol", Level: "high", Weight: ReviewedWeight}},
		{"a re-pin up is where it went", settled("glm-5.3-flash", "low", 40,
			pin.Event{Kind: pin.Override, Previous: "glm-5.3-flash", Model: "gpt-6.1-sol", Level: "high"}), nil, true,
			Label{Kind: Escalated, Model: "gpt-6.1-sol", Level: "high", Weight: EscalatedWeight}},
		{"a redecision up is where it went", settled("deepseek-flash", "low", 40,
			pin.Event{Kind: outcome.Redecision, Previous: "deepseek-flash", Model: "glm-5.3", Level: "medium"}), nil, true,
			Label{Kind: Escalated, Model: "glm-5.3", Level: "medium", Weight: EscalatedWeight}},
		{"an unmoved thread sufficed", settled("glm-5.3", "medium", 40), nil, true,
			Label{Kind: Sufficed, Model: "glm-5.3", Level: "medium", Weight: SufficedWeight}},
		{"a heavy thread that stayed tiny was one step lighter", settled("claude-opus-5-5", "xhigh", 2), nil, true,
			Label{Kind: Lighter, Model: "gpt-6.1-sol", Weight: LighterWeight}},
		{"a light marathon teaches nothing", settled("deepseek-flash", "low", 500), nil, false, Label{}},
		{"a failover teaches nothing", settled("glm-5.3", "low", 40,
			pin.Event{Kind: pin.Failover, Previous: "glm-5.3", Model: "gpt-6.1-sol", Detail: "upstream status 529"}), nil, false, Label{}},
		{"a migration teaches nothing", settled("glm-5.3", "low", 40,
			pin.Event{Kind: pin.Migration, Previous: "glm-5.3", Model: "gpt-6.1-sol", Detail: "retired"}), nil, false, Label{}},
		{"a re-pin down is where the operator sent it", settled("gpt-6.1-sol", "high", 40,
			pin.Event{Kind: pin.Override, Previous: "gpt-6.1-sol", Model: "glm-5.3", Level: "low"}), nil, true,
			Label{Kind: RepinDown, Model: "glm-5.3", Level: "low", Weight: RepinDownWeight}},
		{"an escalation after a re-pin down is the last word", settled("glm-5.3", "low", 40,
			pin.Event{Kind: pin.Override, Previous: "glm-5.3", Model: "deepseek-flash", Level: "low"},
			pin.Event{Kind: outcome.Redecision, Previous: "deepseek-flash", Model: "gpt-6.1-sol", Level: "high"}), nil, true,
			Label{Kind: Escalated, Model: "gpt-6.1-sol", Level: "high", Weight: EscalatedWeight}},
		{"a redecision down teaches nothing", settled("gpt-6.1-sol", "high", 40,
			pin.Event{Kind: outcome.Redecision, Previous: "gpt-6.1-sol", Model: "glm-5.3"}), nil, false, Label{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			label, ok := Derive(c.outcome, c.review, ladder)
			if ok != c.ok {
				t.Fatalf("ok = %v, label %+v", ok, label)
			}
			if !ok {
				return
			}
			c.want.Decision, c.want.Key = 7, "k"
			if label != c.want {
				t.Fatalf("label = %+v, want %+v", label, c.want)
			}
		})
	}
}

func TestAnExploredDecisionCountsForTheOnesThatDidNot(t *testing.T) {
	cases := []struct {
		explored   bool
		propensity float64
		want       float64
	}{
		{true, 0.25, SufficedWeight * 4},
		// A rare exploration counts at most maxExplore times.
		{true, 0.01, SufficedWeight * maxExplore},
		// A decision eligible but not explored counts as any other.
		{false, 0.95, SufficedWeight},
		{true, 0, SufficedWeight},
		{false, 0, SufficedWeight},
	}
	for _, c := range cases {
		thread := pin.ThreadActivity{Key: "k", Decided: "glm-5.3", Level: "medium", Model: "glm-5.3", Decision: 7,
			Requests: 40, First: start, Last: start.Add(time.Hour), Explored: c.explored, Propensity: c.propensity}
		label, ok := Derive(outcome.Derive(thread, rules), nil, ladder)
		if !ok || math.Abs(label.Weight-c.want) > 1e-12 {
			t.Errorf("explored %v at %v: weight = %v, %v; want %v", c.explored, c.propensity, label.Weight, ok, c.want)
		}
	}
}
