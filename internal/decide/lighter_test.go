package decide

import (
	"math"
	"testing"
)

func TestTheStepBelowADecisionAndHowCloseItScored(t *testing.T) {
	selection := NewChain(nil, 0.10, DoubtEscalate, nil).Price(economy(1)).Select(nil)
	priced := Costs{Done: map[string]float64{"priced": 1}}
	cases := []struct {
		name     string
		decision Decision
		lighter  string
		excess   float64
		ok       bool
	}{
		// 0.10 + 0.4*0.25*2 = 0.30 against 0.25 on glm-5.3: 20% dearer.
		{"a borderline call", Decision{Model: "glm-5.3", Choice: "glm-5.3", Costs: priced,
			Probabilities: map[string]float64{"deepseek-flash": 0.6, "glm-5.3": 0.4}}, "deepseek-flash", 0.2, true},
		// The specialist is no step: 0.25 + 0.5*1.20*2 = 1.45 against 1.20.
		{"over a specialist", Decision{Model: "gpt-6.1-sol", Choice: "glm-5.3", Costs: priced,
			Probabilities: map[string]float64{"glm-5.3": 0.5, "gpt-6.1-sol": 0.5}}, "glm-5.3", 0.25 / 1.20, true},
		// Below the choice the step can score less: 0.10 against 0.25.
		{"a sure step below the choice", Decision{Model: "glm-5.3", Choice: "glm-5.3", Costs: priced,
			Probabilities: map[string]float64{"deepseek-flash": 1}}, "deepseek-flash", -0.6, true},
		{"the lightest step", Decision{Model: "deepseek-flash", Choice: "deepseek-flash", Costs: priced,
			Probabilities: map[string]float64{"deepseek-flash": 1}}, "", 0, false},
		{"a specialist", Decision{Model: "ui", Choice: "ui", Costs: priced,
			Probabilities: map[string]float64{"ui": 1}}, "", 0, false},
		{"a decision the economy did not price", Decision{Model: "glm-5.3", Choice: "glm-5.3",
			Probabilities: map[string]float64{"glm-5.3": 1}}, "", 0, false},
	}
	for _, c := range cases {
		lighter, excess, ok := selection.Lighter(ladder, c.decision)
		if lighter != c.lighter || ok != c.ok || math.Abs(excess-c.excess) > 1e-9 {
			t.Errorf("%s: %q %v %v, want %q %v %v", c.name, lighter, excess, ok, c.lighter, c.excess, c.ok)
		}
	}
	unpriced := NewChain(nil, 0.10, DoubtEscalate, nil).Select(nil)
	if _, _, ok := unpriced.Lighter(ladder, cases[0].decision); ok {
		t.Fatal("a chain without an economy scored the step below")
	}
	free := Economy{Miss: 1, Task: func(string, string, []Candidate) (float64, bool) { return 0, true }}
	lighter, excess, ok := NewChain(nil, 0.10, DoubtEscalate, nil).Price(free).Select(nil).Lighter(ladder, cases[0].decision)
	if lighter != "deepseek-flash" || excess != 0 || !ok {
		t.Fatalf("two free steps: %q %v %v", lighter, excess, ok)
	}
}
