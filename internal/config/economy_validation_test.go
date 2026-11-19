package config

import (
	"fmt"
	"strings"
	"testing"
)

func TestEconomicSettingsRejectNonFiniteValues(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	settings := []struct {
		name, document, field string
	}{
		{"miss", strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  cost: {miss: %s}", 1), "auto.cost"},
		{"weight", strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  cost: {weight: %s}", 1), "auto.cost"},
		{"time_value", strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  cost: {time_value: %s}", 1), "auto.cost"},
		{"cost_factor", strings.Replace(minimal, "{id: glm-5.3,", "{id: glm-5.3, cost_factor: %s,", 1), "cost_factor"},
		{"input", minimal + "prices:\n  glm-5.3: {input: %s}\n", "prices.glm-5.3"},
		{"output", minimal + "prices:\n  glm-5.3: {output: %s}\n", "prices.glm-5.3"},
		{"cache_read", minimal + "prices:\n  glm-5.3: {cache_read: %s}\n", "prices.glm-5.3"},
		{"cache_write", minimal + "prices:\n  glm-5.3: {cache_write: %s}\n", "prices.glm-5.3"},
		{"escalation_threshold", strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  escalation_threshold: %s", 1), "auto.escalation_threshold"},
	}
	for _, setting := range settings {
		for _, value := range []string{".nan", ".inf", "-.inf"} {
			t.Run(setting.name+"/"+value, func(t *testing.T) {
				_, err := Parse(fmt.Appendf(nil, setting.document, value))
				if err == nil || !strings.Contains(err.Error(), setting.field) || !strings.Contains(err.Error(), "finite") {
					t.Fatalf("%s = %s: expected finite-value validation, got %v", setting.name, value, err)
				}
			})
		}
	}
}

func TestEconomicSettingsAllowFiniteBoundaryValues(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	document := strings.NewReplacer(
		"default_model: glm-5.3", "default_model: glm-5.3\n  escalation_threshold: 1\n  cost: {miss: 0, weight: 0}",
		"{id: glm-5.3,", "{id: glm-5.3, cost_factor: 0,",
	).Replace(minimal) + "prices:\n  glm-5.3: {input: 0, output: 0, cache_read: 0, cache_write: 0}\n"
	cfg, err := Parse([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	if *cfg.Auto.Cost.Miss != 0 || *cfg.Auto.Cost.Weight != 0 || cfg.CostFactor("glm-5.3") != 0 || cfg.Auto.EscalationThreshold != 1 {
		t.Fatalf("finite boundary values changed: %+v", cfg.Auto)
	}
}

func TestOutcomeSettingsDefaultAndRejectNegativeCounts(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	cfg, err := Parse([]byte(minimal))
	if err != nil {
		t.Fatal(err)
	}
	if *cfg.Auto.Outcomes.Long != 200 || *cfg.Auto.Outcomes.Tiny != 3 || cfg.Auto.Cost.TimeValue != 0 {
		t.Fatalf("outcome defaults = %+v, time_value %v", cfg.Auto.Outcomes, cfg.Auto.Cost.TimeValue)
	}
	document := strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  outcomes: {long: -1}", 1)
	if _, err := Parse([]byte(document)); err == nil || !strings.Contains(err.Error(), "auto.outcomes") {
		t.Fatalf("a negative long count = %v", err)
	}
	document = strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  outcomes: {long: 0, tiny: 0}\n  cost: {time_value: 40}", 1)
	if cfg, err = Parse([]byte(document)); err != nil || *cfg.Auto.Outcomes.Long != 0 || cfg.Auto.Cost.TimeValue != 40 {
		t.Fatalf("outcomes off and time valued = %+v, %v", cfg, err)
	}
}

func TestExplorationIsOffByDefaultAndBounded(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	cfg, err := Parse([]byte(minimal))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Auto.Learn.Explore != 0 || *cfg.Auto.Learn.ExploreMargin != 0.25 || cfg.Auto.Learn.Margin() != 0.25 {
		t.Fatalf("learn defaults = %+v", cfg.Auto.Learn)
	}
	if (Learn{}).Margin() != 0.25 {
		t.Fatal("an unset margin is not the default")
	}
	for _, learn := range []string{"{explore: 0.21}", "{explore: -0.01}", "{explore: 0.05, explore_margin: -1}"} {
		document := strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  learn: "+learn, 1)
		if _, err := Parse([]byte(document)); err == nil || !strings.Contains(err.Error(), "auto.learn") {
			t.Errorf("learn %s = %v", learn, err)
		}
	}
	document := strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  learn: {explore: 0.2, explore_margin: 0.1}", 1)
	if cfg, err = Parse([]byte(document)); err != nil || cfg.Auto.Learn.Explore != 0.2 || cfg.Auto.Learn.Margin() != 0.1 {
		t.Fatalf("the most exploration = %+v, %v", cfg.Auto.Learn, err)
	}
}

func TestASingleAcceptedEffortIsAlwaysFixed(t *testing.T) {
	a := Auto{
		Reasoning: Reasoning{Levels: []Level{{ID: "low"}, {ID: "medium"}, {ID: "high"}, {ID: "xhigh"}}},
		Candidates: []Candidate{
			{ID: "gpt-6.1-sol", Efforts: []string{"high"}},
		},
	}
	for _, level := range []string{"", "unknown", "low", "medium", "high", "xhigh"} {
		if got := a.Effort("gpt-6.1-sol", level); got != "high" {
			t.Errorf("task level %q: effort = %q, want high", level, got)
		}
	}
	if got := a.Effort("not-a-candidate", "high"); got != "" {
		t.Errorf("non-candidate effort = %q, want empty", got)
	}
}
