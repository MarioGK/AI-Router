package app

import (
	"testing"

	"github.com/MarioGK/AI-Router/internal/config"
)

func TestFixedCandidateEffortWinsClientOverrides(t *testing.T) {
	a := &App{cfg: &config.Config{Auto: config.Auto{
		Reasoning: config.Reasoning{Levels: []config.Level{
			{ID: "low"}, {ID: "medium"}, {ID: "high"}, {ID: "xhigh"},
		}},
		Candidates: []config.Candidate{
			{ID: "gpt-6.1-sol", Reasoning: "high", Efforts: []string{"high"}},
		},
	}}}
	for _, level := range []string{"", "low", "medium", "high", "xhigh", "retired"} {
		for _, chosen := range []string{"", "low", "medium", "high", "xhigh", "8192"} {
			t.Run("level="+level+"/chosen="+chosen, func(t *testing.T) {
				if got := a.effort("gpt-6.1-sol", level, chosen); got != "high" {
					t.Fatalf("Sol at level %q with client effort %q runs at %q, want high", level, chosen, got)
				}
			})
		}
	}
}

func TestEffortPreservesOverridesAndThreadMapping(t *testing.T) {
	a := &App{cfg: &config.Config{Auto: config.Auto{
		Reasoning: config.Reasoning{Levels: []config.Level{
			{ID: "low"}, {ID: "medium"}, {ID: "high"}, {ID: "xhigh"},
		}},
		Candidates: []config.Candidate{
			{ID: "glm-5.3", Reasoning: "high", Efforts: []string{"medium", "high", "xhigh"}},
			{ID: "tie", Efforts: []string{"low", "high"}},
			{ID: "configured", Reasoning: "medium"},
			{ID: "single", Efforts: []string{"high"}},
		},
	}}}
	cases := []struct {
		name, model, level, chosen, want string
	}{
		{"named override", "glm-5.3", "high", "low", "low"},
		{"numeric override", "glm-5.3", "low", "8192", "8192"},
		{"nearest effort", "glm-5.3", "low", "", "medium"},
		{"accepted effort", "glm-5.3", "xhigh", "", "xhigh"},
		{"default effort", "glm-5.3", "", "", "high"},
		{"unknown level", "glm-5.3", "retired", "", "high"},
		{"stronger tie", "tie", "medium", "", "high"},
		{"configured override", "configured", "high", "xhigh", "xhigh"},
		{"configured default", "configured", "high", "", "medium"},
		{"single without default", "single", "", "low", "high"},
		{"single without chosen level", "single", "", "", "high"},
		{"auxiliary override", "auxiliary", "high", "8192", "8192"},
		{"auxiliary setting", "auxiliary", "high", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := a.effort(tc.model, tc.level, tc.chosen); got != tc.want {
				t.Fatalf("effort(%q, %q, %q) = %q, want %q", tc.model, tc.level, tc.chosen, got, tc.want)
			}
		})
	}
}
