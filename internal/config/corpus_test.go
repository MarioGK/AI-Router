package config

import (
	"encoding/json/v2"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/bench"
	"github.com/MarioGK/AI-Router/internal/decide"
)

func TestRoutingCorporaAndFamiliesMatchTheExamplePolicy(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "test-proxy-key")
	t.Setenv("CLIPROXYAPI_MGMT_KEY", "test-management-key")
	t.Setenv("JEV_API_KEY", "test-jev-key")
	data, err := os.ReadFile("../../config.example.yaml")
	if err != nil {
		t.Fatal(err)
	}
	document := regexp.MustCompile(`(?m)^secrets_files:\n(  - .*\n)+`).ReplaceAll(data, nil)
	cfg, err := Parse(document)
	if err != nil {
		t.Fatal(err)
	}
	candidates := make([]decide.Candidate, 0, len(cfg.Auto.Candidates))
	for _, candidate := range cfg.Auto.Candidates {
		candidates = append(candidates, candidate.Criterion())
	}
	builder := decide.NewStateBuilder(cfg.Auto.StripTags, cfg.Auto.StateMaxChars)
	for _, corpus := range []string{"routing", "diagnostic"} {
		t.Run(corpus, func(t *testing.T) {
			cases, err := bench.Load("../../bench/" + corpus + ".yaml")
			if err != nil {
				t.Fatal(err)
			}
			if err := bench.Validate(cases, candidates, cfg.Auto.Reasoning.Criteria()); err != nil {
				t.Fatal(err)
			}
			_, err = bench.Prepare(cases, func(c bench.Case) bench.Input {
				input := bench.Input{State: builder.Build(c.Prompt, c.Images).Text}
				for _, candidate := range cfg.Auto.Candidates {
					if cfg.Auto.IsRetired(candidate.ID) || c.Images && !candidate.Vision || len(c.Offered) > 0 && !slices.Contains(c.Offered, candidate.ID) {
						continue
					}
					input.Candidates = append(input.Candidates, candidate.Criterion())
				}
				return input
			})
			if err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile("../../bench/" + corpus + "-families.json")
			if err != nil {
				t.Fatal(err)
			}
			var families map[string]string
			if err := json.Unmarshal(data, &families); err != nil {
				t.Fatal(err)
			}
			if len(families) != len(cases) {
				t.Fatalf("%d family entries for %d cases", len(families), len(cases))
			}
			for _, c := range cases {
				if strings.TrimSpace(families[c.Name]) == "" || c.Group == "" || c.Note == "" || len(c.Reasoning) == 0 {
					t.Errorf("case %s lacks family, group, rationale or task-depth labels", c.Name)
				}
			}
		})
	}
}
