package config

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/bench"
)

const minimal = `
upstream: {base_url: "http://10.11.99.1:8317"}
auto:
  default_model: glm-5.3
  candidates:
    - {id: deepseek-flash, context: 1000000, output: 64000, vision: true, description: cheap}
    - {id: glm-5.3, context: 200000, output: 128000, description: everyday}
    - {id: retired-model, context: 400000, output: 128000, vision: true, description: hard}
  fallback: {retired-model: glm-5.3}
deciders:
  - {name: jev, type: systemone, url: "https://api.typesafe.ai/v1/systemone", model: jev-latest, api_key_env: JEV_API_KEY}
  - {name: judge, type: llm, model: gpt-6-luna}
`

func TestDeciderTimeoutsFallBackToTheDecideTimeout(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	// Jev sets none and the judge sets its own, both under a 10s budget.
	yaml := strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  decide_timeout: 10s", 1)
	yaml = strings.Replace(yaml, "model: gpt-6-luna}", "model: gpt-6-luna, timeout: 2500ms}", 1)
	cfg, err := Parse([]byte(yaml))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.Deciders[0].Timeout; got != 10*time.Second {
		t.Errorf("jev timeout = %v, want auto.decide_timeout", got)
	}
	if got := cfg.Deciders[1].Timeout; got != 2500*time.Millisecond {
		t.Errorf("judge timeout = %v, want its own", got)
	}
	// The edge is allowed; a longer or non-positive one is not.
	if _, err := Parse([]byte(strings.Replace(yaml, "timeout: 2500ms", "timeout: 10s", 1))); err != nil {
		t.Errorf("a timeout equal to decide_timeout: %v", err)
	}
	for _, timeout := range []string{"10001ms", "-1s"} {
		_, err := Parse([]byte(strings.Replace(yaml, "timeout: 2500ms", "timeout: "+timeout, 1)))
		if err == nil || !strings.Contains(err.Error(), "deciders.judge: timeout must be positive and at most auto.decide_timeout (10s)") {
			t.Errorf("timeout %s: err = %v", timeout, err)
		}
	}
}

func TestParseAppliesDefaults(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	cfg, err := Parse([]byte(minimal))
	if err != nil {
		t.Fatal(err)
	}
	a := cfg.Auto
	if cfg.Listen != ":8080" || cfg.LogRetention != 720*time.Hour || a.ModelID != "auto" || a.AuxModel != "glm-5.3" ||
		a.EscalationThreshold != 0.10 || *a.DeciderTrip != 3 || len(a.FailoverStatuses) != 6 || *a.FailoverTrip != 3 || a.StateMaxChars != 12000 ||
		*a.Redecide.Idle != 5*time.Minute || *a.Redecide.GrowRequests != 60 {
		t.Fatalf("defaults = %+v", cfg)
	}
	judge := cfg.Deciders[1]
	if judge.URL != "http://10.11.99.1:8317/v1/chat/completions" || judge.APIKeyEnv != "CLIPROXYAPI_API_KEY" || judge.Timeout != 30*time.Second {
		t.Fatalf("judge defaults = %+v", judge)
	}
	if cfg.Deciders[0].Timeout != 30*time.Second || cfg.UpstreamKey() != "proxy-key" {
		t.Fatalf("jev defaults = %+v", cfg.Deciders[0])
	}
	if a.MinContext() != 200000 || a.MinOutput() != 64000 {
		t.Fatalf("limits = %d/%d", a.MinContext(), a.MinOutput())
	}
	if got := strings.Join(a.Required(), ","); got != "deepseek-flash,glm-5.3,retired-model" {
		t.Fatalf("Required = %s", got)
	}
}

func TestAnyModelCanSupplySpeedEvidence(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	document := strings.ReplaceAll(minimal, "glm-5.3", "private/vendor.new-model:v2")
	document = strings.Replace(document, "output: 128000, description: everyday", "output: 128000, expected_latency: 750ms, description: everyday", 1)
	cfg, err := Parse([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	criterion := cfg.Auto.Candidates[1].Criterion()
	if criterion.ID != "private/vendor.new-model:v2" || criterion.ExpectedLatencyMS != 750 {
		t.Fatalf("arbitrary model metadata = %+v", criterion)
	}
	if _, err := Parse([]byte(strings.Replace(document, "750ms", "-1ms", 1))); err == nil {
		t.Fatal("negative latency estimate was accepted")
	}
}

func TestTheEffortRidesTheModelSuffix(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	cfg, err := Parse([]byte(strings.Replace(minimal, "{id: glm-5.3,", "{id: glm-5.3, reasoning: xhigh,", 1)))
	if err != nil {
		t.Fatal(err)
	}
	if got := Suffixed("glm-5.3", cfg.Auto.Effort("glm-5.3", "")); got != "glm-5.3(xhigh)" {
		t.Fatalf("glm-5.3 runs as %q", got)
	}
	if got := Suffixed("retired-model", cfg.Auto.Effort("retired-model", "")); got != "retired-model" {
		t.Fatalf("retired-model runs as %q", got)
	}
	if got := Suffixed("glm-5.3-flash", cfg.Auto.Effort("glm-5.3-flash", "high")); got != "glm-5.3-flash" {
		t.Fatalf("a non-candidate runs as %q", got)
	}
	if _, err := Parse([]byte(strings.Replace(minimal, "{id: glm-5.3,", "{id: glm-5.3, reasoning: \"very high\",", 1))); err == nil {
		t.Fatal("an invalid reasoning value was accepted")
	}
}

// chooser turns the reasoning chooser on over four levels: DeepSeek takes
// the lower three at high by default, GLM the upper three, Sol stays fixed.
var chooser = strings.NewReplacer(
	"  candidates:\n", `  reasoning:
    levels:
      - {id: low, description: mechanical}
      - {id: medium, description: routine}
      - {id: high, description: tricky}
      - {id: xhigh, description: deep, examples: [design the sync protocol]}
  candidates:
`,
	"{id: deepseek-flash,", "{id: deepseek-flash, reasoning: high, efforts: [low, medium, high],",
	"{id: glm-5.3,", "{id: glm-5.3, efforts: [medium, high, xhigh],",
	"{id: retired-model,", "{id: retired-model, reasoning: xhigh,",
).Replace(minimal)

func TestEffortRunsTheThreadLevelAtTheNearestAcceptedEffort(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	cfg, err := Parse([]byte(chooser))
	if err != nil {
		t.Fatal(err)
	}
	a := &cfg.Auto
	cases := []struct{ model, level, want string }{
		{"deepseek-flash", "low", "low"},
		{"deepseek-flash", "xhigh", "high"},
		{"deepseek-flash", "", "high"},
		{"deepseek-flash", "retired", "high"},
		{"glm-5.3", "low", "medium"},
		{"glm-5.3", "xhigh", "xhigh"},
		{"glm-5.3", "", ""},
		{"retired-model", "low", "xhigh"},
		{"glm-5.3-flash", "high", ""},
	}
	for _, c := range cases {
		if got := a.Effort(c.model, c.level); got != c.want {
			t.Errorf("Effort(%s, %q) = %q, want %q", c.model, c.level, got, c.want)
		}
	}
	if got := Suffixed("glm-5.3", a.Effort("glm-5.3", "low")); got != "glm-5.3(medium)" {
		t.Fatalf("glm-5.3 at low runs as %q", got)
	}
	levels := a.Reasoning.Criteria()
	if len(levels) != 4 || levels[3].ID != "xhigh" || levels[3].Examples[0] != "design the sync protocol" {
		t.Fatalf("Criteria = %+v", levels)
	}
}

func TestEffortBreaksTiesTowardsTheStrongerEffort(t *testing.T) {
	a := Auto{
		Reasoning:  Reasoning{Levels: []Level{{ID: "low"}, {ID: "medium"}, {ID: "high"}}},
		Candidates: []Candidate{{ID: "m", Efforts: []string{"low", "high"}}},
	}
	if got := a.Effort("m", "medium"); got != "high" {
		t.Fatalf("Effort between low and high = %q, want high", got)
	}
}

func TestParseRejectsInvalidConfigs(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	cases := map[string]string{
		"not found in type":                 minimal + "surprise: true\n",
		"is not set":                        strings.Replace(minimal, "JEV_API_KEY}", "MISSING_KEY}", 1),
		"is still a candidate":              minimal[:strings.Index(minimal, "deciders:")] + "  retired: [retired-model]\n",
		"is still a fallback":               strings.Replace(minimal, "fallback: {retired-model: glm-5.3}", "fallback: {retired-model: glm-5.3-flash}\n  retired: [glm-5.3-flash]", 1),
		"duplicated":                        strings.Replace(minimal, "id: deepseek-flash", "id: glm-5.3", 1),
		"type must be":                      strings.Replace(minimal, "type: llm", "type: magic", 1),
		"escalation_threshold":              strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  escalation_threshold: 1.5", 1),
		"not a switch":                      strings.Replace(minimal, "{retired-model: glm-5.3}", "{retired-model: retired-model}", 1),
		"is not an http(s) URL":             strings.Replace(minimal, `"http://10.11.99.1:8317"`, `"ftp://10.11.99.1:8317"`, 1),
		"decider_trip must not be negative": strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  decider_trip: -1", 1),
		"prompt_retention must be positive": "log_retention: -1h\n" + minimal,
		"is not a candidate":                strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3-flash", 1),
		"decide_timeout must be positive":   strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  decide_timeout: -1s", 1),
		"redecide.idle must be 0":           strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  redecide: {idle: 30s}", 1),
		"grow_requests must not be":         strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  redecide: {grow_requests: -1}", 1),
		"200 is not an error status":        strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  failover_statuses: [200, 429]", 1),
		"a tag is empty":                    strings.Replace(minimal, "default_model: glm-5.3", "default_model: glm-5.3\n  strip_tags: [\"\"]", 1),
		"deciders is empty":                 minimal[:strings.Index(minimal, "deciders:")],
		"names the default model":           strings.Replace(minimal, "name: judge", "name: default", 1),
		"at most auto.decide_timeout":       strings.Replace(minimal, "model: gpt-6-luna}", "model: gpt-6-luna, timeout: 45s}", 1),
		"retries between 0 and 10":          "limits: {glm-5.3: {concurrency: 2, retries: 11}}\n" + minimal,
		"cap the bare model":                "limits: {\"glm-5.3(high)\": {concurrency: 2}}\n" + minimal,
		"shutdown_grace and upstream":       "shutdown_grace: 10ms\n" + minimal,
		"key_env not found":                 "dashboard: {key_env: OPERATOR_KEY}\n" + minimal,
		"auto.candidates is emp":            "upstream: {base_url: \"http://x:1\"}\nauto: {default_model: glm-5.3}\n",
		"not one of auto.reasoning.levels":  strings.Replace(chooser, "efforts: [low, medium, high]", "efforts: [low, max]", 1),
		"is not one of its efforts":         strings.Replace(chooser, "reasoning: high, efforts", "reasoning: xhigh, efforts", 1),
		"lightest first, once each":         strings.Replace(chooser, "[medium, high, xhigh]", "[medium, high, high]", 1),
		"follow auto.reasoning.levels":      strings.Replace(chooser, "[medium, high, xhigh]", "[xhigh, medium]", 1),
		"needs 2 to 10 levels":              strings.Replace(chooser, "      - {id: medium, description: routine}\n      - {id: high, description: tricky}\n      - {id: xhigh, description: deep, examples: [design the sync protocol]}\n", "", 1),
		"\"medium\" is duplicated":          strings.Replace(chooser, "{id: high, description: tricky}", "{id: medium, description: tricky}", 1),
		"\"xhigh\" has no description":      strings.Replace(chooser, "description: deep,", "description: \"\",", 1),
		"\"very high\" is not an effort":    strings.Replace(chooser, "{id: low,", "{id: \"very high\",", 1),
		"effort \"low\" is not one":         strings.Replace(minimal, "{id: glm-5.3,", "{id: glm-5.3, efforts: [low],", 1),
	}
	for want, document := range cases {
		if _, err := Parse([]byte(document)); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: err = %v", want, err)
		}
	}
}

func TestRetriesCanBeTurnedOff(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	cfg, err := Parse([]byte("limits: {glm-5.3: {concurrency: 2, retries: 0}, retired-model: {concurrency: 2}}\n" + minimal))
	if err != nil {
		t.Fatal(err)
	}
	if off, unset := *cfg.Limits["glm-5.3"].Retries, *cfg.Limits["retired-model"].Retries; off != 0 || unset != 4 {
		t.Fatalf("retries = %d and %d, want 0 and the default 4", off, unset)
	}
}

func TestLoadPrefersTheEnvironmentDocument(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	t.Setenv(EnvVar, strings.Replace(minimal, "default_model: glm-5.3", "default_model: retired-model", 1))
	cfg, err := Load(filepath.Join(t.TempDir(), "absent.yaml"))
	if err != nil || cfg.Auto.DefaultModel != "retired-model" {
		t.Fatalf("Load = %+v, %v", cfg, err)
	}
	t.Setenv(EnvVar, "")
	if _, err := Load(filepath.Join(t.TempDir(), "absent.yaml")); err == nil {
		t.Fatal("Load without the variable ignored the missing file")
	}
}

func TestSecretsFilesWinOverEnvironment(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "from-env")
	t.Setenv("JEV_API_KEY", "")
	directory := t.TempDir()
	proxy := filepath.Join(directory, "cliproxy.env")
	jev := filepath.Join(directory, "jev.env")
	if err := os.WriteFile(proxy, []byte("# managed by openbao-materializer\nCLIPROXYAPI_API_KEY=from-file\n\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(jev, []byte("export JEV_API_KEY=\"jev key\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Parse([]byte("secrets_files: [" + proxy + ", " + jev + "]\n" + minimal))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.UpstreamKey() != "from-file" || cfg.Secret("JEV_API_KEY") != "jev key" {
		t.Fatalf("secrets = %q %q", cfg.UpstreamKey(), cfg.Secret("JEV_API_KEY"))
	}
	if _, err := Parse([]byte("secrets_files: [" + filepath.Join(directory, "missing.env") + "]\n" + minimal)); err == nil {
		t.Fatal("a missing secrets file was accepted")
	}
}

func TestTheExampleConfigIsValid(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("CLIPROXYAPI_MGMT_KEY", "management-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	data, err := os.ReadFile("../../config.example.yaml")
	if err != nil {
		t.Fatal(err)
	}
	// The secrets files exist only on the router's host.
	document := regexp.MustCompile(`(?m)^secrets_files:\n(  - .*\n)+`).ReplaceAll(data, nil)
	cfg, err := Parse(document)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Auto.Reasoning.Levels) == 0 || cfg.Auto.Effort("claude-opus-5-5", "low") != "high" {
		t.Fatalf("the example config has no working reasoning chooser: %+v", cfg.Auto.Reasoning)
	}
	if cfg.Auto.DefaultModel != "gpt-6.1-sol" ||
		strings.Join(cfg.Auto.CandidateIDs(), ",") != "deepseek-flash,glm-5.3-flash,glm-5.3,gpt-6.1-sol,claude-opus-5-5" {
		t.Fatalf("the example has an unexpected default or routing ladder: %+v", cfg.Auto)
	}
	for _, level := range []string{"", "low", "medium", "high", "xhigh"} {
		if got := Suffixed("gpt-6.1-sol", cfg.Auto.Effort("gpt-6.1-sol", level)); got != "gpt-6.1-sol(high)" {
			t.Errorf("GPT-6.1 Sol at level %q runs as %q", level, got)
		}
	}
	for _, retired := range []string{"claude-sonnet-5-5"} {
		if _, ok := cfg.Auto.Candidate(retired); ok || !cfg.Auto.IsRetired(retired) ||
			cfg.Auto.Fallback[retired] != "gpt-6.1-sol" {
			t.Errorf("%s is not retired in favor of GPT-6.1 Sol", retired)
		}
		for _, required := range cfg.Auto.Required() {
			if required == retired {
				t.Errorf("the example requires retired model %s in the upstream catalog", retired)
			}
		}
	}
	if cfg.Auto.Fallback["gpt-6.1-sol"] != "glm-5.3" {
		t.Fatal("Sol has no GLM quota fallback")
	}
	cases, err := bench.Load("../../bench/routing.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		for _, model := range c.Expect {
			if cfg.Auto.Rank(model) < 0 {
				t.Errorf("benchmark %s expects model %s outside the decision ladder", c.Name, model)
			}
		}
	}
}

func TestRequiredExcludesRetiredFallbackSources(t *testing.T) {
	a := Auto{
		DefaultModel: "glm-5.3",
		AuxModel:     "glm-5.3-flash",
		Candidates:   []Candidate{{ID: "glm-5.3"}, {ID: "gpt-6.1-sol"}},
		Fallback:     map[string]string{"retired-model": "gpt-6.1-sol", "claude-opus-5-5": "gpt-6.1-sol"},
		Retired:      []string{"retired-model"},
	}
	if got := strings.Join(a.Required(), ","); got != "claude-opus-5-5,glm-5.3,glm-5.3-flash,gpt-6.1-sol" {
		t.Fatalf("Required = %s", got)
	}
}

func TestTheCostEconomyNeedsAPricedLadder(t *testing.T) {
	t.Setenv("CLIPROXYAPI_API_KEY", "proxy-key")
	t.Setenv("JEV_API_KEY", "jev-key")
	priced := strings.NewReplacer(
		"  default_model: glm-5.3\n", "  default_model: glm-5.3\n  cost: {enabled: true}\n",
		"{id: glm-5.3,", "{id: glm-5.3, cost_factor: 0.25,",
	).Replace(minimal) + "prices:\n  deepseek-flash: {input: 0.3, output: 1.2}\n  glm-5.3: {input: 1.4, output: 4.4}\n  retired-model: {input: 2, output: 10}\n"
	cfg, err := Parse([]byte(priced))
	if err != nil {
		t.Fatal(err)
	}
	cost := cfg.Auto.Cost
	if !cost.Enabled || *cost.Miss != 3 || *cost.Weight != 20 || cost.Window != 30*24*time.Hour {
		t.Fatalf("cost defaults = %+v", cost)
	}
	if cfg.CostFactor("glm-5.3") != 0.25 || cfg.CostFactor("retired-model") != 1 || cfg.CostFactor("glm-5.3-flash") != 1 {
		t.Fatal("cost factors are wrong")
	}
	if _, err := Parse([]byte(strings.Replace(priced, "  retired-model: {input: 2, output: 10}\n", "", 1))); err == nil ||
		!strings.Contains(err.Error(), `ladder candidate "retired-model" has no price`) {
		t.Fatalf("an unpriced ladder: %v", err)
	}
	if _, err := Parse([]byte(strings.Replace(priced, "cost_factor: 0.25", "cost_factor: -1", 1))); err == nil ||
		!strings.Contains(err.Error(), "cost_factor must not be negative") {
		t.Fatalf("a negative cost factor: %v", err)
	}
}
