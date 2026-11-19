package main

import (
	"bytes"
	"encoding/json/v2"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/MarioGK/AI-Router/internal/bench"
	"github.com/MarioGK/AI-Router/internal/config"
)

type benchTransport func(*http.Request) (*http.Response, error)

func (f benchTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func benchFixture(t *testing.T, cases string) (string, string) {
	t.Helper()
	t.Setenv("CLIPROXYAPI_API_KEY", "sensitive-upstream-key")
	t.Setenv("BENCH_DECIDER_KEY", "sensitive-decider-key")
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	document := `
upstream: {base_url: "https://upstream.invalid"}
auto:
  default_model: strong
  cost: {enabled: true}
  reasoning:
    levels:
      - {id: low, description: ordinary}
      - {id: high, description: difficult}
  candidates:
    - {id: cheap, description: mechanical, efforts: [low], cost_factor: 0.25}
    - {id: strong, description: difficult, vision: true, reasoning: high}
deciders:
  - {name: jev, type: systemone, url: "https://decider.invalid?credential=sensitive-url-key", model: jev-latest, api_key_env: BENCH_DECIDER_KEY}
prices:
  cheap: {input: 1, output: 2}
  strong: {input: 4, output: 8}
  jev-latest: {input: 0.1, output: 0.2}
`
	if err := os.WriteFile(configPath, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	casesPath := filepath.Join(dir, "cases.yaml")
	if err := os.WriteFile(casesPath, []byte(cases), 0o600); err != nil {
		t.Fatal(err)
	}
	return configPath, casesPath
}

func stubBenchRequests(t *testing.T, answer string) *atomic.Int64 {
	t.Helper()
	var calls atomic.Int64
	previous := http.DefaultTransport
	http.DefaultTransport = benchTransport(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		if r.URL.Host != "decider.invalid" {
			t.Errorf("unexpected request host %q", r.URL.Host)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{},
			Body: io.NopCloser(strings.NewReader(answer)), Request: r}, nil
	})
	t.Cleanup(func() { http.DefaultTransport = previous })
	return &calls
}

func TestBenchInvalidFlagsBeforeConfig(t *testing.T) {
	for _, arguments := range [][]string{
		{"--parallel", "0"}, {"--parallel", "-1"}, {"--repeat", "0"}, {"--repeat", "-2"},
		{"--format", "xml"}, {"--min-accuracy", "NaN"}, {"--min-accuracy", "+Inf"},
		{"--score", "other"},
		{"--priority", "unknown"}, {"--ranking", "cheap,cheap"},
		{"--min-accuracy", "-0.1"}, {"--min-accuracy", "1.1"},
		{"--min-group-accuracy", "NaN"}, {"--min-group-accuracy", "-Inf"},
		{"--min-group-accuracy", "-0.1"}, {"--min-group-accuracy", "1.1"},
		{"--min-joint-accuracy", "NaN"}, {"--min-joint-accuracy", "+Inf"},
		{"--min-joint-accuracy", "-0.1"}, {"--min-joint-accuracy", "1.1"},
		{"--min-choice-agreement", "NaN"}, {"--min-choice-agreement", "+Inf"},
		{"--min-choice-agreement", "-0.1"}, {"--min-choice-agreement", "1.1"},
		{"--min-joint-agreement", "NaN"}, {"--min-joint-agreement", "-Inf"},
		{"--min-joint-agreement", "-0.1"}, {"--min-joint-agreement", "1.1"},
	} {
		t.Run(strings.Join(arguments, " "), func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			args := append([]string{"bench", "--config", "/does/not/exist"}, arguments...)
			if code := run(args, &stdout, &stderr); code != 2 || !strings.Contains(stderr.String(), arguments[0]) {
				t.Fatalf("exit = %d, output = %q", code, stderr.String())
			}
			if strings.Contains(stderr.String(), "/does/not/exist") {
				t.Fatalf("loaded config before validating options: %s", stderr.String())
			}
		})
	}
}

func TestBenchAgreementSelectionBeforeConfig(t *testing.T) {
	for _, arguments := range [][]string{
		{"--min-choice-agreement", "0.5"},
		{"--min-joint-agreement", "0.5", "--reference", "teacher"},
		{"--min-choice-agreement", "0.5", "--reference", "jev", "--target-decider", "jev"},
	} {
		t.Run(strings.Join(arguments, " "), func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			args := append([]string{"bench", "--config", "/does/not/exist"}, arguments...)
			if code := run(args, &stdout, &stderr); code != 2 || !strings.Contains(stderr.String(), "agreement gates") || strings.Contains(stderr.String(), "/does/not/exist") {
				t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
			}
		})
	}
}

func TestBenchNoChainSavesCallsAndRecordsPreferences(t *testing.T) {
	configPath, casesPath := benchFixture(t, "cases:\n  - {name: case, expect: [cheap], reasoning: [low], prompt: edit}\n")
	document, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	document = bytes.Replace(document, []byte("prices:\n"), []byte("  - {name: second, type: systemone, url: https://decider.invalid, model: classifier, api_key_env: BENCH_DECIDER_KEY}\nprices:\n"), 1)
	if err := os.WriteFile(configPath, document, 0o600); err != nil {
		t.Fatal(err)
	}
	calls := stubBenchRequests(t, `{"answers":{"model":{"choice":"cheap","probabilities":{"cheap":1}},"reasoning":{"probabilities":{"low":1}}}}`)
	var stdout, stderr bytes.Buffer
	code := run([]string{"bench", "--config", configPath, "--cases", casesPath, "--no-chain", "--priority", "speed",
		"--ranking", "strong,cheap", "--format", "json"}, &stdout, &stderr)
	if code != 0 || calls.Load() != 2 {
		t.Fatalf("no-chain exit %d, calls %d, error %s", code, calls.Load(), stderr.String())
	}
	var report struct {
		Deciders []string `json:"deciders"`
		Policies map[string]struct {
			Preferences struct {
				Priority string   `json:"priority"`
				Ranking  []string `json:"ranking"`
			} `json:"preferences"`
		} `json:"policies"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Deciders) != 2 {
		t.Fatalf("unexpected chain in report: %v", report.Deciders)
	}
	for _, policy := range report.Policies {
		if policy.Preferences.Priority != "speed" || !slices.Equal(policy.Preferences.Ranking, []string{"strong", "cheap"}) {
			t.Fatalf("benchmark lost runtime preferences: %+v", policy)
		}
	}
}

func TestBenchRejectsBadCasesBeforeRequests(t *testing.T) {
	calls := stubBenchRequests(t, `{}`)
	for _, entry := range []struct {
		name  string
		entry string
	}{
		{"unknown model", "{name: case, expect: [unknown], prompt: edit}"},
		{"duplicate model", "{name: case, expect: [cheap, cheap], prompt: edit}"},
		{"unknown offered", "{name: case, expect: [cheap], offered: [cheap, unknown], prompt: edit}"},
		{"duplicate offered", "{name: case, expect: [cheap], offered: [cheap, cheap], prompt: edit}"},
		{"expected unavailable", "{name: case, expect: [cheap, strong], offered: [strong], prompt: edit}"},
		{"unknown reasoning", "{name: case, expect: [cheap], reasoning: [impossible], prompt: edit}"},
		{"duplicate reasoning", "{name: case, expect: [cheap], reasoning: [low, low], prompt: edit}"},
		{"empty cleaned state", "{name: case, expect: [cheap], prompt: '<environment_context>edit</environment_context>'}"},
		{"nonvision expected", "{name: case, expect: [cheap], images: true, prompt: inspect photo}"},
		{"no offered vision", "{name: case, expect: [cheap], offered: [cheap], images: true, prompt: inspect photo}"},
	} {
		t.Run(entry.name, func(t *testing.T) {
			configPath, casesPath := benchFixture(t, "cases:\n  - "+entry.entry+"\n")
			var stdout, stderr bytes.Buffer
			if code := run([]string{"bench", "--config", configPath, "--cases", casesPath}, &stdout, &stderr); code != 1 {
				t.Fatalf("exit = %d, output = %q", code, stderr.String())
			}
			if calls.Load() != 0 {
				t.Fatalf("made %d requests before rejecting the case", calls.Load())
			}
		})
	}
}

func TestBenchPreparationUsesImagesAvailabilityAndAppliedEffort(t *testing.T) {
	configPath, _ := benchFixture(t, "cases: []\n")
	cfg, err := config.Load(configPath)
	if err != nil {
		t.Fatal(err)
	}
	prepare := benchPreparer(cfg)
	input := prepare(bench.Case{Images: true, Prompt: "<environment_context>ignore</environment_context>\ninspect photo"})
	if input.State != "images=yes\n\ninspect photo" || len(input.Candidates) != 1 || input.Candidates[0].ID != "strong" {
		t.Fatalf("image input = %#v", input)
	}
	if input.Efforts["strong"]["low"] != "high" {
		t.Fatalf("strong effort for low level = %q", input.Efforts["strong"]["low"])
	}
	input = prepare(bench.Case{Offered: []string{"strong", "cheap"}, Prompt: "edit"})
	var ids []string
	for _, candidate := range input.Candidates {
		ids = append(ids, candidate.ID)
	}
	if !slices.Equal(ids, []string{"cheap", "strong"}) || input.Efforts["cheap"]["high"] != "low" {
		t.Fatalf("availability input = %#v", input)
	}
	cfg.Auto.Retired = []string{"cheap"}
	input = benchPreparer(cfg)(bench.Case{Offered: []string{"cheap", "strong"}, Prompt: "edit"})
	if len(input.Candidates) != 1 || input.Candidates[0].ID != "strong" {
		t.Fatalf("retired input = %#v", input)
	}
}

func TestBenchJSONIncludesTrialsEffectiveSettingsAndNoCredentials(t *testing.T) {
	calls := stubBenchRequests(t, `{"answers":{"model":{"choice":"strong","probabilities":{"strong":1}},"reasoning":{"score":0}},"usage":{"input_tokens":123,"output_tokens":7}}`)
	configPath, casesPath := benchFixture(t, "cases:\n  - {name: photo, group: visual, expect: [strong], reasoning: [low], images: true, offered: [strong], prompt: inspect photo}\n")
	var stdout, stderr bytes.Buffer
	args := []string{"bench", "--config", configPath, "--cases", casesPath, "--repeat", "2", "--format", "json", "--reference", "jev", "--min-group-accuracy", "1", "--min-joint-accuracy", "1"}
	if code := run(args, &stdout, &stderr); code != 0 {
		t.Fatalf("exit = %d, stderr = %s, stdout = %s", code, stderr.String(), stdout.String())
	}
	if calls.Load() != 2 {
		t.Fatalf("requests = %d, want 2", calls.Load())
	}
	var document map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &document); err != nil {
		t.Fatalf("report JSON: %v: %s", err, stdout.String())
	}
	for _, key := range []string{"settings", "cases", "outcomes", "inputs", "policies", "metrics"} {
		if _, ok := document[key]; !ok {
			t.Fatalf("report has no %q: %s", key, stdout.String())
		}
	}
	if document["schema_version"] != float64(2) {
		t.Fatalf("schema version = %v", document["schema_version"])
	}
	if document["reference"] != "jev" {
		t.Fatalf("reference = %v", document["reference"])
	}
	rows := document["cases"].([]any)
	for i, row := range rows {
		c := row.(map[string]any)
		if c["name"] != "photo" || c["trial"] != float64(i+1) {
			t.Fatalf("trial %d = %v", i, c)
		}
	}
	outcomes := document["outcomes"].([]any)
	for _, row := range outcomes {
		outcome := row.([]any)[0].(map[string]any)
		if outcome["task_depth"] != "low" || outcome["applied_effort"] != "high" || outcome["started"] != true {
			t.Fatalf("outcome = %v", outcome)
		}
		if len(outcome["attempts"].([]any)) == 0 || len(outcome["routing_costs"].(map[string]any)) == 0 {
			t.Fatalf("outcome omits attempts or costs: %v", outcome)
		}
	}
	settings := document["settings"].(map[string]any)
	cost := settings["cost"].(map[string]any)
	if cost["estimate_source"] != "typical_task_prior" || len(cost["task_estimates"].([]any)) != 4 {
		t.Fatalf("cost settings = %v", cost)
	}
	if settings["decide_timeout_ms"] != float64(30_000) || settings["run_timeout_ms"] != float64(1_800_000) {
		t.Fatalf("budgets = %v", settings)
	}
	for _, sensitive := range []string{"sensitive-upstream-key", "sensitive-decider-key", "sensitive-url-key", "BENCH_DECIDER_KEY", "decider.invalid", "upstream.invalid"} {
		if strings.Contains(stdout.String(), sensitive) {
			t.Fatalf("report contains credential or endpoint %q", sensitive)
		}
	}
}

func TestBenchJointGateNeedsLabelsBeforeRequests(t *testing.T) {
	calls := stubBenchRequests(t, `{}`)
	configPath, casesPath := benchFixture(t, "cases:\n  - {name: edit, expect: [cheap], prompt: edit}\n")
	var stdout, stderr bytes.Buffer
	if code := run([]string{"bench", "--config", configPath, "--cases", casesPath, "--min-joint-accuracy", "0"}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "reasoning labels") {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	if calls.Load() != 0 {
		t.Fatalf("requests = %d", calls.Load())
	}
}

func TestBenchUnknownReferenceBeforeRequests(t *testing.T) {
	calls := stubBenchRequests(t, `{}`)
	configPath, casesPath := benchFixture(t, "cases:\n  - {name: edit, expect: [cheap], prompt: edit}\n")
	for _, name := range []string{"unknown", "jev ", "chain(escalate)"} {
		t.Run(name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := run([]string{"bench", "--config", configPath, "--cases", casesPath, "--reference", name}, &stdout, &stderr); code != 2 || !strings.Contains(stderr.String(), "not a configured decider") {
				t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
			}
			if calls.Load() != 0 {
				t.Fatalf("requests = %d", calls.Load())
			}
		})
	}
}

func TestBenchUnknownTargetBeforeRequests(t *testing.T) {
	calls := stubBenchRequests(t, `{}`)
	configPath, casesPath := benchFixture(t, "cases:\n  - {name: edit, expect: [cheap], prompt: edit}\n")
	var stdout, stderr bytes.Buffer
	if code := run([]string{"bench", "--config", configPath, "--cases", casesPath, "--reference", "jev", "--target-decider", "unknown", "--min-choice-agreement", "0.9"}, &stdout, &stderr); code != 2 || !strings.Contains(stderr.String(), "--target-decider") {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	if calls.Load() != 0 {
		t.Fatalf("requests = %d", calls.Load())
	}
}

func TestBenchJointAgreementNeedsConfiguredDepthBeforeRequests(t *testing.T) {
	calls := stubBenchRequests(t, `{}`)
	cfg := &config.Config{Deciders: []config.Decider{{Name: "jev"}, {Name: "teacher"}}}
	options := benchOptions{Parallel: 1, Repeat: 1, Format: "text", Score: "final", Reference: "teacher", TargetDecider: "jev", MinJointAgreement: new(0.5)}
	var stdout, stderr bytes.Buffer
	if code := runBench(cfg, "/does/not/exist", options, &stdout, &stderr); code != 2 || !strings.Contains(stderr.String(), "configured reasoning chooser") || strings.Contains(stderr.String(), "/does/not/exist") {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	if calls.Load() != 0 {
		t.Fatalf("requests = %d", calls.Load())
	}
}

func TestBenchReferenceRejectsSyntheticChainCollisionBeforeRequests(t *testing.T) {
	calls := stubBenchRequests(t, `{}`)
	configPath, casesPath := benchFixture(t, "cases:\n  - {name: edit, expect: [cheap], prompt: edit}\n")
	document, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Replace(string(document), "{name: jev,", "{name: 'chain(escalate)',", 1)
	text = strings.Replace(text, "\nprices:", "\n  - {name: fallback, type: systemone, url: 'https://decider.invalid', model: jev-latest, api_key_env: BENCH_DECIDER_KEY}\nprices:", 1)
	if err := os.WriteFile(configPath, []byte(text), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := run([]string{"bench", "--config", configPath, "--cases", casesPath, "--reference", "chain(escalate)"}, &stdout, &stderr); code != 2 || !strings.Contains(stderr.String(), "conflicts with the benchmark") {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	if calls.Load() != 0 {
		t.Fatalf("requests = %d", calls.Load())
	}
}

func TestBenchGatesAndIncompleteRun(t *testing.T) {
	report := bench.Report{Cases: []bench.Case{
		{Name: "first", Group: "easy", Expect: []string{"cheap"}, Reasoning: []string{"low"}},
		{Name: "second", Group: "hard", Expect: []string{"strong"}, Reasoning: []string{"high"}},
	}, Deciders: []string{"chain"}, Outcomes: [][]bench.Outcome{
		{{Started: true, Model: "cheap", Level: "low"}},
		{{Started: true, Model: "cheap", Level: "high"}},
	}}
	options := benchOptions{Parallel: 1, Repeat: 1, Format: "text", MinAccuracy: 0.6}
	var stderr bytes.Buffer
	if code := benchExit(report, options, &stderr); code != 1 || !strings.Contains(stderr.String(), "routed 50% of the cases correctly, below 60%") {
		t.Fatalf("model gate exit = %d, stderr = %s", code, stderr.String())
	}
	options.MinAccuracy = 0
	options.MinGroupAccuracy = new(0.1)
	stderr.Reset()
	if code := benchExit(report, options, &stderr); code != 1 || !strings.Contains(stderr.String(), `group "hard"`) {
		t.Fatalf("group gate exit = %d, stderr = %s", code, stderr.String())
	}
	options.MinGroupAccuracy = nil
	options.MinJointAccuracy = new(0.6)
	stderr.Reset()
	if code := benchExit(report, options, &stderr); code != 1 || !strings.Contains(stderr.String(), "model and reasoning") {
		t.Fatalf("joint gate exit = %d, stderr = %s", code, stderr.String())
	}
	options.MinJointAccuracy = nil
	report.Outcomes[1][0] = bench.Outcome{Err: "context canceled", Cancelled: true}
	stderr.Reset()
	if code := benchExit(report, options, &stderr); code != 1 || !strings.Contains(stderr.String(), "incomplete") {
		t.Fatalf("incomplete exit = %d, stderr = %s", code, stderr.String())
	}
}

func TestBenchInitialScoreAndTargetDoNotUseFinalOrFallback(t *testing.T) {
	report := bench.Report{Cases: []bench.Case{{Name: "hard", Group: "hard", Expect: []string{"strong"}, Reasoning: []string{"high"}}},
		Deciders: []string{"jev", "teacher", "chain(escalate)"}, Outcomes: [][]bench.Outcome{{
			{Started: true, Choice: "cheap", Model: "strong", Level: "high"},
			{Started: true, Choice: "strong", Model: "strong", Level: "high"},
			{Started: true, Choice: "strong", Model: "strong", Level: "high"},
		}}}
	options := benchOptions{TargetDecider: "jev", Score: "initial", MinAccuracy: 1, MinGroupAccuracy: new(1.0), MinJointAccuracy: new(1.0)}
	var stderr bytes.Buffer
	if code := benchExit(report, options, &stderr); code != 1 || !strings.Contains(stderr.String(), "jev selected the initial model") || !strings.Contains(stderr.String(), `group "hard"`) || !strings.Contains(stderr.String(), "model and reasoning") {
		t.Fatalf("initial gate exit = %d, stderr = %s", code, stderr.String())
	}
	options.Score = "final"
	stderr.Reset()
	if code := benchExit(report, options, &stderr); code != 0 {
		t.Fatalf("final gate exit = %d, stderr = %s", code, stderr.String())
	}
	report.Outcomes[0][0].Model = "cheap"
	stderr.Reset()
	if code := benchExit(report, options, &stderr); code != 1 || !strings.Contains(stderr.String(), "jev routed 0%") {
		t.Fatalf("target gate exit = %d, stderr = %s", code, stderr.String())
	}
	options.TargetDecider = ""
	stderr.Reset()
	if code := benchExit(report, options, &stderr); code != 0 {
		t.Fatalf("default chain gate exit = %d, stderr = %s", code, stderr.String())
	}
}

func TestBenchAgreementGatesRequireFullCoverageAndRawChoice(t *testing.T) {
	report := bench.Report{Cases: []bench.Case{
		{Name: "one", Expect: []string{"strong"}}, {Name: "two", Expect: []string{"strong"}},
	}, Deciders: []string{"jev", "teacher"}, Outcomes: [][]bench.Outcome{
		{{Started: true, Choice: "strong", Model: "strong", Level: "high"}, {Started: true, Choice: "strong", Model: "strong", Level: "high"}},
		{{Started: true, Err: "classifier failed"}, {Started: true, Choice: "strong", Model: "strong", Level: "high"}},
	}}
	options := benchOptions{TargetDecider: "jev", Reference: "teacher", MinChoiceAgreement: new(1.0)}
	var stderr bytes.Buffer
	if code := benchExit(report, options, &stderr); code != 1 || !strings.Contains(stderr.String(), "comparison coverage 1/2") {
		t.Fatalf("choice coverage exit = %d, stderr = %s", code, stderr.String())
	}
	report.Outcomes[1][0] = bench.Outcome{Started: true, Choice: "strong", Model: "strong"}
	options.MinChoiceAgreement = nil
	options.MinJointAgreement = new(1.0)
	stderr.Reset()
	if code := benchExit(report, options, &stderr); code != 1 || !strings.Contains(stderr.String(), "comparison coverage 1/2") {
		t.Fatalf("joint coverage exit = %d, stderr = %s", code, stderr.String())
	}
	report.Outcomes[1][0] = bench.Outcome{Started: true, Choice: "cheap", Model: "strong", Level: "high"}
	options.MinChoiceAgreement = new(0.9)
	stderr.Reset()
	if code := benchExit(report, options, &stderr); code != 1 || !strings.Contains(stderr.String(), "50% of raw choice") || !strings.Contains(stderr.String(), "50% of joint raw choice") {
		t.Fatalf("raw agreement exit = %d, stderr = %s", code, stderr.String())
	}
	report.Outcomes[1][0].Choice = "strong"
	stderr.Reset()
	if code := benchExit(report, options, &stderr); code != 0 {
		t.Fatalf("agreement exit = %d, stderr = %s", code, stderr.String())
	}
}
