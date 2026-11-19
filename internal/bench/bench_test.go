package bench

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
)

type keyword struct {
	name  string
	route map[string]string
}

func (k keyword) Name() string { return k.name }

func (k keyword) Info() decide.DeciderInfo { return decide.DeciderInfo{Name: k.name, Kind: "keyword"} }

func (k keyword) Decide(_ context.Context, state string, _ []decide.Candidate, _ []decide.Level) (decide.Verdict, error) {
	for word, model := range k.route {
		if strings.Contains(state, word) {
			return decide.Verdict{Choice: model}, nil
		}
	}
	return decide.Verdict{}, errors.New("status 503: unavailable")
}

func TestRunScoresAgreementAndErrors(t *testing.T) {
	candidates := []decide.Candidate{{ID: "deepseek-flash"}, {ID: "gpt-6.1-sol"}, {ID: "claude-opus-5-5", Specialist: true}}
	cases := []Case{
		{Name: "small", Expect: []string{"deepseek-flash"}, Prompt: "rename x"},
		{Name: "ui", Expect: []string{"claude-opus-5-5"}, Prompt: "style the button"},
		{Name: "either", Expect: []string{"deepseek-flash", "gpt-6.1-sol"}, Prompt: "add a field"},
	}
	var chains []Chain
	for _, decider := range []keyword{
		{"a", map[string]string{"rename": "deepseek-flash", "button": "claude-opus-5-5", "field": "gpt-6.1-sol"}},
		{"b", map[string]string{"rename": "deepseek-flash", "button": "gpt-6.1-sol"}},
	} {
		chains = append(chains, Chain{Name: decider.name, Chain: decide.NewChain([]decide.Step{{Decider: decider, Timeout: time.Second}}, 0.10, decide.DoubtEscalate, nil)})
	}
	report := Run(t.Context(), chains, func(c Case) Input {
		return Input{State: c.Prompt, Candidates: candidates}
	}, cases, 2, time.Second)
	if !report.Correct(0, 0) || !report.Correct(1, 0) || !report.Correct(2, 0) || report.Correct(1, 1) || report.Correct(2, 1) {
		t.Fatalf("outcomes = %+v", report.Outcomes)
	}
	if !report.Agree(0) || report.Agree(1) || report.Agree(2) {
		t.Fatal("agreement is wrong")
	}
	var out bytes.Buffer
	report.Write(&out)
	for _, want := range []string{"MISS gpt-6.1-sol", "ERROR", "a        3/3      100%      3/3", "b        1/3", "agreement: 1/3", "error b/either: b: status 503"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("report lacks %q:\n%s", want, out.String())
		}
	}
}

func TestLoadValidatesCases(t *testing.T) {
	directory := t.TempDir()
	good := filepath.Join(directory, "good.yaml")
	_ = os.WriteFile(good, []byte("cases:\n  - {name: a, expect: [glm-5.3], prompt: hi}\n"), 0o600)
	if cases, err := Load(good); err != nil || len(cases) != 1 {
		t.Fatalf("Load = %v, %v", cases, err)
	}
	bad := filepath.Join(directory, "bad.yaml")
	_ = os.WriteFile(bad, []byte("cases:\n  - {name: a, prompt: hi}\n"), 0o600)
	if _, err := Load(bad); err == nil {
		t.Fatal("a case without expect was accepted")
	}
}

func TestShippedCasesLoad(t *testing.T) {
	cases, err := Load("../../bench/routing.yaml")
	if err != nil || len(cases) < 40 {
		t.Fatalf("bench/routing.yaml: %d cases, %v", len(cases), err)
	}
}

func TestLoadRejectsMalformedLabelsAndDocuments(t *testing.T) {
	for name, data := range map[string]string{
		"blank name":         "cases: [{name: ' ', expect: [m], prompt: hi}]",
		"blank expectation":  "cases: [{name: a, expect: [''], prompt: hi}]",
		"padded label":       "cases: [{name: a, expect: [' m'], prompt: hi}]",
		"duplicate model":    "cases: [{name: a, expect: [m, m], prompt: hi}]",
		"blank reasoning":    "cases: [{name: a, expect: [m], reasoning: [' '], prompt: hi}]",
		"duplicate level":    "cases: [{name: a, expect: [m], reasoning: [low, low], prompt: hi}]",
		"duplicate offered":  "cases: [{name: a, expect: [m], offered: [m, m], prompt: hi}]",
		"unknown field":      "cases: [{name: a, expect: [m], reasonng: [low], prompt: hi}]",
		"duplicate name":     "cases: [{name: a, expect: [m], prompt: hi}, {name: a, expect: [m], prompt: bye}]",
		"second document":    "cases: [{name: a, expect: [m], prompt: hi}]\n---\ncases: []\n",
		"malformed trailing": "cases: [{name: a, expect: [m], prompt: hi}]\n---\n[\n",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "cases.yaml")
			if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(path); err == nil {
				t.Fatal("malformed cases were accepted")
			}
		})
	}
}

func TestValidateLabelsAgainstConfiguration(t *testing.T) {
	candidates := []decide.Candidate{{ID: "m"}, {ID: "n"}}
	levels := []decide.Level{{ID: "low"}, {ID: "high"}}
	good := Case{Name: "a", Expect: []string{"m"}, Reasoning: []string{"low"}, Offered: []string{"m", "n"}, Prompt: "hi"}
	if err := Validate([]Case{good}, candidates, levels); err != nil {
		t.Fatal(err)
	}
	for name, modify := range map[string]func(*Case){
		"unknown expectation": func(c *Case) { c.Expect = []string{"typo"} },
		"unknown offered":     func(c *Case) { c.Offered = []string{"m", "typo"} },
		"unknown level":       func(c *Case) { c.Reasoning = []string{"typo"} },
		"unoffered expected":  func(c *Case) { c.Offered = []string{"n"} },
		"negative trial":      func(c *Case) { c.Trial = -1 },
	} {
		t.Run(name, func(t *testing.T) {
			c := good
			modify(&c)
			if err := Validate([]Case{c}, candidates, levels); err == nil {
				t.Fatal("invalid labels were accepted")
			}
		})
	}
	if err := Validate([]Case{good}, candidates, nil); err == nil {
		t.Fatal("reasoning labels accepted with chooser disabled")
	}
	if err := Validate([]Case{good, good}, candidates, levels); err == nil {
		t.Fatal("duplicate case identity accepted")
	}
	if err := Validate(nil, candidates, levels); err == nil {
		t.Fatal("empty benchmark accepted")
	}
}

func TestRepeatPreservesIdentityAndSnapshotsLabels(t *testing.T) {
	cases := []Case{{Name: "a", Group: "tiny", Expect: []string{"m"}, Reasoning: []string{"low"}, Offered: []string{"m"}, Prompt: "hi"}}
	trials := Repeat(cases, 3)
	if len(trials) != 3 {
		t.Fatalf("trials = %+v", trials)
	}
	for i, trial := range trials {
		if trial.Name != "a" || trial.Group != "tiny" || trial.Trial != i+1 {
			t.Errorf("trial %d = %+v", i, trial)
		}
	}
	if err := Validate(trials, []decide.Candidate{{ID: "m"}}, []decide.Level{{ID: "low"}}); err != nil {
		t.Fatal(err)
	}
	trials[0].Expect[0], trials[0].Reasoning[0], trials[0].Offered[0] = "changed", "changed", "changed"
	if cases[0].Expect[0] != "m" || trials[1].Expect[0] != "m" || trials[1].Reasoning[0] != "low" || trials[1].Offered[0] != "m" {
		t.Fatal("repeated labels alias their source or another trial")
	}
	if len(Repeat(cases, 0)) != 0 {
		t.Fatal("zero repetitions must yield no trials")
	}
}

func TestPrepareSnapshotsReusedBuffers(t *testing.T) {
	cases := []Case{{Name: "a", Expect: []string{"m"}, Prompt: "first"}, {Name: "b", Expect: []string{"n"}, Prompt: "second"}}
	candidates := []decide.Candidate{{Examples: []string{""}}}
	efforts := map[string]map[string]string{"shared": {"low": ""}}
	inputs, err := Prepare(cases, func(c Case) Input {
		candidates[0].ID, candidates[0].Examples[0] = c.Expect[0], c.Prompt
		efforts["shared"]["low"] = c.Prompt
		return Input{State: c.Prompt, Candidates: candidates, Efforts: efforts}
	})
	if err != nil {
		t.Fatal(err)
	}
	candidates[0].ID, candidates[0].Examples[0], efforts["shared"]["low"] = "changed", "changed", "changed"
	for i, want := range []string{"first", "second"} {
		if inputs[i].State != want || inputs[i].Candidates[0].ID != cases[i].Expect[0] || inputs[i].Candidates[0].Examples[0] != want || inputs[i].Efforts["shared"]["low"] != want {
			t.Errorf("input %d = %+v", i, inputs[i])
		}
	}
}

func TestPrepareRejectsUnclassifiableCases(t *testing.T) {
	c := Case{Name: "a", Expect: []string{"m", "n"}, Prompt: "hi"}
	for name, input := range map[string]Input{
		"empty state":         {State: "  ", Candidates: []decide.Candidate{{ID: "m"}, {ID: "n"}}},
		"no candidates":       {State: "hi"},
		"missing alternative": {State: "hi", Candidates: []decide.Candidate{{ID: "m"}}},
		"duplicate candidate": {State: "hi", Candidates: []decide.Candidate{{ID: "m"}, {ID: "m"}, {ID: "n"}}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Prepare([]Case{c}, func(Case) Input { return input }); err == nil {
				t.Fatal("unclassifiable case accepted")
			}
		})
	}
	if _, err := Prepare([]Case{c}, nil); err == nil {
		t.Fatal("nil input preparation accepted")
	}
	var calls atomic.Int64
	chain := testChain(functionDecider{call: func(context.Context, string, []decide.Candidate, []decide.Level) (decide.Verdict, error) {
		calls.Add(1)
		return decide.Verdict{Choice: "m"}, nil
	}})
	report := Run(t.Context(), []Chain{chain}, func(Case) Input { return Input{} }, []Case{c}, 2, time.Second)
	if calls.Load() != 0 || report.Outcomes[0][0].Started || report.Outcomes[0][0].Err == "" {
		t.Fatalf("invalid preparation classified: calls=%d outcome=%+v", calls.Load(), report.Outcomes[0][0])
	}
}

type functionDecider struct {
	call func(context.Context, string, []decide.Candidate, []decide.Level) (decide.Verdict, error)
}

func (functionDecider) Name() string { return "probe" }
func (functionDecider) Info() decide.DeciderInfo {
	return decide.DeciderInfo{Name: "probe", Kind: "test"}
}
func (d functionDecider) Decide(ctx context.Context, state string, candidates []decide.Candidate, levels []decide.Level) (decide.Verdict, error) {
	return d.call(ctx, state, candidates, levels)
}

func testChain(decider decide.Decider) Chain {
	return Chain{Name: decider.Name(), Chain: decide.NewChain([]decide.Step{{Decider: decider, Timeout: time.Second}}, 0.1, decide.DoubtEscalate, nil)}
}

func testInput(c Case) Input {
	return Input{State: c.Prompt, Candidates: []decide.Candidate{{ID: "m"}}}
}

func TestRunBoundsConcurrencyAndGivesQueuedCallsTheirBudget(t *testing.T) {
	cases := Repeat([]Case{{Name: "a", Expect: []string{"m"}, Prompt: "hi"}}, 6)
	var active, peak atomic.Int64
	decider := functionDecider{call: func(ctx context.Context, _ string, _ []decide.Candidate, _ []decide.Level) (decide.Verdict, error) {
		n := active.Add(1)
		defer active.Add(-1)
		for previous := peak.Load(); n > previous; previous = peak.Load() {
			if peak.CompareAndSwap(previous, n) {
				break
			}
		}
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) < 80*time.Millisecond {
			return decide.Verdict{}, errors.New("queued call lost its own budget")
		}
		timer := time.NewTimer(40 * time.Millisecond)
		defer timer.Stop()
		select {
		case <-timer.C:
			return decide.Verdict{Choice: "m"}, nil
		case <-ctx.Done():
			return decide.Verdict{}, ctx.Err()
		}
	}}
	report := Run(t.Context(), []Chain{testChain(decider)}, testInput, cases, 2, 100*time.Millisecond)
	if peak.Load() != 2 {
		t.Fatalf("peak concurrency = %d, want 2", peak.Load())
	}
	for i := range cases {
		if !report.Correct(i, 0) || !report.Outcomes[i][0].Started || report.Outcomes[i][0].Cancelled {
			t.Errorf("trial %d = %+v", i, report.Outcomes[i][0])
		}
	}
}

func TestRunCancellationMarksStartedAndQueuedTrials(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	started := make(chan struct{})
	decider := functionDecider{call: func(ctx context.Context, _ string, _ []decide.Candidate, _ []decide.Level) (decide.Verdict, error) {
		close(started)
		<-ctx.Done()
		return decide.Verdict{}, ctx.Err()
	}}
	cases := Repeat([]Case{{Name: "a", Expect: []string{"m"}, Prompt: "hi"}}, 21)
	finished := make(chan Report, 1)
	go func() { finished <- Run(ctx, []Chain{testChain(decider)}, testInput, cases, 1, time.Second) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("trial did not start")
	}
	cancel()
	var report Report
	select {
	case report = <-finished:
	case <-time.After(time.Second):
		t.Fatal("canceled benchmark did not finish")
	}
	for i := range cases {
		outcome := report.Outcomes[i][0]
		if outcome.Started != (i == 0) || !outcome.Cancelled || outcome.Err != context.Canceled.Error() {
			t.Errorf("trial %d = %+v", i, outcome)
		}
		if i > 0 && (outcome.Latency != 0 || len(outcome.Attempts) != 0) {
			t.Errorf("unstarted trial %d recorded a sample: %+v", i, outcome)
		}
	}
	if len(report.Outcomes[0][0].Attempts) != 1 {
		t.Fatal("started cancellation lost its attempt")
	}
}

type pairDecider struct{ functionDecider }

func (pairDecider) Pair(ctx context.Context, _ string, _, _ decide.Candidate) (decide.Verdict, error) {
	<-ctx.Done()
	return decide.Verdict{}, ctx.Err()
}

func TestRunPreservesUsableDecisionWhenPairConsumesBudget(t *testing.T) {
	decider := pairDecider{functionDecider{call: func(context.Context, string, []decide.Candidate, []decide.Level) (decide.Verdict, error) {
		return decide.Verdict{Choice: "m", Probabilities: map[string]float64{"m": 0.8, "n": 0.2}}, nil
	}}}
	chain := decide.NewChain([]decide.Step{{Decider: decider, Timeout: time.Second}}, 0.1, decide.DoubtPair, nil)
	cases := []Case{{Name: "a", Expect: []string{"n"}, Prompt: "hi"}}
	report := Run(t.Context(), []Chain{{Name: "probe", Chain: chain}}, func(c Case) Input {
		return Input{State: c.Prompt, Candidates: []decide.Candidate{{ID: "m"}, {ID: "n"}}}
	}, cases, 1, 5*time.Millisecond)
	outcome := report.Outcomes[0][0]
	if !outcome.Started || outcome.Cancelled || outcome.Err != "" || !report.Correct(0, 0) || outcome.Model != "n" || len(outcome.Attempts) != 2 || outcome.Attempts[1].Outcome != decide.Failed {
		t.Fatalf("usable pair-timeout decision = %+v", outcome)
	}
}

func TestRunPreservesPoliciesCostsAndAppliedEffort(t *testing.T) {
	candidates := []decide.Candidate{{ID: "m", Examples: []string{"first"}}, {ID: "n"}}
	efforts := map[string]map[string]string{"m": {"low": "low"}, "n": {"low": "high"}}
	decider := functionDecider{call: func(context.Context, string, []decide.Candidate, []decide.Level) (decide.Verdict, error) {
		return decide.Verdict{Choice: "m", Probabilities: map[string]float64{"m": 0.6, "n": 0.4}, Levels: map[string]float64{"low": 1}}, nil
	}}
	chain := decide.NewChain([]decide.Step{{Decider: decider, Timeout: time.Second}}, 0.1, decide.DoubtEscalate, []decide.Level{{ID: "low"}})
	chain.Price(decide.Economy{Task: func(model, _ string, _ []decide.Candidate) (float64, bool) {
		if model == "m" {
			return 2, true
		}
		return 3, true
	}, Miss: 1})
	cases := []Case{{Name: "a", Expect: []string{"n"}, Reasoning: []string{"low"}, Prompt: "hi"}}
	report := Run(t.Context(), []Chain{{Name: "probe", Chain: chain}}, func(c Case) Input {
		return Input{State: c.Prompt, Candidates: candidates, Efforts: efforts}
	}, cases, 1, time.Second)
	candidates[0].Examples[0], candidates[0].ID, efforts["n"]["low"], cases[0].Expect[0] = "changed", "changed", "changed", "changed"
	outcome := report.Outcomes[0][0]
	policy := report.Policies[outcome.Policy]
	if !report.Correct(0, 0) || outcome.Effort != "high" || outcome.Costs.Task["m"] != 2 || outcome.Costs.Done["n"] != 3 || len(outcome.Attempts) != 1 {
		t.Fatalf("outcome = %+v", outcome)
	}
	if policy.Fingerprint() != outcome.Policy || policy.Candidates[0].ID != "m" || !slices.Equal(policy.Candidates[0].Examples, []string{"first"}) || report.Inputs[0].Efforts["n"]["low"] != "high" {
		t.Fatalf("policy/input did not remain a snapshot: %+v, %+v", policy, report.Inputs[0])
	}
}

func TestWriteExcludesUnstartedLatencyAndMarksMissingReasoning(t *testing.T) {
	report := Report{Cases: Repeat([]Case{{Name: "a", Expect: []string{"m"}, Reasoning: []string{"low"}, Prompt: "hi"}}, 21), Deciders: []string{"probe"}}
	for i := range report.Cases {
		outcome := Outcome{Cancelled: true, Err: context.Canceled.Error()}
		if i == 0 {
			outcome = Outcome{Started: true, Model: "m", Choice: "m", Latency: 30 * time.Millisecond}
		}
		report.Outcomes = append(report.Outcomes, []Outcome{outcome})
	}
	var out bytes.Buffer
	report.Write(&out)
	if !strings.Contains(out.String(), "@missing!") || !strings.Contains(out.String(), "a#1") || !strings.Contains(out.String(), "CANCELLED") {
		t.Fatalf("missing trial/level/cancellation indicators:\n%s", out.String())
	}
	lines := strings.Split(out.String(), "\n")
	for i, line := range lines {
		if strings.HasPrefix(line, "decider") && strings.Contains(line, "p50") {
			fields := strings.Fields(lines[i+1])
			if fields[len(fields)-2] != "30ms" || fields[len(fields)-1] != "30ms" {
				t.Fatalf("unstarted trials altered percentiles: %s", lines[i+1])
			}
			return
		}
	}
	t.Fatal("latency summary missing")
}

func TestPercentileUsesNearestRank(t *testing.T) {
	var samples []time.Duration
	for i := range 100 {
		samples = append(samples, time.Duration(i+1)*time.Millisecond)
	}
	for _, p := range []int{1, 50, 95, 100} {
		if got := percentile(samples, p); got != time.Duration(p)*time.Millisecond {
			t.Errorf("p%d = %s", p, got)
		}
	}
	if percentile(nil, 50) != 0 || percentile([]time.Duration{7 * time.Millisecond}, 95) != 7*time.Millisecond {
		t.Fatal("empty/single sample percentile is wrong")
	}
}
