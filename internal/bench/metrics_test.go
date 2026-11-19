package bench

import (
	"bytes"
	"encoding/json/v2"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
)

func metricReport(cases []Case, outcomes []Outcome) Report {
	r := Report{Cases: cases, Deciders: []string{"chain"}}
	for i, outcome := range outcomes {
		r.Outcomes = append(r.Outcomes, []Outcome{outcome})
		r.Inputs = append(r.Inputs, Input{State: cases[i].Prompt, Candidates: []decide.Candidate{
			{ID: "cheap"}, {ID: "medium"}, {ID: "strong"}, {ID: "hard"}, {ID: "ui", Specialist: true},
		}})
	}
	return r
}

func TestSummaryExposesRareGroupAndMissingDepth(t *testing.T) {
	var cases []Case
	var outcomes []Outcome
	for i := range 9 {
		cases = append(cases, Case{Name: fmt.Sprintf("quick/%d", i), Expect: []string{"cheap"}, Reasoning: []string{"low"}})
		outcomes = append(outcomes, Outcome{Started: true, Model: "cheap", Choice: "cheap", Level: "low"})
	}
	cases = append(cases, Case{Name: "hard/security", Group: "difficult", Expect: []string{"hard"}, Reasoning: []string{"high"}},
		Case{Name: "unlabelled", Expect: []string{"medium"}})
	outcomes = append(outcomes, Outcome{Started: true, Model: "cheap", Choice: "cheap"},
		Outcome{Started: true, Model: "medium", Choice: "medium"})
	m := metricReport(cases, outcomes).Summary(0)
	if m.Correct != 10 || m.Planned != 11 || m.Accuracy <= 0.9 || m.Labelled != 10 ||
		m.JointCorrect != 9 || m.JointAccuracy != 0.9 || m.MissingDepth != 1 || m.LevelCorrect != 9 {
		t.Fatalf("denominators or missing reasoning = %+v", m)
	}
	if !m.Completed || m.WeakestGroupAccuracy != 0 || m.WeakestGroupJointAccuracy != 0 ||
		math.Abs(m.GroupMacroAccuracy-2.0/3) > 1e-12 {
		t.Fatalf("masked rare group = %+v", m)
	}
	i := slices.IndexFunc(m.Groups, func(g GroupMetrics) bool { return g.Name == "difficult" })
	if i < 0 || m.Groups[i].Planned != 1 || m.Groups[i].Labelled != 1 || m.Groups[i].Correct != 0 {
		t.Fatalf("rare group support = %+v", m.Groups)
	}
}

func TestSummaryLatencyExcludesUnstartedAndIncludesBudgetFailure(t *testing.T) {
	cases := []Case{{Name: "success", Expect: []string{"cheap"}}, {Name: "budget", Expect: []string{"cheap"}}, {Name: "cancelled", Expect: []string{"cheap"}}}
	outcomes := []Outcome{
		{Started: true, Model: "cheap", Latency: 200 * time.Microsecond},
		{Started: true, Err: "decision budget exceeded", Latency: 2 * time.Millisecond},
		{Cancelled: true, Err: "context canceled"},
	}
	m := metricReport(cases, outcomes).Summary(0)
	if m.Started != 2 || m.NotStarted != 1 || m.Cancelled != 1 || m.Completed || m.Errors != 2 ||
		m.LatencySamples != 2 || m.P50MS != 0.2 || m.P95MS != 2 || m.Accuracy != 1.0/3 {
		t.Fatalf("cancelled and started metrics = %+v", m)
	}
	m = metricReport(cases[:2], outcomes[:2]).Summary(0)
	if !m.Completed {
		t.Fatal("a completed failed decision was mistaken for an abandoned trial")
	}
	latencies := make([]time.Duration, 20)
	for i := range latencies {
		latencies[i] = time.Duration(i+1) * time.Millisecond
	}
	if latencyPercentileMS(latencies, 50) != 10 || latencyPercentileMS(latencies, 95) != 19 {
		t.Fatal("percentiles do not use nearest rank")
	}
	if latencyPercentileMS(nil, 95) != 0 {
		t.Fatal("empty latency sample has a value")
	}
}

func TestSummaryRoutingDirectionsDoNotFillExpectationGaps(t *testing.T) {
	cases := []Case{
		{Name: "under", Expect: []string{"strong", "hard"}},
		{Name: "over", Expect: []string{"cheap", "medium"}},
		{Name: "gap", Expect: []string{"cheap", "hard"}},
		{Name: "specialist", Expect: []string{"ui"}},
		{Name: "unoffered", Expect: []string{"missing"}},
		{Name: "error", Expect: []string{"cheap"}},
	}
	outcomes := []Outcome{
		{Started: true, Model: "medium"}, {Started: true, Model: "strong"},
		{Started: true, Model: "medium"}, {Started: true, Model: "cheap"},
		{Started: true, Model: "cheap"}, {Started: true, Err: "unavailable"},
	}
	m := metricReport(cases, outcomes).Summary(0)
	if m.Correct != 0 || m.UnderRouted != 1 || m.OverRouted != 1 || m.OtherMisses != 3 || m.Errors != 1 {
		t.Fatalf("direction classification = %+v", m)
	}
	if len(m.Tiers) != 6 {
		t.Fatalf("acceptable sets were counted more than once: %+v", m.Tiers)
	}
}

func TestSummaryUsageIncludesFailedPairsAndDistinguishesKnownZero(t *testing.T) {
	cases := []Case{{Name: "pair", Expect: []string{"strong"}}, {Name: "fallback", Expect: []string{"cheap"}}, {Name: "cancelled", Expect: []string{"cheap"}}}
	outcomes := []Outcome{
		{Started: true, Model: "strong", Attempts: []decide.Attempt{
			{Outcome: decide.Doubted, UsageKnown: true, InputTokens: 10, OutputTokens: 2},
			{Outcome: decide.Failed, Pair: []string{"cheap", "strong"}, UsageKnown: true, InputTokens: 20, OutputTokens: 3},
		}},
		{Started: true, Model: "cheap", Attempts: []decide.Attempt{
			{Outcome: decide.Failed, InputTokens: 7}, {Outcome: decide.Skipped}, {Outcome: decide.Decided, UsageKnown: true},
		}},
		{Cancelled: true, Err: "context canceled"},
	}
	m := metricReport(cases, outcomes).Summary(0)
	if m.Calls != 4 || m.FailedCalls != 2 || m.SkippedAttempts != 1 || m.KnownUsageCalls != 3 || m.UnknownUsageCalls != 1 ||
		m.InputTokens != 37 || m.OutputTokens != 5 || m.KnownInputTokens != 30 || m.PartialInputTokens != 7 || m.PairCalls != 1 || m.FailedPairs != 1 ||
		m.PairKnownUsageCalls != 1 || m.PairInputTokens != 20 || m.PairOutputTokens != 3 {
		t.Fatalf("usage coverage = %+v", m)
	}
	if m.PairRate != 0.5 || m.CallsPerStarted != 2 || m.MeanReportedTokensPerKnownPair != 23 ||
		m.AmortizedReportedPairTokensPerStarted != 11.5 || math.Abs(m.MeanReportedTokensPerKnownCall-35.0/3) > 1e-12 {
		t.Fatalf("conditional and amortized cost = %+v", m)
	}
	var text bytes.Buffer
	metricReport(cases, outcomes).writeMetrics(&text)
	if !strings.Contains(text.String(), "Token totals include available partial counters") || !strings.Contains(text.String(), "amortized 11.5") {
		t.Fatalf("usage limits are missing from text: %s", text.String())
	}
}

func TestSummaryRepeatsMeasureStabilityOnUniqueTasks(t *testing.T) {
	var cases []Case
	var outcomes []Outcome
	for _, name := range []string{"stable", "changes", "fails"} {
		for trial := range 2 {
			cases = append(cases, Case{Name: name, Trial: trial + 1, Expect: []string{"cheap"}, Reasoning: []string{"low"}})
			outcome := Outcome{Started: true, Model: "cheap", Choice: "cheap", Level: "low"}
			if name == "changes" && trial == 1 {
				outcome.Model = "medium"
			}
			if name == "fails" && trial == 1 {
				outcome.Err = "unavailable"
			}
			outcomes = append(outcomes, outcome)
		}
	}
	cases = append(cases, Case{Name: "single", Expect: []string{"cheap"}})
	outcomes = append(outcomes, Outcome{Started: true, Model: "cheap"})
	m := metricReport(cases, outcomes).Summary(0)
	if m.Planned != 7 || m.UniqueCases != 4 || m.RepeatedCases != 3 || m.StableModelCases != 1 ||
		m.JointRepeatedCases != 3 || m.StableJointCases != 1 || m.RobustCorrectCases != 2 || m.UniqueCaseRobustAccuracy != 0.5 {
		t.Fatalf("repeat support/stability = %+v", m)
	}
}

func TestWriteJSONPreservesReplayInputsAndAppliedEffort(t *testing.T) {
	state := "Investigate the quoted \"request\"\nwith Unicode: ação"
	candidates := []decide.Candidate{{ID: "cheap", Description: "routine"}, {ID: "strong", Description: "difficult"}}
	chain := decide.NewChain([]decide.Step{{Decider: decide.NewSystemOne("jev", "http://unused", "jev-test", "", nil), Timeout: time.Second}}, 0.1, decide.DoubtPair,
		[]decide.Level{{ID: "low", Description: "mechanical"}, {ID: "high", Description: "investigation"}})
	policy := chain.Policy(candidates)
	body, ok := policy.Request("jev", state)
	if !ok {
		t.Fatal("policy request cannot be constructed")
	}
	digest := decide.Digest(body)
	r := Report{
		Cases:    []Case{{Name: "depth/clamped", Trial: 2, Expect: []string{"strong"}, Reasoning: []string{"low"}, Prompt: state}},
		Deciders: []string{"chain"},
		Inputs:   []Input{{State: state, Candidates: candidates, Efforts: map[string]map[string]string{"strong": {"low": "high"}}}},
		Outcomes: [][]Outcome{{{Started: true, Model: "strong", Choice: "strong", Level: "low", Effort: "high", Policy: policy.Fingerprint(),
			Latency: 250 * time.Microsecond, Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Digest: digest, UsageKnown: true}},
			Costs: decide.Costs{Task: map[string]float64{"strong": 3}, Done: map[string]float64{"strong": 3}},
		}}},
		Policies: map[string]decide.Policy{policy.Fingerprint(): policy},
		Settings: map[string]any{"parallel": 2, "cost_basis": "typical task prior"},
	}
	var out bytes.Buffer
	if err := r.WriteJSON(&out); err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		SchemaVersion int            `json:"schema_version"`
		Settings      map[string]any `json:"settings"`
		Cases         []struct {
			Trial int `json:"trial"`
		} `json:"cases"`
		Inputs []struct {
			State   string                       `json:"state"`
			Efforts map[string]map[string]string `json:"efforts"`
		} `json:"inputs"`
		Outcomes [][]struct {
			Level     string           `json:"task_depth"`
			Effort    string           `json:"applied_effort"`
			LatencyMS float64          `json:"latency_ms"`
			Attempts  []decide.Attempt `json:"attempts"`
			Costs     decide.Costs     `json:"routing_costs"`
		} `json:"outcomes"`
		Policies map[string]decide.Policy `json:"policies"`
		Metrics  []DeciderMetrics         `json:"metrics"`
	}
	if err := json.Unmarshal(out.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.SchemaVersion != 2 || decoded.Cases[0].Trial != 2 || decoded.Settings["cost_basis"] != "typical task prior" {
		t.Fatalf("metadata = %s", out.String())
	}
	o := decoded.Outcomes[0][0]
	if o.Level != "low" || o.Effort != "high" || o.LatencyMS != 0.25 || o.Costs.Task["strong"] != 3 ||
		decoded.Inputs[0].Efforts["strong"]["low"] != "high" || decoded.Metrics[0].JointAccuracy != 1 {
		t.Fatalf("depth, effort or costs were lost = %s", out.String())
	}
	rebuilt, ok := decoded.Policies[policy.Fingerprint()].Request("jev", decoded.Inputs[0].State)
	if !ok || decide.Digest(rebuilt) != o.Attempts[0].Digest {
		t.Fatal("JSON policy and prepared input cannot reproduce the recorded request")
	}
	if err := r.WriteJSON(failingWriter{}); err == nil {
		t.Fatal("JSON writer failure was discarded")
	}
}

type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestSummaryScoresFrozenRawLabelsSeparatelyFromFinalRoutes(t *testing.T) {
	r := metricReport([]Case{{Name: "teacher/task", Expect: []string{"cheap"}, Reasoning: []string{"low"}}},
		[]Outcome{{Started: true, Choice: "cheap", Level: "low", Model: "strong"}})
	m := r.Summary(0)
	if m.ChoiceAccuracy != 1 || m.ChoiceJointCorrect != 1 || m.ChoiceJointAccuracy != 1 || m.Accuracy != 0 || m.JointAccuracy != 0 ||
		m.Groups[0].ChoiceAccuracy != 1 || m.Groups[0].ChoiceJointAccuracy != 1 || m.Groups[0].Accuracy != 0 || m.Groups[0].JointAccuracy != 0 {
		t.Fatalf("teacher imitation was mixed with economy-final routes: %+v", m)
	}
}
