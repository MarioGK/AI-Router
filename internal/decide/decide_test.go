package decide

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

var three = []Candidate{
	{ID: "deepseek-flash", Description: "cheap"},
	{ID: "glm-5.3", Description: "everyday"},
	{ID: "gpt-6.1-sol", Description: "hard", NotFor: "trivia", Examples: []string{"fix the race"}},
}

// withSpecialist puts a frontend specialist between and after ladder steps.
var withSpecialist = []Candidate{
	{ID: "deepseek-flash", Description: "small"},
	{ID: "claude-opus-5-5", Description: "frontend", Specialist: true},
	{ID: "gpt-6.1-sol", Description: "medium"},
	{ID: "glm-5.3", Description: "complicated"},
}

func TestEscalate(t *testing.T) {
	cases := []struct {
		name          string
		verdict       Verdict
		want          string
		wantEscalated bool
	}{
		{"confident cheap", Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.91, "glm-5.3": 0.09}}, "deepseek-flash", false},
		{"ten percent above", Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.90, "glm-5.3": 0.10}}, "glm-5.3", true},
		{"rounding counts", Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.90, "glm-5.3": 0.07, "gpt-6.1-sol": 0.03}}, "glm-5.3", true},
		{"doubt below never upgrades", Verdict{Choice: "glm-5.3", Probabilities: map[string]float64{"deepseek-flash": 0.40, "glm-5.3": 0.55, "gpt-6.1-sol": 0.05}}, "glm-5.3", false},
		{"one tier only", Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.50, "gpt-6.1-sol": 0.50}}, "glm-5.3", true},
		{"top is capped", Verdict{Choice: "gpt-6.1-sol", Probabilities: map[string]float64{"glm-5.3": 0.50, "gpt-6.1-sol": 0.50}}, "gpt-6.1-sol", false},
	}
	for _, c := range cases {
		index, escalated := Escalate(three, c.verdict, 0.10)
		if three[index].ID != c.want || escalated != c.wantEscalated {
			t.Errorf("%s: Escalate = %s/%v, want %s/%v", c.name, three[index].ID, escalated, c.want, c.wantEscalated)
		}
	}
}

func TestEscalateSkipsSpecialists(t *testing.T) {
	cases := []struct {
		name          string
		verdict       Verdict
		want          string
		wantEscalated bool
	}{
		{"doubt towards the specialist stays", Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.6, "claude-opus-5-5": 0.4}}, "deepseek-flash", false},
		{"next ladder step skips the specialist", Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.85, "gpt-6.1-sol": 0.10, "claude-opus-5-5": 0.05}}, "gpt-6.1-sol", true},
		{"specialist choice stands", Verdict{Choice: "claude-opus-5-5", Probabilities: map[string]float64{"claude-opus-5-5": 0.5, "glm-5.3": 0.5}}, "claude-opus-5-5", false},
		{"ladder top", Verdict{Choice: "glm-5.3", Probabilities: map[string]float64{"glm-5.3": 0.7, "claude-opus-5-5": 0.3}}, "glm-5.3", false},
	}
	for _, c := range cases {
		index, escalated := Escalate(withSpecialist, c.verdict, 0.10)
		if withSpecialist[index].ID != c.want || escalated != c.wantEscalated {
			t.Errorf("%s: Escalate = %s/%v, want %s/%v", c.name, withSpecialist[index].ID, escalated, c.want, c.wantEscalated)
		}
	}
}

func TestMassAbove(t *testing.T) {
	verdict := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.8, "claude-opus-5-5": 0.05, "gpt-6.1-sol": 0.1, "glm-5.3": 0.05}}
	if above := MassAbove(withSpecialist, verdict); above < 0.149 || above > 0.151 {
		t.Fatalf("MassAbove = %v, want 0.15 (the specialist does not count)", above)
	}
	if above := MassAbove(withSpecialist, Verdict{Choice: "claude-opus-5-5", Probabilities: verdict.Probabilities}); above != 0 {
		t.Fatalf("MassAbove of a specialist = %v", above)
	}
}

func TestNormalize(t *testing.T) {
	v, err := normalize(Verdict{Choice: "claude-opus-5", Probabilities: map[string]float64{"glm-5.3": 2, "gpt-6.1-sol": 6, "claude-opus-5": 2}}, three)
	if err != nil || v.Choice != "gpt-6.1-sol" || v.Probabilities["gpt-6.1-sol"] != 0.75 || len(v.Probabilities) != 3 {
		t.Fatalf("normalize = %+v, %v", v, err)
	}
	v, err = normalize(Verdict{Choice: "glm-5.3"}, three)
	if err != nil || v.Probabilities["glm-5.3"] != 1 {
		t.Fatalf("normalize choice only = %+v, %v", v, err)
	}
	if _, err := normalize(Verdict{Choice: "nope"}, three); err == nil {
		t.Fatal("normalize accepted a verdict without any candidate")
	}
}

type fakeDecider struct {
	name    string
	verdict Verdict
	err     error
	wait    time.Duration
	calls   int
}

func (f *fakeDecider) Name() string { return f.name }

func (f *fakeDecider) Info() DeciderInfo { return DeciderInfo{Name: f.name, Kind: "fake"} }

func (f *fakeDecider) Decide(ctx context.Context, _ string, _ []Candidate, _ []Level) (Verdict, error) {
	f.calls++
	if f.wait > 0 {
		select {
		case <-time.After(f.wait):
		case <-ctx.Done():
			return Verdict{}, ctx.Err()
		}
	}
	return f.verdict, f.err
}

func TestASlowDeciderIsCutAtItsOwnTimeout(t *testing.T) {
	// The first decider has little time and stalls; the second has plenty
	// and answers after a pause that the first one's budget cannot cover.
	stalled := &fakeDecider{name: "stalled", wait: 5 * time.Second, verdict: Verdict{Choice: "gpt-6.1-sol"}}
	patient := &fakeDecider{name: "patient", wait: 60 * time.Millisecond,
		verdict: Verdict{Probabilities: map[string]float64{"deepseek-flash": 0.2, "glm-5.3": 0.8}}}
	chain := NewChain([]Step{
		{Decider: stalled, Timeout: 30 * time.Millisecond},
		{Decider: patient, Timeout: 5 * time.Second},
	}, 0.10, DoubtEscalate, nil)
	start := time.Now()
	decision, ok := chain.Decide(context.Background(), "state", three)
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("the chain took %v: the stalled decider ran past its own timeout", elapsed)
	}
	if !ok || decision.Decider != "patient" || decision.Model != "glm-5.3" {
		t.Fatalf("Decide = %+v, %v", decision, ok)
	}
	if len(decision.Attempts) != 2 || decision.Attempts[0].Outcome != Failed ||
		!strings.Contains(decision.Attempts[0].Error, "deadline exceeded") || decision.Attempts[1].Outcome != Decided {
		t.Fatalf("attempts = %+v", decision.Attempts)
	}
	if got := decision.Attempts[0].LatencyMS; got >= 1000 {
		t.Errorf("the stalled attempt took %d ms", got)
	}
}

func TestChainFallsThroughFailingDeciders(t *testing.T) {
	broken := &fakeDecider{name: "jev", err: errors.New("status 529: overloaded")}
	slow := &fakeDecider{name: "slow", wait: time.Second, verdict: Verdict{Choice: "gpt-6.1-sol"}}
	invalid := &fakeDecider{name: "invalid", verdict: Verdict{Choice: "claude-opus-5"}}
	judge := &fakeDecider{name: "judge", verdict: Verdict{Probabilities: map[string]float64{"deepseek-flash": 0.2, "glm-5.3": 0.8}}}
	chain := NewChain([]Step{
		{Decider: broken, Timeout: time.Second},
		{Decider: slow, Timeout: 10 * time.Millisecond},
		{Decider: invalid, Timeout: time.Second},
		{Decider: judge, Timeout: time.Second},
	}, 0.10, DoubtEscalate, nil)
	decision, ok := chain.Decide(context.Background(), "state", three)
	if !ok || decision.Model != "glm-5.3" || decision.Decider != "judge" || decision.Escalated {
		t.Fatalf("Decide = %+v, %v", decision, ok)
	}
	failures := decision.Failures()
	if len(failures) != 3 || !strings.HasPrefix(failures[0], "jev: status 529") || !strings.HasPrefix(failures[1], "slow: ") {
		t.Fatalf("failures = %q", failures)
	}
	outcomes := make([]string, len(decision.Attempts))
	for i, attempt := range decision.Attempts {
		outcomes[i] = attempt.Decider + " " + attempt.Outcome
	}
	if !slices.Equal(outcomes, []string{"jev failed", "slow failed", "invalid failed", "judge decided"}) {
		t.Fatalf("attempts = %q", outcomes)
	}
	if last := decision.Attempts[3]; last.Choice != "glm-5.3" || last.Probabilities["glm-5.3"] != 0.8 || last.Above != 0 {
		t.Fatalf("deciding attempt = %+v", last)
	}
}

func TestChainReportsTotalFailure(t *testing.T) {
	chain := NewChain([]Step{{Decider: &fakeDecider{name: "jev", err: errors.New("down")}, Timeout: time.Second}}, 0.10, DoubtEscalate, nil)
	if decision, ok := chain.Decide(context.Background(), "state", three); ok || len(decision.Failures()) != 1 {
		t.Fatalf("Decide = %+v, %v", decision, ok)
	}
}

// fakePairer also answers pair questions.
type fakePairer struct {
	fakeDecider
	pair    Verdict
	pairErr error
	asked   []string
}

func (f *fakePairer) Pair(_ context.Context, _ string, cheap, strong Candidate) (Verdict, error) {
	f.asked = append(f.asked, cheap.ID+" vs "+strong.ID)
	return f.pair, f.pairErr
}

func TestPairQuestionsSettleDoubt(t *testing.T) {
	unsure := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.8, "glm-5.3": 0.2}}
	sure := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 1}}
	stronger := Verdict{Choice: "glm-5.3", Probabilities: map[string]float64{"deepseek-flash": 0.3, "glm-5.3": 0.7}}
	cheaper := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.66, "glm-5.3": 0.34}}
	cases := []struct {
		name      string
		jev       Decider
		doubt     string
		want      string
		escalated bool
		outcomes  []string
	}{
		{"escalate keeps the rule", &fakePairer{fakeDecider: fakeDecider{name: "jev", verdict: unsure}, pair: cheaper},
			DoubtEscalate, "glm-5.3", true, []string{"decided"}},
		{"a sure verdict needs no pair", &fakePairer{fakeDecider: fakeDecider{name: "jev", verdict: sure}, pair: stronger},
			DoubtPair, "deepseek-flash", false, []string{"decided"}},
		{"the pair moves up", &fakePairer{fakeDecider: fakeDecider{name: "jev", verdict: unsure}, pair: stronger},
			DoubtPair, "glm-5.3", true, []string{"doubted", "decided"}},
		{"the pair keeps the choice", &fakePairer{fakeDecider: fakeDecider{name: "jev", verdict: unsure}, pair: cheaper},
			DoubtPair, "deepseek-flash", false, []string{"doubted", "decided"}},
		{"a failed pair escalates", &fakePairer{fakeDecider: fakeDecider{name: "jev", verdict: unsure}, pairErr: errors.New("timeout")},
			DoubtPair, "glm-5.3", true, []string{"doubted", "failed"}},
		{"a decider without pairs escalates", &fakeDecider{name: "jev", verdict: unsure},
			DoubtPair, "glm-5.3", true, []string{"decided"}},
	}
	for _, c := range cases {
		fallback := &fakeDecider{name: "deepseek", verdict: sure}
		chain := NewChain([]Step{{Decider: c.jev, Timeout: time.Second}, {Decider: fallback, Timeout: time.Second}}, 0.10, c.doubt, nil)
		decision, ok := chain.Decide(context.Background(), "state", three)
		var outcomes []string
		for _, attempt := range decision.Attempts {
			outcomes = append(outcomes, attempt.Outcome)
		}
		if !ok || decision.Model != c.want || decision.Decider != "jev" || decision.Choice != "deepseek-flash" ||
			decision.Escalated != c.escalated || !slices.Equal(outcomes, c.outcomes) || fallback.calls != 0 {
			t.Errorf("%s: Decide = %+v, %v (fallback called %d times)", c.name, decision, ok, fallback.calls)
		}
		if len(decision.Attempts) == 2 {
			pair := decision.Attempts[1]
			if !slices.Equal(pair.Pair, []string{"deepseek-flash", "glm-5.3"}) || pair.Decider != "jev" {
				t.Errorf("%s: pair attempt = %+v", c.name, pair)
			}
			if pair.Outcome == Decided && pair.Above != c.jev.(*fakePairer).pair.Probabilities["glm-5.3"] {
				t.Errorf("%s: pair above = %v", c.name, pair.Above)
			}
		}
	}
}

func TestTheLastResortDecidesWhenJevFails(t *testing.T) {
	jev := &fakePairer{fakeDecider: fakeDecider{name: "jev", err: errors.New("status 503")}}
	unsure := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.8, "glm-5.3": 0.2}}
	keep := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.9, "glm-5.3": 0.1}}
	fallback := &fakePairer{fakeDecider: fakeDecider{name: "deepseek", verdict: unsure}, pair: keep}
	chain := NewChain([]Step{{Decider: jev, Timeout: time.Second}, {Decider: fallback, Timeout: time.Second}}, 0.10, DoubtPair, nil)
	decision, ok := chain.Decide(context.Background(), "state", three)
	// The last resort settles its own doubt with a pair question too.
	if !ok || decision.Decider != "deepseek" || decision.Model != "deepseek-flash" || decision.Escalated ||
		len(jev.asked) != 0 || !slices.Equal(fallback.asked, []string{"deepseek-flash vs glm-5.3"}) {
		t.Fatalf("Decide = %+v, %v", decision, ok)
	}
}

// choiceRequest is a systemone request as a test reads it: choice
// questions only.
type choiceRequest struct {
	Model     string `json:"model"`
	State     string `json:"state"`
	Questions map[string]struct {
		Type         string               `json:"type"`
		Instructions Instructions         `json:"instructions"`
		Criteria     map[string]criterion `json:"criteria"`
	} `json:"questions"`
}

func TestSystemOneSpeaksTheChoiceAPI(t *testing.T) {
	var received choiceRequest
	server := newTestEndpoint(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/systemone" || r.Header.Get("Authorization") != "Bearer jev-key" {
			t.Errorf("request %s with %q", r.URL.Path, r.Header.Get("Authorization"))
		}
		_ = json.NewDecoder(r.Body).Decode(&received)
		_, _ = w.Write([]byte(`{"model":"jev-1.13.0","answers":{"model":{"type":"choice","choice":"glm-5.3",
			"probabilities":{"deepseek-flash":0.1,"glm-5.3":0.85,"gpt-6.1-sol":0.05},"confidence":0.78}},"usage":{"input_tokens":318}}`))
	}))
	decider := NewSystemOne("jev", server.URL+"/v1/systemone", "jev-latest", "jev-key", server.Client())
	verdict, err := decider.Decide(context.Background(), "images=no\n\nadd a flag", three, nil)
	if err != nil || verdict.Choice != "glm-5.3" || verdict.Probabilities["glm-5.3"] != 0.85 ||
		verdict.Model != "jev-1.13.0" || verdict.Confidence != 0.78 || verdict.InputTokens != 318 {
		t.Fatalf("Decide = %+v, %v", verdict, err)
	}
	q := received.Questions["model"]
	if received.Model != "jev-latest" || q.Type != "choice" || len(q.Criteria) != 3 ||
		q.Criteria["deepseek-flash"].What != "cheap" || q.Criteria["gpt-6.1-sol"].NotFor != "trivia" ||
		!slices.Equal(q.Criteria["gpt-6.1-sol"].Examples, []string{"fix the race"}) || !strings.Contains(q.Instructions.Focus, "Frontend") {
		t.Fatalf("request = %+v", received)
	}
}

func TestSystemOneAnswersPairQuestions(t *testing.T) {
	var received choiceRequest
	server := newTestEndpoint(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&received)
		_, _ = w.Write([]byte(`{"model":"jev-1.13.0","answers":{"pair":{"choice":"gpt-6.1-sol","probabilities":{"glm-5.3":0.43,"gpt-6.1-sol":0.57}}}}`))
	}))
	decider := NewSystemOne("jev", server.URL, "jev-latest", "k", server.Client())
	verdict, err := decider.Pair(context.Background(), "state", three[1], three[2])
	if err != nil || verdict.Choice != "gpt-6.1-sol" || verdict.Probabilities["gpt-6.1-sol"] != 0.57 {
		t.Fatalf("Pair = %+v, %v", verdict, err)
	}
	q, ok := received.Questions["pair"]
	if !ok || len(q.Criteria) != 2 || q.Criteria["glm-5.3"].What != "everyday" ||
		!strings.Contains(q.Instructions.Question, "glm-5.3") ||
		!strings.Contains(q.Instructions.Question, "gpt-6.1-sol") ||
		q.Instructions.Probabilities != instructions.Probabilities {
		t.Fatalf("pair request = %+v", received)
	}
}

func TestSystemOneErrorCarriesStatus(t *testing.T) {
	client := &http.Client{Transport: judgeMetadataTransport(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusTooManyRequests, Body: io.NopCloser(strings.NewReader(`{"error":{"code":"RATE_LIMITED"}}`)), Header: http.Header{}}, nil
	})}
	_, err := NewSystemOne("jev", "https://decider.invalid/systemone", "jev-latest", "k", client).Decide(t.Context(), "s", three, nil)
	if err == nil || err.Error() != "status 429" {
		t.Fatalf("err = %v", err)
	}
}

func TestJudgeParsesReplies(t *testing.T) {
	cases := map[string]Verdict{
		"```json\n{\"probabilities\": {\"glm-5.3\": 0.7, \"gpt-6.1-sol\": 0.3}}\n```":          {Probabilities: map[string]float64{"glm-5.3": 0.7, "gpt-6.1-sol": 0.3}},
		`{"choice":"gpt-6.1-sol","probabilities":{"gpt-6.1-sol":1}} trailing words`:            {Choice: "gpt-6.1-sol", Probabilities: map[string]float64{"gpt-6.1-sol": 1}},
		`Sure: {"deepseek-flash": 0.6, "glm-5.3": 0.4, "note": "not a number"}`:                {Probabilities: map[string]float64{"deepseek-flash": 0.6, "glm-5.3": 0.4}},
		`{"probabilities":{"glm-5.3":1},"reason":"  Routine multi-file\n change. "}`:           {Probabilities: map[string]float64{"glm-5.3": 1}, Reason: "Routine multi-file change."},
		"<think>maybe {\"glm-5.3\": 1}?</think>{\"probabilities\":{\"gpt-6.1-sol\":1}}":        {Probabilities: map[string]float64{"gpt-6.1-sol": 1}},
		`Models are {listed}; answer: {"probabilities":{"glm-5.3":"0.8","gpt-6.1-sol":"20%"}}`: {Probabilities: map[string]float64{"glm-5.3": 0.8, "gpt-6.1-sol": 0.2}},
		`{"choice":"glm-5.3","reason":"everyday"}`:                                             {Choice: "glm-5.3", Reason: "everyday"},
	}
	for reply, want := range cases {
		got, err := parseJudgement(reply, three)
		if err != nil || got.Choice != want.Choice || got.Reason != want.Reason || len(got.Probabilities) != len(want.Probabilities) {
			t.Errorf("parseJudgement(%q) = %+v, %v", reply, got, err)
			continue
		}
		for id, p := range want.Probabilities {
			if got.Probabilities[id] != p {
				t.Errorf("parseJudgement(%q)[%s] = %v, want %v", reply, id, got.Probabilities[id], p)
			}
		}
	}
	if _, err := parseJudgement("I cannot decide", three); err == nil {
		t.Error("parseJudgement accepted a reply without JSON")
	}
	if _, err := parseJudgement(`{"note": "no judgement here"}`, three); err == nil {
		t.Error("parseJudgement accepted an object without a judgement")
	}
}

func TestJudgeCallsChatCompletions(t *testing.T) {
	server := newTestEndpoint(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model    string        `json:"model"`
			Messages []chatMessage `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&request)
		if request.Model != "gpt-6-luna" || len(request.Messages) != 2 ||
			request.Messages[0].Role != "system" || !strings.Contains(request.Messages[0].Content, "- glm-5.3: everyday\n") ||
			request.Messages[1].Role != "user" || request.Messages[1].Content != "s" {
			t.Errorf("request = %+v", request)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-6-luna-2026-09","choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"{\"probabilities\":{\"gpt-6.1-sol\":0.9,\"glm-5.3\":0.1},\"reason\":\"Cross-cutting bug.\"}"}]}}],"usage":{"prompt_tokens":420,"completion_tokens":31}}`))
	}))
	verdict, err := NewJudge("judge", server.URL, "gpt-6-luna", "k", server.Client()).Decide(context.Background(), "s", three, nil)
	if err != nil || verdict.Probabilities["gpt-6.1-sol"] != 0.9 || verdict.Reason != "Cross-cutting bug." ||
		verdict.Model != "gpt-6-luna-2026-09" || verdict.InputTokens != 420 || verdict.OutputTokens != 31 {
		t.Fatalf("Decide = %+v, %v", verdict, err)
	}
}

func TestPolicyFingerprintsWhatShapesDecisions(t *testing.T) {
	jev := Step{Decider: &fakeDecider{name: "jev"}, Timeout: time.Second}
	judge := Step{Decider: &fakeDecider{name: "judge"}, Timeout: time.Second}
	base := NewChain([]Step{jev, judge}, 0.10, DoubtPair, nil).Policy(three).Fingerprint()
	if len(base) != 12 || NewChain([]Step{jev, judge}, 0.10, DoubtPair, nil).Policy(three).Fingerprint() != base {
		t.Fatalf("policy = %q", base)
	}
	reworded := slices.Clone(three)
	reworded[1].Description = "everyday coding"
	example := slices.Clone(three)
	example[0].Examples = []string{"fix a typo"}
	for name, other := range map[string]string{
		"threshold":   NewChain([]Step{jev, judge}, 0.25, DoubtPair, nil).Policy(three).Fingerprint(),
		"doubt":       NewChain([]Step{jev, judge}, 0.10, DoubtEscalate, nil).Policy(three).Fingerprint(),
		"deciders":    NewChain([]Step{jev}, 0.10, DoubtPair, nil).Policy(three).Fingerprint(),
		"candidates":  NewChain([]Step{jev, judge}, 0.10, DoubtPair, nil).Policy(three[:2]).Fingerprint(),
		"description": NewChain([]Step{jev, judge}, 0.10, DoubtPair, nil).Policy(reworded).Fingerprint(),
		"examples":    NewChain([]Step{jev, judge}, 0.10, DoubtPair, nil).Policy(example).Fingerprint(),
	} {
		if other == base {
			t.Errorf("a different %s kept the policy %s", name, base)
		}
	}
}

func TestStateStripsPrefixWrappersAndKeepsRepresentativeExcerpts(t *testing.T) {
	builder := NewStateBuilder([]string{"system-reminder", "environment_context"}, 300)
	state := builder.Build("<system-reminder>\nsecret context\n</system-reminder><environment_context cwd=\"/x\">env</environment_context>Please refactor the parser", true)
	if state != (State{Text: "images=yes\n\nPlease refactor the parser", Prompt: "Please refactor the parser", Chars: 26}) {
		t.Fatalf("state = %+v", state)
	}
	const requirement = "This is the team's top priority today."
	long := strings.Repeat("a", 400) + requirement + strings.Repeat("z", 400)
	state = builder.Build(long, false)
	body := strings.TrimPrefix(state.Text, "images=no\n\n")
	if !strings.HasPrefix(body, strings.Repeat("a", 90)) || !strings.HasSuffix(body, strings.Repeat("z", 90)) || !strings.Contains(body, requirement) || strings.Count(body, omission) != 2 {
		t.Fatalf("cut state = %q", body)
	}
	if state.Chars != 800+len(requirement) || !state.Truncated || state.Prompt != long {
		t.Fatalf("cut state = %d chars, truncated %v", state.Chars, state.Truncated)
	}
	if builder.Build("<system-reminder>only</system-reminder>", false) != (State{}) {
		t.Fatal("state of wrapper-only text is not empty")
	}
}

func TestPolicyRebuildsTheExactDeciderRequests(t *testing.T) {
	var sent [][]byte
	var mu sync.Mutex
	jevDown := false
	server := newTestEndpoint(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		sent = append(sent, body)
		mu.Unlock()
		switch {
		case r.URL.Path == "/chat":
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"probabilities\":{\"glm-5.3\":1}}"}}]}`))
		case jevDown:
			w.WriteHeader(http.StatusServiceUnavailable)
		case bytes.Contains(body, []byte(`"pair":`)):
			_, _ = w.Write([]byte(`{"answers":{"pair":{"choice":"glm-5.3","probabilities":{"glm-5.3":0.6,"gpt-6.1-sol":0.4}}}}`))
		default:
			_, _ = w.Write([]byte(`{"answers":{"model":{"choice":"glm-5.3","probabilities":{"glm-5.3":0.5,"gpt-6.1-sol":0.5}}}}`))
		}
	}))
	chain := NewChain([]Step{
		{Decider: NewSystemOne("jev", server.URL+"/systemone", "jev-latest", "k", server.Client()), Timeout: time.Second},
		{Decider: NewJudge("deepseek", server.URL+"/chat", "deepseek-flash", "k", server.Client()), Timeout: time.Second},
	}, 0.10, DoubtPair, nil)
	policy := chain.Policy(three)
	const state = "images=no\n\nadd a --dry-run flag"

	decision, ok := chain.Decide(context.Background(), state, three)
	if !ok || decision.Model != "glm-5.3" || len(decision.Attempts) != 2 || decision.Attempts[1].Response == "" || len(sent) != 2 {
		t.Fatalf("decision = %+v", decision)
	}
	choice, _ := policy.Request("jev", state)
	pair, ok := policy.PairRequest("jev", state, "glm-5.3", "gpt-6.1-sol")
	if !ok || !bytes.Equal(choice, sent[0]) || !bytes.Equal(pair, sent[1]) {
		t.Errorf("jev requests rebuilt as\n%s\n%s\nsent\n%s\n%s", choice, pair, sent[0], sent[1])
	}

	jevDown, sent = true, nil
	decision, ok = chain.Decide(context.Background(), state, three)
	if !ok || decision.Decider != "deepseek" || decision.Attempts[1].Response != `{"probabilities":{"glm-5.3":1}}` {
		t.Fatalf("last resort decision = %+v", decision)
	}
	if judge, _ := policy.Request("deepseek", state); !bytes.Equal(judge, sent[1]) {
		t.Errorf("judge request rebuilt as\n%s\nsent\n%s", judge, sent[1])
	}
	if _, ok := policy.Request("ollama", state); ok {
		t.Error("a decider outside the policy has a request")
	}
	if _, ok := policy.PairRequest("ollama", state, "glm-5.3", "gpt-6.1-sol"); ok {
		t.Error("a decider outside the policy has a pair request")
	}
}

func TestJudgeAnswersPairQuestions(t *testing.T) {
	var request struct {
		Messages []chatMessage `json:"messages"`
	}
	server := newTestEndpoint(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&request)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"probabilities\":{\"glm-5.3\":0.2,\"gpt-6.1-sol\":0.8}}"}}]}`))
	}))
	verdict, err := NewJudge("deepseek", server.URL, "deepseek-flash", "k", server.Client()).Pair(context.Background(), "s", three[1], three[2])
	if err != nil || verdict.Probabilities["gpt-6.1-sol"] != 0.8 || len(request.Messages) != 2 ||
		!strings.Contains(request.Messages[0].Content, "Which of glm-5.3 and gpt-6.1-sol fits this request") ||
		strings.Contains(request.Messages[0].Content, "- deepseek-flash:") || !strings.Contains(request.Messages[0].Content, "- gpt-6.1-sol: hard Not for: trivia") || request.Messages[1].Content != "s" {
		t.Fatalf("Pair = %+v, %v; request %+v", verdict, err, request)
	}
}

func TestAFailingDeciderRestsAfterFailingInARow(t *testing.T) {
	broken := &fakeDecider{name: "jev", err: errors.New("status 503")}
	judge := &fakeDecider{name: "judge", verdict: Verdict{Choice: "glm-5.3"}}
	chain := NewChain([]Step{{Decider: broken, Timeout: time.Second}, {Decider: judge, Timeout: time.Second}},
		0.10, DoubtEscalate, nil).Rest(2, time.Minute)
	now := time.Date(2026, 9, 29, 12, 0, 0, 0, time.UTC)
	chain.now = func() time.Time { return now }
	for range 2 {
		if decision, ok := chain.Decide(context.Background(), "state", three); !ok || decision.Attempts[0].Outcome != Failed {
			t.Fatalf("decision = %+v", decision)
		}
	}
	decision, ok := chain.Decide(context.Background(), "state", three)
	if !ok || broken.calls != 2 || decision.Attempts[0].Outcome != Skipped || decision.Decider != "judge" ||
		!strings.Contains(decision.Failures()[0], "resting until 12:01:00 UTC after 2 failures") {
		t.Fatalf("resting decision = %+v after %d calls", decision, broken.calls)
	}
	if health := chain.Health(); health[0].Failures != 2 || !health[0].Resting.Equal(now.Add(time.Minute)) || health[1].Failures != 0 {
		t.Fatalf("health = %+v", health)
	}
	// After the rest one more failure rests it again; a success ends the run.
	now = now.Add(time.Minute)
	chain.Decide(context.Background(), "state", three)
	if broken.calls != 3 || chain.Health()[0].Resting.IsZero() {
		t.Fatalf("after the rest: %d calls, health %+v", broken.calls, chain.Health())
	}
	now = now.Add(time.Minute)
	broken.err, broken.verdict = nil, Verdict{Choice: "deepseek-flash"}
	if decision, _ := chain.Decide(context.Background(), "state", three); decision.Decider != "jev" || chain.Health()[0].Failures != 0 {
		t.Fatalf("recovered decision = %+v, health %+v", decision, chain.Health())
	}
}

func TestTheChainDeadlineIsNoFaultOfTheDecider(t *testing.T) {
	slow := &fakeDecider{name: "jev", wait: time.Second, verdict: Verdict{Choice: "glm-5.3"}}
	chain := NewChain([]Step{{Decider: slow, Timeout: time.Second}}, 0.10, DoubtEscalate, nil).Rest(1, time.Minute)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, ok := chain.Decide(ctx, "state", three); ok {
		t.Fatal("a decider past the chain's deadline decided")
	}
	if health := chain.Health(); health[0].Failures != 0 || !health[0].Resting.IsZero() {
		t.Fatalf("health = %+v", health)
	}
}

func TestAttemptsCarryTheDigestOfTheRequestSent(t *testing.T) {
	var sent []byte
	server := newTestEndpoint(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sent, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusBadGateway)
	}))
	chain := NewChain([]Step{{Decider: NewSystemOne("jev", server.URL, "jev-latest", "k", server.Client()), Timeout: time.Second}},
		0.10, DoubtEscalate, nil)
	decision, _ := chain.Decide(context.Background(), "state", three)
	rebuilt, _ := chain.Policy(three).Request("jev", "state")
	if digest := decision.Attempts[0].Digest; len(digest) != 16 || digest != Digest(sent) || digest != Digest(rebuilt) {
		t.Fatalf("digest %q of a failed attempt, sent %q, rebuilt %q", digest, Digest(sent), Digest(rebuilt))
	}
}
