package decide

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

var efforts = []Level{
	{ID: "low", Description: "mechanical"},
	{ID: "medium", Description: "routine"},
	{ID: "high", Description: "tricky", Examples: []string{"find the race"}},
	{ID: "xhigh", Description: "deep"},
}

func TestChooseLevelTakesTheWeightedLevel(t *testing.T) {
	cases := []struct {
		name          string
		probabilities map[string]float64
		want          string
	}{
		{"certain", map[string]float64{"medium": 1}, "medium"},
		{"an even split settles on the stronger", map[string]float64{"medium": 0.5, "high": 0.5}, "high"},
		{"an even split over a gap skips the empty levels", map[string]float64{"low": 0.3, "medium": 0.2, "xhigh": 0.5}, "xhigh"},
		{"mostly low", map[string]float64{"low": 0.7, "medium": 0.3}, "low"},
		{"split extremes never meet in the middle", map[string]float64{"low": 0.5, "xhigh": 0.5}, "xhigh"},
		{"far tail does not move the median", map[string]float64{"low": 0.8, "xhigh": 0.2}, "low"},
		{"unnormalized", map[string]float64{"high": 3, "xhigh": 1}, "high"},
	}
	for _, c := range cases {
		if got, ok := ChooseLevel(efforts, c.probabilities); !ok || got != c.want {
			t.Errorf("%s: ChooseLevel = %q, %v, want %q", c.name, got, ok, c.want)
		}
	}
	if _, ok := ChooseLevel(efforts, map[string]float64{"max": 1}); ok {
		t.Error("ChooseLevel chose without probability on any level")
	}
}

func TestNormalizeLevelsDropsInvalidProbabilities(t *testing.T) {
	levels := normalizeLevels(map[string]float64{"low": math.Inf(1), "medium": math.NaN(), "high": 3, "xhigh": 1, "max": 5}, efforts)
	if levels["high"] != 0.75 || levels["xhigh"] != 0.25 || levels["low"] != 0 || len(levels) != 4 {
		t.Fatalf("normalizeLevels = %v", levels)
	}
	if normalizeLevels(map[string]float64{"low": -1}, efforts) != nil {
		t.Fatal("normalizeLevels kept a verdict without positive probability")
	}
}

func TestChainChoosesTheLevelOfTheDecidingVerdict(t *testing.T) {
	jev := &fakeDecider{name: "jev", verdict: Verdict{Choice: "glm-5.3",
		Probabilities: map[string]float64{"glm-5.3": 1}, Levels: map[string]float64{"medium": 0.4, "high": 0.6}}}
	decision, ok := NewChain([]Step{{Decider: jev, Timeout: time.Second}}, 0.10, DoubtEscalate, efforts).
		Decide(context.Background(), "state", three)
	if !ok || decision.Level != "high" || decision.Levels["high"] != 0.6 || decision.Attempts[0].Level != "high" {
		t.Fatalf("decision = %+v", decision)
	}
	silent := &fakeDecider{name: "jev", verdict: Verdict{Choice: "glm-5.3", Levels: map[string]float64{"max": 1}}}
	decision, ok = NewChain([]Step{{Decider: silent, Timeout: time.Second}}, 0.10, DoubtEscalate, efforts).
		Decide(context.Background(), "state", three)
	if !ok || decision.Model != "glm-5.3" || decision.Level != "" || decision.Levels != nil {
		t.Fatalf("a verdict without a usable level = %+v, %v", decision, ok)
	}
}

func TestSystemOneAsksTheReasoningScoreInTheSameRequest(t *testing.T) {
	var received struct {
		Questions map[string]struct {
			Type         string          `json:"type"`
			Instructions Instructions    `json:"instructions"`
			Criteria     json.RawMessage `json:"criteria"`
		} `json:"questions"`
	}
	server := newTestEndpoint(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&received)
		_, _ = w.Write([]byte(`{"model":"jev-1.13.0","answers":{
			"model":{"type":"choice","choice":"gpt-6.1-sol","probabilities":{"glm-5.3":0.05,"gpt-6.1-sol":0.95}},
			"reasoning":{"type":"score","score":2.2,"confidence":0.6,"probabilities":{"0":0,"1":0.1,"2":0.6,"3":0.3}}}}`))
	}))
	verdict, err := NewSystemOne("jev", server.URL, "jev-latest", "k", server.Client()).
		Decide(context.Background(), "state", three, efforts)
	if err != nil || verdict.Choice != "gpt-6.1-sol" || verdict.Levels["high"] != 0.6 || verdict.Levels["xhigh"] != 0.3 {
		t.Fatalf("Decide = %+v, %v", verdict, err)
	}
	score := received.Questions["reasoning"]
	var criteria []criterion
	if err := json.Unmarshal(score.Criteria, &criteria); err != nil || score.Type != "score" || len(criteria) != 4 ||
		criteria[2].What != "tricky" || criteria[2].Examples[0] != "find the race" ||
		!strings.Contains(score.Instructions.Question, "How much reasoning") || received.Questions["model"].Type != "choice" {
		t.Fatalf("reasoning question = %+v (%v)", score, err)
	}
}

func TestScoreLevelsFallsBackToTheScore(t *testing.T) {
	score := 2.6
	if levels := scoreLevels(systemOneAnswer{Score: &score}, efforts); levels["xhigh"] != 1 || len(levels) != 1 {
		t.Fatalf("scoreLevels of a bare score = %v", levels)
	}
	if levels := scoreLevels(systemOneAnswer{Probabilities: map[string]float64{"7": 1, "x": 1}}, efforts); levels != nil {
		t.Fatalf("scoreLevels kept indexes outside the levels: %v", levels)
	}
}

func TestSystemOneWithoutAReasoningAnswerStillDecides(t *testing.T) {
	server := newTestEndpoint(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"answers":{"model":{"choice":"glm-5.3","probabilities":{"glm-5.3":1}}}}`))
	}))
	verdict, err := NewSystemOne("jev", server.URL, "jev-latest", "k", server.Client()).
		Decide(context.Background(), "state", three, efforts)
	if err != nil || verdict.Choice != "glm-5.3" || verdict.Levels != nil {
		t.Fatalf("Decide = %+v, %v", verdict, err)
	}
}

func TestJudgeReadsTheReasoningAnswer(t *testing.T) {
	cases := map[string]map[string]float64{
		`{"probabilities":{"glm-5.3":1},"reasoning":{"medium":0.8,"high":0.2}}`: {"medium": 0.8, "high": 0.2},
		`{"probabilities":{"glm-5.3":1},"reasoning":" high "}`:                  {"high": 1},
		`{"probabilities":{"glm-5.3":1},"reasoning":null}`:                      nil,
		`{"probabilities":{"glm-5.3":1}}`:                                       nil,
	}
	for reply, want := range cases {
		verdict, err := parseJudgement(reply, three)
		if err != nil || len(verdict.Levels) != len(want) {
			t.Errorf("parseJudgement(%s) = %+v, %v", reply, verdict, err)
			continue
		}
		for id, p := range want {
			if verdict.Levels[id] != p {
				t.Errorf("parseJudgement(%s) level %s = %v, want %v", reply, id, verdict.Levels[id], p)
			}
		}
	}
}

func TestJudgeListsTheReasoningLevels(t *testing.T) {
	var request struct {
		Messages []chatMessage `json:"messages"`
	}
	server := newTestEndpoint(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&request)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"probabilities\":{\"glm-5.3\":1},\"reasoning\":{\"low\":1}}"}}]}`))
	}))
	verdict, err := NewJudge("judge", server.URL, "gpt-6-luna", "k", server.Client()).
		Decide(context.Background(), "state", three, efforts)
	if err != nil || verdict.Levels["low"] != 1 || len(request.Messages) != 2 ||
		!strings.Contains(request.Messages[0].Content, "How much reasoning") ||
		!strings.Contains(request.Messages[0].Content, `"reasoning": {"<level id>": <number>}`) ||
		!strings.Contains(request.Messages[0].Content, "Reasoning levels, lightest first:\n- low: mechanical\n") ||
		!strings.Contains(request.Messages[0].Content, "- high: tricky Examples: find the race\n") {
		t.Fatalf("Decide = %+v, %v; request %+v", verdict, err, request)
	}
}

func TestPolicyRecordsAndRebuildsTheReasoningQuestion(t *testing.T) {
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
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"probabilities\":{\"glm-5.3\":1},\"reasoning\":\"medium\"}"}}]}`))
		case jevDown:
			w.WriteHeader(http.StatusServiceUnavailable)
		case bytes.Contains(body, []byte(`"pair":`)):
			_, _ = w.Write([]byte(`{"answers":{"pair":{"choice":"glm-5.3","probabilities":{"glm-5.3":0.6,"gpt-6.1-sol":0.4}}}}`))
		default:
			_, _ = w.Write([]byte(`{"answers":{"model":{"choice":"glm-5.3","probabilities":{"glm-5.3":0.5,"gpt-6.1-sol":0.5}},
				"reasoning":{"probabilities":{"2":1}}}}`))
		}
	}))
	jev := Step{Decider: NewSystemOne("jev", server.URL+"/systemone", "jev-latest", "k", server.Client()), Timeout: time.Second}
	judge := Step{Decider: NewJudge("deepseek", server.URL+"/chat", "deepseek-flash", "k", server.Client()), Timeout: time.Second}
	chain := NewChain([]Step{jev, judge}, 0.10, DoubtPair, efforts)
	policy := chain.Policy(three)
	if policy.Reasoning == nil || len(policy.Reasoning.Levels) != 4 ||
		policy.Fingerprint() == NewChain([]Step{jev, judge}, 0.10, DoubtPair, nil).Policy(three).Fingerprint() ||
		NewChain([]Step{jev, judge}, 0.10, DoubtPair, nil).Policy(three).Reasoning != nil {
		t.Fatalf("policy reasoning = %+v", policy.Reasoning)
	}
	const state = "images=no\n\nadd a --dry-run flag"

	decision, ok := chain.Decide(context.Background(), state, three)
	if !ok || decision.Level != "high" || len(sent) != 2 {
		t.Fatalf("decision = %+v", decision)
	}
	choice, _ := policy.Request("jev", state)
	pair, _ := policy.PairRequest("jev", state, "glm-5.3", "gpt-6.1-sol")
	if !bytes.Equal(choice, sent[0]) || !bytes.Equal(pair, sent[1]) || bytes.Contains(pair, []byte(`"reasoning"`)) {
		t.Errorf("jev requests rebuilt as\n%s\n%s\nsent\n%s\n%s", choice, pair, sent[0], sent[1])
	}

	jevDown, sent = true, nil
	decision, ok = chain.Decide(context.Background(), state, three)
	if !ok || decision.Decider != "deepseek" || decision.Level != "medium" {
		t.Fatalf("judge decision = %+v", decision)
	}
	if rebuilt, _ := policy.Request("deepseek", state); !bytes.Equal(rebuilt, sent[1]) {
		t.Errorf("judge request rebuilt as\n%s\nsent\n%s", rebuilt, sent[1])
	}
}
