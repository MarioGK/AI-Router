package decide

import (
	"context"
	"errors"
	"maps"
	"math"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ladder is four ladder steps and a specialist, with list costs per task.
var ladder = []Candidate{
	{ID: "deepseek-flash"}, {ID: "glm-5.3"}, {ID: "ui", Specialist: true}, {ID: "gpt-6.1-sol"}, {ID: "claude-opus-5-5"},
}

var perTask = map[string]float64{"deepseek-flash": 0.10, "glm-5.3": 0.25, "ui": 0.50, "gpt-6.1-sol": 1.20, "claude-opus-5-5": 2.00}

func economy(miss float64) Economy {
	return Economy{Miss: miss, Task: func(model, _ string, _ []Candidate) (float64, bool) {
		cost, ok := perTask[model]
		return cost, ok
	}}
}

func TestTheCheapestWayToGetATaskDone(t *testing.T) {
	e := economy(1)
	cases := []struct {
		name          string
		probabilities map[string]float64
		want          string
	}{
		{"a sure tiny edit stays", map[string]float64{"deepseek-flash": 0.95, "glm-5.3": 0.05}, "deepseek-flash"},
		// 0.10 + 0.3*0.25*2 = 0.25 against 0.25: an even trade keeps the lighter step.
		{"an even trade stays", map[string]float64{"deepseek-flash": 0.7, "glm-5.3": 0.3}, "deepseek-flash"},
		{"a likely miss moves up", map[string]float64{"deepseek-flash": 0.6, "glm-5.3": 0.4}, "glm-5.3"},
		// Opus holds half: 0.10 + 0.5*2*2 = 2.10 against 2.00 on Opus.
		{"a split verdict goes where it is cheapest", map[string]float64{"deepseek-flash": 0.5, "claude-opus-5-5": 0.5}, "claude-opus-5-5"},
		{"the specialist never counts", map[string]float64{"deepseek-flash": 0.5, "ui": 0.5}, "deepseek-flash"},
	}
	for _, c := range cases {
		index, costs, ok := e.cheapest(ladder, Verdict{Choice: "deepseek-flash", Probabilities: c.probabilities}, "")
		if !ok || ladder[index].ID != c.want {
			t.Errorf("%s: cheapest = %s (%v), done %v", c.name, ladder[index].ID, ok, costs.Done)
		}
	}
	index, costs, ok := e.cheapest(ladder, Verdict{Choice: "glm-5.3", Probabilities: map[string]float64{"deepseek-flash": 0.9, "glm-5.3": 0.1}}, "")
	if !ok || ladder[index].ID != "glm-5.3" || len(costs.Task) != 3 || costs.Task["deepseek-flash"] != 0 {
		t.Fatalf("doubt towards a lighter step moved the thread: %s, %+v", ladder[index].ID, costs)
	}
	if _, _, ok := e.cheapest(ladder, Verdict{Choice: "ui"}, ""); ok {
		t.Fatal("a specialist choice was priced")
	}
	priceless := Economy{Miss: 1, Task: func(model, _ string, _ []Candidate) (float64, bool) { return 0, model != "gpt-6.1-sol" }}
	if _, _, ok := priceless.cheapest(ladder, Verdict{Choice: "glm-5.3"}, ""); ok {
		t.Fatal("a ladder without every cost was priced")
	}
}

func TestMovingUpPaysFromTheBreakEvenProbability(t *testing.T) {
	e := economy(1)
	if cutoff := e.cutoff(0.25, 1.20); math.Abs(cutoff-0.95/2.4) > 1e-12 {
		t.Fatalf("cutoff = %v", cutoff)
	}
	if e.cutoff(1.20, 1.20) != pairCutoff || e.cutoff(0, 0) != pairCutoff {
		t.Fatal("a free or cheaper step still needs majority task fit")
	}
}

func TestACheaperTierWithoutPolicyFitCannotReplaceTheChoice(t *testing.T) {
	e := Economy{Miss: 1, Task: func(model, _ string, _ []Candidate) (float64, bool) {
		if model == "glm-5.3" {
			return 0, true
		}
		cost, ok := perTask[model]
		return cost, ok
	}}
	for _, probabilities := range []map[string]float64{
		{"deepseek-flash": 1},
		{"deepseek-flash": 0.9, "gpt-6.1-sol": 0.1},
	} {
		index, costs, ok := e.cheapest(ladder, Verdict{Choice: "deepseek-flash", Probabilities: probabilities}, "")
		if !ok || ladder[index].ID != "deepseek-flash" {
			t.Fatalf("a tier without policy fit replaced the choice: %s (%v), probabilities %v, costs %+v", ladder[index].ID, ok, probabilities, costs)
		}
	}
}

func TestEconomyUsesOfferedCandidatesForSpillPricing(t *testing.T) {
	e := Economy{Miss: 3, Task: func(model, _ string, offered []Candidate) (float64, bool) {
		if model == "glm-5.3" && slices.ContainsFunc(offered, func(c Candidate) bool { return c.ID == "gpt-6.1-sol" }) {
			model = "gpt-6.1-sol"
		}
		cost, ok := perTask[model]
		return cost, ok
	}}
	verdict := Verdict{Choice: "glm-5.3", Probabilities: map[string]float64{"glm-5.3": 0.85, "claude-opus-5-5": 0.15}}
	for _, c := range []struct {
		name       string
		candidates []Candidate
		want       string
	}{
		{"unavailable spill retains its own cost", []Candidate{{ID: "glm-5.3"}, {ID: "claude-opus-5-5"}}, "glm-5.3"},
		{"available spill changes the cost", []Candidate{{ID: "glm-5.3"}, {ID: "gpt-6.1-sol"}, {ID: "claude-opus-5-5"}}, "claude-opus-5-5"},
	} {
		t.Run(c.name, func(t *testing.T) {
			index, costs, ok := e.cheapest(c.candidates, verdict, "medium")
			if !ok || c.candidates[index].ID != c.want {
				t.Fatalf("choice = %s (%v), want %s; costs %+v", c.candidates[index].ID, ok, c.want, costs)
			}
		})
	}
}

func TestAPricedChainSettlesDoubtAtTheBreakEven(t *testing.T) {
	jev := &pairingDecider{fakeDecider: fakeDecider{name: "jev", verdict: Verdict{Choice: "glm-5.3",
		Probabilities: map[string]float64{"glm-5.3": 0.4, "gpt-6.1-sol": 0.6}}}}
	chain := NewChain([]Step{{Decider: jev, Timeout: time.Second}}, 0.10, DoubtPair, nil).Price(economy(1))
	// The pair answer puts 30% on Sol, under the 0.95/2.4 = 40% it takes.
	jev.pair = Verdict{Probabilities: map[string]float64{"glm-5.3": 0.7, "gpt-6.1-sol": 0.3}}
	decision, ok := chain.Decide(context.Background(), "state", ladder)
	if !ok || decision.Model != "glm-5.3" || decision.Escalated || decision.Costs.Done["gpt-6.1-sol"] != 1.20 {
		t.Fatalf("decision = %+v", decision)
	}
	pair := decision.Attempts[1]
	if math.Abs(pair.Cutoff-0.95/2.4) > 1e-12 || pair.Above != 0.3 {
		t.Fatalf("pair attempt = %+v", pair)
	}
	jev.pair = Verdict{Probabilities: map[string]float64{"glm-5.3": 0.5, "gpt-6.1-sol": 0.5}}
	if decision, _ := chain.Decide(context.Background(), "state", ladder); decision.Model != "gpt-6.1-sol" {
		t.Fatalf("a pair answer past the break-even stayed: %+v", decision)
	}
	if miss := chain.Policy(ladder).Miss; miss == nil || *miss != 1 {
		t.Fatalf("policy miss = %v", miss)
	}
	if NewChain(nil, 0.10, DoubtPair, nil).Policy(ladder).Miss != nil {
		t.Fatal("an unpriced chain records a miss")
	}
}

// pairingDecider answers pair questions with pair.
type pairingDecider struct {
	fakeDecider
	pair Verdict
}

func (p *pairingDecider) Pair(context.Context, string, Candidate, Candidate) (Verdict, error) {
	return p.pair, nil
}

func TestOnlyOneTurnProbesADeciderAfterItsRest(t *testing.T) {
	var calls atomic.Int32
	release := make(chan struct{})
	down := &blockingDecider{name: "jev", calls: &calls, release: release}
	chain := NewChain([]Step{{Decider: down, Timeout: time.Second}}, 0.10, DoubtEscalate, nil).Rest(1, time.Minute)
	now := time.Date(2026, 9, 29, 12, 0, 0, 0, time.UTC)
	var mu sync.Mutex
	chain.now = func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	close(release)
	chain.Decide(context.Background(), "state", three)
	mu.Lock()
	now = now.Add(time.Minute)
	mu.Unlock()
	down.release = make(chan struct{})
	var wg sync.WaitGroup
	for range 10 {
		wg.Go(func() { chain.Decide(context.Background(), "state", three) })
	}
	for calls.Load() < 2 {
		time.Sleep(time.Millisecond)
	}
	time.Sleep(20 * time.Millisecond)
	close(down.release)
	wg.Wait()
	if n := calls.Load(); n != 2 {
		t.Fatalf("the decider was called %d times; want the first failure and one probe", n)
	}
}

// blockingDecider fails every turn once release is closed.
type blockingDecider struct {
	name    string
	calls   *atomic.Int32
	release chan struct{}
}

func (b *blockingDecider) Name() string { return b.name }

func (b *blockingDecider) Info() DeciderInfo { return DeciderInfo{Name: b.name, Kind: "fake"} }

func (b *blockingDecider) Decide(ctx context.Context, _ string, _ []Candidate, _ []Level) (Verdict, error) {
	b.calls.Add(1)
	select {
	case <-b.release:
	case <-ctx.Done():
	}
	return Verdict{}, errors.New("status 503")
}

func TestAHangGivenItsWholeTimeoutCounts(t *testing.T) {
	slow := &fakeDecider{name: "jev", wait: time.Second}
	chain := NewChain([]Step{{Decider: slow, Timeout: 20 * time.Millisecond}}, 0.10, DoubtEscalate, nil).Rest(1, time.Minute)
	// The chain's deadline and the step's end together.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	chain.Decide(ctx, "state", three)
	if health := chain.Health(); health[0].Failures != 1 || health[0].Resting.IsZero() {
		t.Fatalf("health = %+v", health)
	}
}

func TestScoreAnswersTolerateTheirVariants(t *testing.T) {
	two, huge := 2.0, 1e300
	cases := []struct {
		name   string
		answer systemOneAnswer
		want   string
	}{
		{"zero probabilities fall back to the score", systemOneAnswer{Score: &two, Probabilities: map[string]float64{"0": 0, "1": 0}}, "high"},
		{"level ids as keys", systemOneAnswer{Probabilities: map[string]float64{"medium": 0.2, "xhigh": 0.8}}, "xhigh"},
		{"a huge score clamps to the top", systemOneAnswer{Score: &huge}, "xhigh"},
		{"a padded index is no index", systemOneAnswer{Probabilities: map[string]float64{"01": 1, "0": 0.1}}, "low"},
	}
	for _, c := range cases {
		levels := normalizeLevels(scoreLevels(c.answer, efforts), efforts)
		if got, ok := ChooseLevel(efforts, levels); !ok || got != c.want {
			t.Errorf("%s: level %q (%v) from %v", c.name, got, ok, levels)
		}
	}
}

func TestJudgeRepliesAreReadForTheCandidates(t *testing.T) {
	reply := `Consider {"choice":"the cheaper one"} first. {"probabilities":{"glm-5.3":0.9,"gpt-6.1-sol":0.1},"reasoning":{"High":1}}`
	verdict, err := parseJudgement(reply, three)
	if err != nil || verdict.Probabilities["glm-5.3"] != 0.9 || verdict.Levels["high"] != 1 {
		t.Fatalf("parseJudgement = %+v, %v", verdict, err)
	}
	if verdict, err := parseJudgement(`{"reasoning":" Medium ","choice":"glm-5.3"}`, three); err != nil || verdict.Levels["medium"] != 1 {
		t.Fatalf("a capitalized level = %+v, %v", verdict, err)
	}
	if _, err := parseJudgement(`{"claude-opus-9": 1}`, three); err == nil {
		t.Fatal("a reply judging no candidate was accepted")
	}
}

func TestPairQuestionsSettleOneStepThatCosts(t *testing.T) {
	ask := func(economy Economy, probabilities map[string]float64) (Decision, *pairingDecider) {
		jev := &pairingDecider{fakeDecider: fakeDecider{name: "jev", verdict: Verdict{Choice: "deepseek-flash", Probabilities: probabilities}},
			pair: Verdict{Probabilities: map[string]float64{"deepseek-flash": 1}}}
		decision, _ := NewChain([]Step{{Decider: jev, Timeout: time.Second}}, 0.10, DoubtPair, nil).Price(economy).
			Decide(context.Background(), "state", ladder)
		return decision, jev
	}
	// Spread over several steps, the verdict goes straight to the cheapest.
	decision, _ := ask(economy(1), map[string]float64{"deepseek-flash": 0.3, "glm-5.3": 0.1, "gpt-6.1-sol": 0.1, "claude-opus-5-5": 0.5})
	if decision.Model != "claude-opus-5-5" || len(decision.Attempts) != 1 || decision.Attempts[0].Outcome != Decided {
		t.Fatalf("a multi-step jump = %+v", decision)
	}
	// A negligible fit does not consume the equal-priced stronger tier.
	free := Economy{Miss: 1, Task: func(model, _ string, _ []Candidate) (float64, bool) {
		return map[string]float64{"deepseek-flash": 0.1, "glm-5.3": 0.1, "gpt-6.1-sol": 1, "claude-opus-5-5": 2}[model], true
	}}
	decision, _ = ask(free, map[string]float64{"deepseek-flash": 0.9, "glm-5.3": 0.1})
	if decision.Model != "deepseek-flash" || len(decision.Attempts) != 1 {
		t.Fatalf("an unsupported equal-cost step up = %+v", decision)
	}
}

func TestAbandonedTurnsNeverRestADecider(t *testing.T) {
	slow := &fakeDecider{name: "jev", wait: time.Second}
	chain := NewChain([]Step{{Decider: slow, Timeout: 50 * time.Millisecond}}, 0.10, DoubtEscalate, nil).Rest(1, time.Minute)
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()
	chain.Decide(ctx, "state", three)
	if health := chain.Health(); health[0].Failures != 0 || !health[0].Resting.IsZero() {
		t.Fatalf("an abandoned turn counted: %+v", health)
	}
}

func TestFreeTierRequiresTaskFitAndSettlesPairs(t *testing.T) {
	e := Economy{Miss: 3, Task: func(model, _ string, _ []Candidate) (float64, bool) {
		if model == "glm-5.3" {
			return 0, true
		}
		cost, ok := perTask[model]
		return cost, ok
	}}
	for _, c := range []struct {
		probability float64
		want        string
	}{
		{0.01, "deepseek-flash"}, {0.4, "deepseek-flash"}, {0.6, "glm-5.3"},
	} {
		v := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 1 - c.probability, "glm-5.3": c.probability}}
		i, _, ok := e.cheapest(ladder, v, "low")
		if !ok || ladder[i].ID != c.want {
			t.Fatalf("free tier with fit %v chose %s", c.probability, ladder[i].ID)
		}
	}
	jev := &pairingDecider{fakeDecider: fakeDecider{name: "jev", verdict: Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.4, "glm-5.3": 0.6}}},
		pair: Verdict{Probabilities: map[string]float64{"deepseek-flash": 0.9, "glm-5.3": 0.1}}}
	chain := NewChain([]Step{{Decider: jev, Timeout: time.Second}}, 0.1, DoubtPair, nil).Price(e)
	d, ok := chain.Decide(t.Context(), "exact local edit", ladder)
	if !ok || d.Model != "deepseek-flash" || len(d.Attempts) != 2 || d.Attempts[1].Cutoff != pairCutoff {
		t.Fatalf("free pair did not settle task fit: %+v", d)
	}
}

// learning is an economy that has learned how long a task takes on each
// step and how each step's miss rate scales the penalty.
func learning(miss, timeValue float64, hours, factor map[string]float64) Economy {
	e := economy(miss)
	e.TimeValue = timeValue
	e.Learn = func(model, _ string, _ []Candidate) Learned {
		f := factor[model]
		if f == 0 {
			f = 1
		}
		return Learned{Hours: hours[model], MissRate: 0.1 * f, Factor: f}
	}
	return e
}

func TestNothingLearnedAndNoTimeValueChangesNoScore(t *testing.T) {
	verdict := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.7, "glm-5.3": 0.2, "gpt-6.1-sol": 0.1}}
	plain := economy(3)
	_, want, _ := plain.cheapest(ladder, verdict, "")
	plainCutoff := plain.cutoff(0.10, 0.25)
	for _, e := range []Economy{
		learning(3, 0, map[string]float64{"deepseek-flash": 1, "glm-5.3": 2}, nil),
		learning(3, 0, nil, map[string]float64{}),
	} {
		_, got, _ := e.cheapest(ladder, verdict, "")
		if !maps.Equal(got.Done, want.Done) || got.Time != nil || got.Miss != nil || got.Hours != nil {
			t.Fatalf("an economy that learned nothing scored %+v, want %+v", got, want)
		}
		if got, want := e.stepCutoff(got, "deepseek-flash", "glm-5.3"), plainCutoff; got != want {
			t.Fatalf("cutoff = %v, want %v", got, want)
		}
	}
}

func TestTimeValueMakesASlowStepPayForItsTime(t *testing.T) {
	verdict := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.8, "glm-5.3": 0.2}}
	// Without time: 0.10 + 0.2*0.25*2 = 0.20 against 0.25, deepseek-flash.
	plain := economy(1)
	if index, _, _ := plain.cheapest(ladder, verdict, ""); ladder[index].ID != "deepseek-flash" {
		t.Fatalf("without time = %s", ladder[index].ID)
	}
	// deepseek-flash takes an hour and glm-5.3 six minutes at $1 an hour:
	// 1.10 + 0.2*0.35*2 = 1.24 against 0.35.
	e := learning(1, 1, map[string]float64{"deepseek-flash": 1, "glm-5.3": 0.1, "gpt-6.1-sol": 0.1, "claude-opus-5-5": 0.1}, nil)
	index, costs, ok := e.cheapest(ladder, verdict, "")
	if !ok || ladder[index].ID != "glm-5.3" {
		t.Fatalf("a slow step kept the task: %s, %+v", ladder[index].ID, costs)
	}
	if costs.Time["deepseek-flash"] != 1 || costs.Hours["glm-5.3"] != 0.1 || costs.Task["deepseek-flash"] != 0.10 {
		t.Fatalf("time was not recorded beside the cost: %+v", costs)
	}
	if done := costs.Done["deepseek-flash"]; math.Abs(done-1.24) > 1e-12 {
		t.Fatalf("deepseek-flash to done = %v, want 1.24", done)
	}
	if got := e.stepCutoff(costs, "deepseek-flash", "glm-5.3"); got > 0 {
		t.Fatalf("a dearer step that is cheaper with its time was put to a pair question: cutoff %v", got)
	}
}

func TestALearnedMissRateScalesItsStepsPenalty(t *testing.T) {
	verdict := Verdict{Choice: "deepseek-flash", Probabilities: map[string]float64{"deepseek-flash": 0.75, "glm-5.3": 0.25}}
	// With miss 1: 0.10 + 0.25*0.25*2 = 0.225 against 0.25, deepseek-flash.
	plain := economy(1)
	if index, _, _ := plain.cheapest(ladder, verdict, ""); ladder[index].ID != "deepseek-flash" {
		t.Fatalf("without learning = %s", ladder[index].ID)
	}
	// deepseek-flash moved up twice as often as the fleet: miss 2,
	// 0.10 + 0.25*0.25*3 = 0.2875.
	e := learning(1, 0, nil, map[string]float64{"deepseek-flash": 2})
	index, costs, ok := e.cheapest(ladder, verdict, "")
	if !ok || ladder[index].ID != "glm-5.3" {
		t.Fatalf("a step that misses often kept the task: %s, %+v", ladder[index].ID, costs)
	}
	if costs.Miss["deepseek-flash"] != 2 || costs.MissRate["deepseek-flash"] != 0.2 {
		t.Fatalf("the learned penalty was not recorded: %+v", costs)
	}
	if _, moved := costs.Miss["glm-5.3"]; moved {
		t.Fatalf("a step without a learned rate recorded one: %+v", costs.Miss)
	}
	if got, want := e.stepCutoff(costs, "deepseek-flash", "glm-5.3"), cutoffAt(0.10, 0.25, 2); got != want {
		t.Fatalf("cutoff = %v, want %v with the learned penalty", got, want)
	}
	// A step that rarely misses keeps a task the plain economy moves up.
	verdict.Probabilities = map[string]float64{"deepseek-flash": 0.6, "glm-5.3": 0.4}
	rare := learning(1, 0, nil, map[string]float64{"deepseek-flash": 0.25})
	if index, _, _ := rare.cheapest(ladder, verdict, ""); ladder[index].ID != "deepseek-flash" {
		t.Fatalf("a step that rarely misses lost the task: %s", ladder[index].ID)
	}
}

// timed is an economy over two steps, slow and fast, with the given task
// and time costs.
func timed(task, time map[string]float64) Economy {
	return Economy{Miss: 1, TimeValue: 1,
		Task:  func(model, _ string, _ []Candidate) (float64, bool) { return task[model], true },
		Learn: func(model, _ string, _ []Candidate) Learned { return Learned{Hours: time[model], Factor: 1} }}
}

var twoSteps = []Candidate{{ID: "slow"}, {ID: "fast"}}

func TestADearerStepCheaperWithItsTimeNeedsNoPairQuestion(t *testing.T) {
	// $1 + $9 of time against $2 + $0: fast is cheaper at any fit.
	e := timed(map[string]float64{"slow": 1, "fast": 2}, map[string]float64{"slow": 9, "fast": 0})
	jev := &pairingDecider{fakeDecider: fakeDecider{name: "jev", verdict: Verdict{Choice: "slow",
		Probabilities: map[string]float64{"slow": 0.8, "fast": 0.2}}}}
	jev.pair = Verdict{Probabilities: map[string]float64{"slow": 0.8, "fast": 0.2}}
	chain := NewChain([]Step{{Decider: jev, Timeout: time.Second}}, 0.10, DoubtPair, nil).Price(e)
	decision, ok := chain.Decide(context.Background(), "state", twoSteps)
	if !ok || decision.Model != "fast" || len(decision.Attempts) != 1 {
		t.Fatalf("decision = %+v", decision)
	}
	_, costs, _ := e.cheapest(twoSteps, Verdict{Choice: "slow", Probabilities: map[string]float64{"slow": 0.8, "fast": 0.2}}, "")
	if cutoff := e.stepCutoff(costs, "slow", "fast"); cutoff > 0 {
		t.Fatalf("cutoff = %v, want none", cutoff)
	}
}

func TestAFreeStepKeepsItsMajorityFloorWhateverItsTime(t *testing.T) {
	// $1 + $0 of time against a free step with $2 of time: the economic
	// cutoff alone would be 0.25.
	e := timed(map[string]float64{"slow": 1, "fast": 0}, map[string]float64{"slow": 0, "fast": 2})
	verdict := Verdict{Choice: "slow", Probabilities: map[string]float64{"slow": 0.4, "fast": 0.6}}
	_, costs, _ := e.cheapest(twoSteps, verdict, "")
	if cutoff := e.stepCutoff(costs, "slow", "fast"); cutoff != pairCutoff {
		t.Fatalf("cutoff = %v, want the majority floor", cutoff)
	}
	jev := &pairingDecider{fakeDecider: fakeDecider{name: "jev", verdict: verdict}}
	jev.pair = Verdict{Probabilities: map[string]float64{"slow": 0.7, "fast": 0.3}}
	chain := NewChain([]Step{{Decider: jev, Timeout: time.Second}}, 0.10, DoubtPair, nil).Price(e)
	if decision, _ := chain.Decide(context.Background(), "state", twoSteps); decision.Model != "slow" {
		t.Fatalf("a free step without majority fit on the pair question kept the task: %+v", decision)
	}
}
