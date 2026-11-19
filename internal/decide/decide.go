// Package decide picks the model a new thread is pinned to and, with the
// reasoning chooser on, the reasoning effort it runs at. A chain of deciders
// (Jev-style systemone endpoints, any LLM through CLIProxy) is tried in
// order; the first valid verdict wins and is escalated one ladder step when
// the probability mass on stronger candidates reaches the threshold.
package decide

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"sync"
	"time"
)

// Candidate is a model a thread can be pinned to. Non-specialist candidates
// form the escalation ladder in list order, lightest first; specialists are
// picked only on their own merit.
type Candidate struct {
	ID string `json:"id"`
	// Description is what the candidate is for, NotFor what it is not for,
	// and Examples are requests it should get.
	Description       string   `json:"description"`
	NotFor            string   `json:"not_for,omitempty"`
	Examples          []string `json:"examples,omitempty"`
	Specialist        bool     `json:"specialist,omitempty"`
	ExpectedLatencyMS int64    `json:"expected_latency_ms,omitzero"`
	ExpectedCostUSD   *float64 `json:"expected_cost_usd,omitzero"`
}

// Verdict is a decider's answer: a choice plus a probability per candidate,
// with what the decider reports about itself.
type Verdict struct {
	Choice        string
	Probabilities map[string]float64
	// Levels is the probability per reasoning level, when the decider was
	// asked the reasoning question and answered it.
	Levels map[string]float64
	// Model is the decider model that answered (jev-1.13.0, gpt-6-luna).
	Model string
	// Confidence is the decider's own confidence in its choice, when it
	// reports one (Jev).
	Confidence float64
	// Reason is the decider's short explanation, when it gives one (an LLM
	// judge).
	Reason       string
	InputTokens  int64
	OutputTokens int64
	// UsageKnown distinguishes reported zero usage from a missing usage answer.
	UsageKnown bool
	// Response is the decider's raw reply, cut to maxResponse bytes.
	Response string
	// Digest identifies the request the decider was sent; a decider sets
	// it on errors too.
	Digest string
}

// maxResponse bounds a decider's raw reply kept with its attempt.
const maxResponse = 8 << 10

// Decider judges which candidate should handle a request and, when levels
// are given, how much reasoning it needs.
type Decider interface {
	Name() string
	// Info names the decider's kind and model for the policy record.
	Info() DeciderInfo
	Decide(ctx context.Context, state string, candidates []Candidate, levels []Level) (Verdict, error)
}

// Step is a decider with its time budget.
type Step struct {
	Decider Decider
	Timeout time.Duration
}

// Pairer is a decider that can settle a doubt between two ladder steps.
type Pairer interface {
	Pair(ctx context.Context, state string, cheap, strong Candidate) (Verdict, error)
}

// pairCutoff is the probability on the stronger step at which a pair
// answer moves the decision up.
const pairCutoff = 0.5

// Chain tries its steps in order.
type Chain struct {
	steps     []Step
	threshold float64
	doubt     string
	levels    []Level
	// trip failures in a row rest a decider for rest; zero never rests one.
	trip int
	rest time.Duration
	now  func() time.Time
	// economy, when set, weighs the cost of getting a task done instead of
	// the threshold.
	economy *Economy

	mu     sync.Mutex
	health map[string]*health
}

// Selection is the chain's deciders selected for a request. Its steps stay
// fixed while the original chain's economy and decider health remain shared.
type Selection struct {
	chain       *Chain
	steps       []Step
	preferences Preferences
	// prior, when set, is what similar past prompts suggest, blended into
	// every verdict.
	prior *Prior
}

// Select snapshots the steps enabled for a request, preserving chain order.
// A nil predicate selects every step. Excluded steps are never asked and
// leave no attempt or change to their health.
func (c *Chain) Select(enabled func(DeciderInfo) bool) *Selection {
	selection := &Selection{chain: c}
	for _, step := range c.steps {
		if enabled == nil || enabled(step.Decider.Info()) {
			selection.steps = append(selection.steps, step)
		}
	}
	return selection
}

// health is a decider's run of failures and, once it tripped, when its
// rest ends; probing is set while one turn tries it after its rest.
type health struct {
	failures int
	until    time.Time
	probing  bool
	// generation changes when the decider rests, so earlier turns cannot
	// settle a probe that started after them.
	generation uint64
}

// Turn outcomes a decider's health records.
const (
	turnSucceeded = iota
	turnFailed
	// turnCut: the chain's deadline ended a turn the decider was not given
	// its whole timeout for, which says nothing about the decider.
	turnCut
)

// DeciderHealth is a decider's state in a chain that rests failing
// deciders.
type DeciderHealth struct {
	Name string `json:"name"`
	// Failures is the run of failed turns; a success ends it.
	Failures int `json:"failures"`
	// Resting is when the decider's turns start again, zero when it is
	// not resting; Probing is set while one turn tries it after its rest.
	Resting time.Time `json:"resting,omitzero"`
	Probing bool      `json:"probing,omitempty"`
}

// NewChain returns a chain whose deciders are tried in order until one
// answers. A verdict that puts at least threshold on stronger ladder
// candidates is in doubt: with DoubtPair a decider that answers pairs is
// asked to choose between its choice and the next ladder step, and that
// answer stands; otherwise, or when the pair question fails, the decision
// moves one ladder step up. With levels, the deciders are also asked how
// much reasoning the request needs.
func NewChain(steps []Step, threshold float64, doubt string, levels []Level) *Chain {
	return &Chain{steps: steps, threshold: threshold, doubt: doubt, levels: levels, now: time.Now, health: map[string]*health{}}
}

// Rest makes a decider that failed trip turns in a row sit out for rest:
// its turns are skipped, so an outage does not delay every new thread by
// its timeout. After the rest one turn tries it again while the others
// still skip it, and one more failure rests it again.
func (c *Chain) Rest(trip int, rest time.Duration) *Chain {
	c.trip, c.rest = trip, rest
	return c
}

// Price makes the chain move a verdict along the ladder to the step with
// the least expected cost to get the task done, and settle a doubt at the
// probability where moving up pays, instead of by the threshold.
func (c *Chain) Price(economy Economy) *Chain {
	c.economy = &economy
	return c
}

// Health is the state of every decider, in chain order.
func (c *Chain) Health() []DeciderHealth {
	return c.healthFor(c.steps)
}

// Health is the shared state of the selected deciders, in chain order.
func (s *Selection) Health() []DeciderHealth {
	return s.chain.healthFor(s.steps)
}

func (c *Chain) healthFor(steps []Step) []DeciderHealth {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	list := make([]DeciderHealth, len(steps))
	for i, step := range steps {
		list[i].Name = step.Decider.Name()
		if h := c.health[list[i].Name]; h != nil {
			list[i].Failures, list[i].Probing = h.failures, h.probing
			if h.until.After(now) {
				list[i].Resting = h.until
			}
		}
	}
	return list
}

// resting reports whether a decider sits out its turn, and until when:
// during its rest, and after it while another turn probes it. The returned
// generation owns any turn admitted here.
func (c *Chain) resting(name string) (time.Time, bool, uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	h := c.health[name]
	if h == nil {
		h = &health{}
		c.health[name] = h
	}
	switch {
	case c.trip <= 0 || h.failures < c.trip:
		return time.Time{}, false, h.generation
	case h.until.After(c.now()) || h.probing:
		return h.until, true, h.generation
	}
	h.probing = true
	return time.Time{}, false, h.generation
}

// settle records a decider's turn: a failure extends its run and rests it
// once the run reaches trip; a success ends the run. Turns admitted before
// its latest rest cannot change the current generation's health.
func (c *Chain) settle(name string, generation uint64, outcome int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	h := c.health[name]
	if h == nil {
		h = &health{}
		c.health[name] = h
	}
	if h.generation != generation {
		return
	}
	h.probing = false
	switch outcome {
	case turnSucceeded:
		*h = health{generation: generation}
	case turnFailed:
		if h.failures++; c.trip > 0 && h.failures >= c.trip {
			h.until = c.now().Add(c.rest)
			h.generation++
		}
	}
}

// Decision is the outcome of a chain: the deciding verdict and every
// decider's turn on the way to it.
type Decision struct {
	Model         string
	Decider       string
	Choice        string
	Probabilities map[string]float64
	Escalated     bool
	// Level is the reasoning level the deciding verdict chose, and Levels
	// its probability per level; empty when the chooser is off or the
	// decider did not answer the reasoning question.
	Level  string
	Levels map[string]float64
	// Costs is how the economy judged the deciding verdict; empty without
	// one.
	Costs    Costs
	Latency  time.Duration
	Attempts []Attempt
}

// Failures lists the failed attempts as "decider: error".
func (d Decision) Failures() []string {
	return Failures(d.Attempts)
}

// Failures lists the failed and skipped attempts as "decider: error".
func Failures(attempts []Attempt) []string {
	var failures []string
	for _, attempt := range attempts {
		if attempt.Outcome == Failed || attempt.Outcome == Skipped {
			failures = append(failures, attempt.Decider+": "+attempt.Error)
		}
	}
	return failures
}

// Attempt outcomes.
const (
	// Decided: the verdict is the decision.
	Decided = "decided"
	// Doubted: the verdict put at least the threshold on stronger ladder
	// candidates, so the decider was asked the pair question that follows
	// it; when that fails, the verdict stands one step up.
	Doubted = "doubted"
	// Failed: the decider erred, timed out or gave no usable verdict.
	Failed = "failed"
	// Skipped: the decider was resting after failing too often in a row
	// and was not asked.
	Skipped = "skipped"
)

// Attempt is one decider's turn in a decision, kept whole so decisions can
// be audited and the policy improved later.
type Attempt struct {
	Decider       string             `json:"decider"`
	Outcome       string             `json:"outcome"`
	Model         string             `json:"model,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence,omitempty"`
	// Pair holds the two ladder steps a pair question compared, cheap
	// first.
	Pair []string `json:"pair,omitempty"`
	// Above is the probability on ladder candidates stronger than Choice;
	// for a pair question, on the stronger step, which moves the decision
	// up from Cutoff on.
	Above  float64 `json:"above,omitempty"`
	Cutoff float64 `json:"cutoff,omitempty"`
	// Blended is a pair question's probability on the stronger step with
	// the history blended in, which the cutoff was compared with; 0 when no
	// history weighed.
	Blended float64 `json:"blended,omitempty"`
	// Level is the reasoning level the decider chose, and Levels its
	// probability per level.
	Level        string             `json:"level,omitempty"`
	Levels       map[string]float64 `json:"levels,omitempty"`
	Reason       string             `json:"reason,omitempty"`
	InputTokens  int64              `json:"input_tokens,omitempty"`
	OutputTokens int64              `json:"output_tokens,omitempty"`
	UsageKnown   bool               `json:"usage_known,omitzero"`
	LatencyMS    int64              `json:"latency_ms"`
	Error        string             `json:"error,omitempty"`
	// Response is the decider's raw reply.
	Response string `json:"response,omitempty"`
	// Digest identifies the request the decider was sent, so a request
	// rebuilt from the policy can be checked against it.
	Digest string `json:"digest,omitempty"`
}

// Decide returns the decision of the first decider that answers. ok is
// false when every decider failed; the caller then applies its default.
func (c *Chain) Decide(ctx context.Context, state string, candidates []Candidate) (Decision, bool) {
	return c.decide(ctx, state, candidates, c.steps, Preferences{}, nil)
}

// Decide tries the selected deciders. ok is false when none answers or the
// selection is empty; the caller then applies its default.
func (s *Selection) Decide(ctx context.Context, state string, candidates []Candidate) (Decision, bool) {
	return s.chain.decide(ctx, state, candidates, s.steps, s.preferences, s.prior)
}

// Lighter is the ladder step just below a decision's model among the
// candidates it was decided among, and by how much the economy's score of
// that step exceeds the model's, relative to the model's score (negative
// when it scores less): how close a call keeping the model was. ok is
// false for a decision the economy did not price, a specialist, the
// lightest step, and when a step has no cost.
func (s *Selection) Lighter(candidates []Candidate, decision Decision) (string, float64, bool) {
	economy := preferenceEconomy(s.chain.economy, s.preferences.Priority)
	if economy == nil || len(decision.Costs.Done) == 0 {
		return "", 0, false
	}
	index := slices.IndexFunc(candidates, func(c Candidate) bool { return c.ID == decision.Model })
	verdict := Verdict{Choice: decision.Choice, Probabilities: decision.Probabilities}
	return economy.lighter(candidates, verdict, decision.Level, index)
}

func (c *Chain) decide(ctx context.Context, state string, candidates []Candidate, steps []Step, preferences Preferences, prior *Prior) (decision Decision, ok bool) {
	start := time.Now()
	defer func() { decision.Latency = time.Since(start) }()
	economy := preferenceEconomy(c.economy, preferences.Priority)
	for _, step := range steps {
		if ctx.Err() != nil {
			break
		}
		name := step.Decider.Name()
		until, resting, generation := c.resting(name)
		if resting {
			why := "resting until " + until.UTC().Format(time.TimeOnly) + " UTC after " + strconv.Itoa(c.trip) + " failures in a row"
			if !until.After(c.now()) {
				why = "another turn is trying it after its rest"
			}
			decision.Attempts = append(decision.Attempts, Attempt{Decider: name, Outcome: Skipped, Error: why})
			continue
		}
		// A turn the chain's own deadline cut short says nothing about the
		// decider, unless it was given its whole timeout.
		granted := true
		if deadline, ok := ctx.Deadline(); ok {
			granted = time.Until(deadline) >= step.Timeout
		}
		verdict, attempt := try(ctx, step, candidates, c.levels, func(ctx context.Context) (Verdict, error) {
			return step.Decider.Decide(ctx, state, candidates, c.levels)
		})
		decision.Attempts = append(decision.Attempts, attempt)
		switch {
		case attempt.Outcome != Failed:
			c.settle(name, generation, turnSucceeded)
		case ctx.Err() == nil || granted && errors.Is(ctx.Err(), context.DeadlineExceeded):
			c.settle(name, generation, turnFailed)
		default:
			// Canceled by whoever asked (an abandoned trial), or cut short.
			c.settle(name, generation, turnCut)
		}
		if attempt.Outcome == Failed {
			if ctx.Err() != nil {
				break
			}
			continue
		}
		// The attempt keeps the decider's own verdict; the decision goes on
		// with the prior blended in.
		level := attempt.Level
		if verdict = prior.blend(verdict, candidates, c.levels); verdict.Levels != nil {
			level, _ = ChooseLevel(c.levels, verdict.Levels)
		}
		index, escalated := Escalate(candidates, verdict, c.threshold)
		cutoff, pairable := pairCutoff, escalated
		if economy != nil {
			if cheapest, costs, priced := economy.cheapest(candidates, verdict, level); priced {
				index, escalated, decision.Costs = cheapest, candidates[cheapest].ID != verdict.Choice, costs
				cutoff = economy.stepCutoff(costs, verdict.Choice, candidates[cheapest].ID)
				// A pair question settles a doubt about one step: a verdict
				// spread over several steps goes straight to the cheapest.
				// A cheaper stronger step still settles its task fit.
				next, _ := ladderAbove(candidates, verdict, slices.IndexFunc(candidates, func(c Candidate) bool { return c.ID == verdict.Choice }))
				pairable = escalated && index == next && cutoff > 0
			}
		}
		decision.Decider, decision.Choice, decision.Probabilities = step.Decider.Name(), verdict.Choice, verdict.Probabilities
		decision.Model, decision.Escalated = candidates[index].ID, escalated
		decision.Level, decision.Levels = level, verdict.Levels
		if pairer, pairs := step.Decider.(Pairer); pairable && pairs && c.doubt == DoubtPair {
			decision.Attempts[len(decision.Attempts)-1].Outcome = Doubted
			cheap := candidates[slices.IndexFunc(candidates, func(c Candidate) bool { return c.ID == verdict.Choice })]
			strong := candidates[index]
			answer, pair := try(ctx, step, []Candidate{cheap, strong}, nil, func(ctx context.Context) (Verdict, error) {
				return pairer.Pair(ctx, state, cheap, strong)
			})
			pair.Pair, pair.Cutoff = []string{cheap.ID, strong.ID}, cutoff
			if pair.Outcome != Failed {
				// The attempt keeps the decider's answer; the history blends
				// into it as into the first verdict.
				pair.Above = answer.Probabilities[strong.ID]
				above := pair.Above
				if blended, ok := prior.pair(above, cheap.ID, strong.ID); ok {
					above, pair.Blended = blended, blended
				}
				if above+1e-9 < cutoff {
					decision.Model, decision.Escalated = cheap.ID, false
				}
			}
			// A failed pair question leaves the escalation.
			decision.Attempts = append(decision.Attempts, pair)
		}
		return decision, true
	}
	return decision, false
}

// try asks a decider one question within its step's time budget and
// records the attempt; the verdict is normalized over the candidates and
// levels asked about. A missing or unusable reasoning answer leaves the
// level empty and does not fail the attempt.
func try(ctx context.Context, step Step, candidates []Candidate, levels []Level, ask func(context.Context) (Verdict, error)) (Verdict, Attempt) {
	start := time.Now()
	stepCtx, cancel := context.WithTimeout(ctx, step.Timeout)
	var verdict Verdict
	err := stepCtx.Err()
	if err == nil {
		verdict, err = ask(stepCtx)
		if err == nil {
			err = stepCtx.Err()
		}
	}
	cancel()
	attempt := Attempt{
		Decider: step.Decider.Name(), LatencyMS: time.Since(start).Milliseconds(), Digest: verdict.Digest,
		Model: verdict.Model, Confidence: verdict.Confidence, Reason: verdict.Reason, Response: verdict.Response,
		InputTokens: verdict.InputTokens, OutputTokens: verdict.OutputTokens, UsageKnown: verdict.UsageKnown,
	}
	if err == nil {
		verdict, err = normalize(verdict, candidates)
	}
	if err != nil {
		attempt.Outcome, attempt.Error = Failed, err.Error()
		return Verdict{}, attempt
	}
	attempt.Outcome = Decided
	attempt.Model, attempt.Choice, attempt.Probabilities = verdict.Model, verdict.Choice, verdict.Probabilities
	attempt.Above = MassAbove(candidates, verdict)
	if verdict.Levels = normalizeLevels(verdict.Levels, levels); verdict.Levels != nil {
		attempt.Level, _ = ChooseLevel(levels, verdict.Levels)
		attempt.Levels = verdict.Levels
	}
	return verdict, attempt
}

// Policy is what shapes the chain's decisions over the given candidates.
func (c *Chain) Policy(candidates []Candidate) Policy {
	return c.policy(candidates, c.steps, Preferences{}, nil)
}

// Policy records the same selected deciders Decide will try, so its
// fingerprint identifies the request's enabled deciders.
func (s *Selection) Policy(candidates []Candidate) Policy {
	return s.chain.policy(candidates, s.steps, s.preferences, s.prior)
}

func (c *Chain) policy(candidates []Candidate, steps []Step, preferences Preferences, prior *Prior) Policy {
	preferences.Ranking = slices.Clone(preferences.Ranking)
	policy := Policy{Preferences: preferences, Instructions: preferences.instructions(), PairQuestion: pairQuestion, JudgeFormat: judgeFormat,
		Reasoning: reasoningQuestion(c.levels), Threshold: c.threshold, Doubt: c.doubt, Candidates: candidates,
		Learning: prior.learning()}
	if economy := preferenceEconomy(c.economy, preferences.Priority); economy != nil {
		policy.Miss, policy.TimeValue = &economy.Miss, economy.TimeValue
	}
	for _, step := range steps {
		policy.Deciders = append(policy.Deciders, step.Decider.Info())
	}
	return policy
}

// Escalate returns the index of the candidate to use: the choice, or the
// next ladder step above it when the probabilities of all stronger ladder
// candidates add up to at least threshold. A specialist choice stands, and
// doubt towards a specialist never moves the ladder.
func Escalate(candidates []Candidate, verdict Verdict, threshold float64) (int, bool) {
	index := slices.IndexFunc(candidates, func(c Candidate) bool { return c.ID == verdict.Choice })
	next, above := ladderAbove(candidates, verdict, index)
	// Tolerate float rounding: 0.07 + 0.03 must count as 0.10.
	if next >= 0 && above+1e-9 >= threshold {
		return next, true
	}
	return max(index, 0), false
}

// MassAbove is the probability the verdict puts on ladder candidates
// stronger than its choice, the quantity escalation compares with the
// threshold.
func MassAbove(candidates []Candidate, verdict Verdict) float64 {
	index := slices.IndexFunc(candidates, func(c Candidate) bool { return c.ID == verdict.Choice })
	_, above := ladderAbove(candidates, verdict, index)
	return above
}

// ladderAbove returns the next ladder step above index (-1 for none) and the
// probability on every ladder candidate above it. A specialist or unknown
// choice has no ladder above it.
func ladderAbove(candidates []Candidate, verdict Verdict, index int) (int, float64) {
	if index < 0 || candidates[index].Specialist {
		return -1, 0
	}
	next, above := -1, 0.0
	for i := index + 1; i < len(candidates); i++ {
		if candidates[i].Specialist {
			continue
		}
		if next < 0 {
			next = i
		}
		above += verdict.Probabilities[candidates[i].ID]
	}
	return next, above
}

// normalize restricts probabilities to the candidates, scales them to sum
// to one, and makes the choice the most likely candidate when the decider
// named none (or one outside the candidates).
func normalize(verdict Verdict, candidates []Candidate) (Verdict, error) {
	probabilities := map[string]float64{}
	total := 0.0
	for _, candidate := range candidates {
		p := verdict.Probabilities[candidate.ID]
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 {
			p = 0
		}
		probabilities[candidate.ID] = p
		total += p
	}
	valid := slices.ContainsFunc(candidates, func(c Candidate) bool { return c.ID == verdict.Choice })
	switch {
	case math.IsInf(total, 0):
		return Verdict{}, fmt.Errorf("probabilities add up to infinity")
	case total > 0:
		for id := range probabilities {
			probabilities[id] /= total
		}
	case valid:
		probabilities[verdict.Choice] = 1
	default:
		return Verdict{}, fmt.Errorf("no probability for any candidate")
	}
	if !valid {
		verdict.Choice = ""
		for _, candidate := range candidates {
			if verdict.Choice == "" || probabilities[candidate.ID] > probabilities[verdict.Choice] {
				verdict.Choice = candidate.ID
			}
		}
	}
	verdict.Probabilities = probabilities
	return verdict, nil
}
