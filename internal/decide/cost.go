package decide

import (
	"math"
	"slices"
)

// Economy ranks supported ladder tiers with a cost-weighted policy
// heuristic instead of a fixed escalation threshold. It adds the cost of
// each stronger tier, weighted by the verdict and Miss, to a candidate's
// own task cost. The probabilities describe uncertainty over suitable
// tiers; this penalty approximates the cost of choosing too low a tier.
// Once settled threads show how often a tier sent its tasks to a stronger
// one, that rate against the fleet's scales the tier's Miss, and with
// TimeValue set every tier also pays for the wall-clock time its tasks
// take.
type Economy struct {
	// Task is a candidate's expected cost per task at a reasoning level;
	// offered contains the available candidates, so pricing can account
	// for eligible spills. ok is false when the candidate has no cost.
	Task func(model, level string, offered []Candidate) (cost float64, ok bool)
	// Miss is what a miss costs on top of redoing the task, as a share of
	// the redo's cost.
	Miss float64
	// Learn is what the outcomes of settled threads taught about a
	// candidate at a reasoning level; nil learns nothing.
	Learn func(model, level string, offered []Candidate) Learned
	// TimeValue is what an hour of a task's wall-clock time is worth, in
	// US dollars; 0 leaves time out.
	TimeValue float64
}

// Learned is what the outcomes of settled threads taught about a candidate
// at a reasoning level: the wall-clock hours a task takes, the share of
// its threads an operator or a redecision moved up to a stronger step
// (MissRate), and how that
// scales the miss penalty against the fleet's (Factor, 1 without enough
// evidence).
type Learned struct {
	Hours    float64
	MissRate float64
	Factor   float64
}

// Costs is how an economy judged a verdict: the cost per task of every
// ladder candidate from the choice up, and each one's cost-weighted policy
// score (Done). With time valued, Hours is each step's expected wall-clock
// time per task and Time what it is worth, counted in Done; where learned
// miss rates moved a step's miss penalty, MissRate is the rate and Miss
// the penalty the step's score counted.
type Costs struct {
	Task     map[string]float64 `json:"task,omitempty"`
	Done     map[string]float64 `json:"done,omitempty"`
	Hours    map[string]float64 `json:"hours,omitempty"`
	Time     map[string]float64 `json:"time,omitempty"`
	MissRate map[string]float64 `json:"miss_rate,omitempty"`
	Miss     map[string]float64 `json:"miss,omitempty"`
}

// price is a step's cost per task with its time.
func (c Costs) price(id string) float64 {
	return c.Task[id] + c.Time[id]
}

// cheapest returns the index of the ladder candidate from the verdict's
// choice up with the least expected cost to get the task done, the lighter
// on a tie, and the costs it compared. ok is false for a specialist or
// unknown choice and when a candidate has no cost, where the threshold
// decides.
func (e *Economy) cheapest(candidates []Candidate, verdict Verdict, level string) (int, Costs, bool) {
	choice := slices.IndexFunc(candidates, func(c Candidate) bool { return c.ID == verdict.Choice })
	if choice < 0 || candidates[choice].Specialist {
		return choice, Costs{}, false
	}
	ladder, costs, ok := e.ladder(candidates, verdict, level, choice)
	if !ok {
		return choice, Costs{}, false
	}
	best := choice
	for _, i := range ladder {
		id := candidates[i].ID
		// A cheaper tier with no policy fit cannot replace the choice.
		if i != choice && verdict.Probabilities[id] == 0 {
			continue
		}
		// A free or discounted stronger tier still needs majority task fit.
		// Otherwise a tiny probability would send almost every simple task
		// to a scarce free model and consume its concurrency slots. A tier
		// cheaper only for its time is not one of those.
		if i != choice && costs.Task[id] <= costs.Task[verdict.Choice] &&
			verdict.Probabilities[id] < pairCutoff {
			continue
		}
		// Tolerate float rounding: an even trade keeps the lighter step.
		if costs.Done[id] < costs.Done[candidates[best].ID]-1e-12 {
			best = i
		}
	}
	return best, costs, true
}

// ladder prices the ladder candidates from index from up and scores each
// one: its own cost per task, plus, for every stronger step the verdict
// puts probability on, that probability times the stronger step's cost
// times one plus its miss penalty. A step's score depends only on itself
// and the steps above it. ok is false when a candidate has no cost.
func (e *Economy) ladder(candidates []Candidate, verdict Verdict, level string, from int) ([]int, Costs, bool) {
	var ladder []int
	costs := Costs{Task: map[string]float64{}, Done: map[string]float64{}}
	miss := map[string]float64{}
	for i := from; i < len(candidates); i++ {
		if candidates[i].Specialist {
			continue
		}
		id := candidates[i].ID
		cost, ok := e.Task(id, level, candidates)
		if !ok {
			return nil, Costs{}, false
		}
		costs.Task[id] = cost
		miss[id] = e.Miss
		if e.Learn != nil {
			learned := e.Learn(id, level, candidates)
			if e.TimeValue > 0 {
				costs.set(&costs.Hours, id, learned.Hours)
				costs.set(&costs.Time, id, learned.Hours*e.TimeValue)
			}
			if learned.Factor > 0 && learned.Factor != 1 {
				miss[id] = e.Miss * learned.Factor
				costs.set(&costs.MissRate, id, learned.MissRate)
				costs.set(&costs.Miss, id, miss[id])
			}
		}
		ladder = append(ladder, i)
	}
	for n, i := range ladder {
		id := candidates[i].ID
		done := costs.price(id)
		for _, j := range ladder[n+1:] {
			done += verdict.Probabilities[candidates[j].ID] * costs.price(candidates[j].ID) * (1 + miss[id])
		}
		costs.Done[id] = done
	}
	return ladder, costs, true
}

// lighter is the ladder step just below the step at index among the
// candidates, and by how much the economy's score of it exceeds the
// step's, relative to the step's score: negative when the lighter step
// scores less, as one below the verdict's choice can, since the economy
// never moves a verdict down. ok is false for a specialist, the lightest
// step, and when a step has no cost.
func (e *Economy) lighter(candidates []Candidate, verdict Verdict, level string, index int) (string, float64, bool) {
	if index < 0 || candidates[index].Specialist {
		return "", 0, false
	}
	below := -1
	for i := index - 1; i >= 0 && below < 0; i-- {
		if !candidates[i].Specialist {
			below = i
		}
	}
	if below < 0 {
		return "", 0, false
	}
	_, costs, ok := e.ladder(candidates, verdict, level, below)
	if !ok {
		return "", 0, false
	}
	light, step := costs.Done[candidates[below].ID], costs.Done[candidates[index].ID]
	switch {
	case step > 0:
		return candidates[below].ID, (light - step) / step, true
	case light > step:
		return candidates[below].ID, math.Inf(1), true
	}
	return candidates[below].ID, 0, true
}

// set records a step's value in a map it makes on first use.
func (c *Costs) set(values *map[string]float64, id string, value float64) {
	if *values == nil {
		*values = map[string]float64{}
	}
	(*values)[id] = value
}

// cutoff is the probability on the stronger of two steps at which moving
// up pays: the extra cost of the stronger step against what a miss on the
// cheaper one would cost. A cheaper or free stronger step still
// needs majority task fit.
func (e *Economy) cutoff(cheap, strong float64) float64 {
	return cutoffAt(cheap, strong, e.Miss)
}

// stepCutoff is the cutoff between two steps of a judged verdict, with
// their time and the cheaper step's learned miss penalty, judged like
// cheapest judges them: a stronger step free or discounted in money still
// needs majority task fit, whatever its time; one dearer in money but
// cheaper with its time needs no pair question at all (a cutoff of 0).
func (e *Economy) stepCutoff(costs Costs, cheap, strong string) float64 {
	miss := e.Miss
	if m, ok := costs.Miss[cheap]; ok {
		miss = m
	}
	cutoff := cutoffAt(costs.price(cheap), costs.price(strong), miss)
	switch {
	case costs.Task[strong] <= costs.Task[cheap]:
		return max(cutoff, pairCutoff)
	case costs.price(strong) <= costs.price(cheap):
		return 0
	}
	return cutoff
}

func cutoffAt(cheap, strong, miss float64) float64 {
	if strong <= cheap || strong <= 0 {
		return pairCutoff
	}
	return (strong - cheap) / ((1 + miss) * strong)
}
