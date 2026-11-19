package learn

import (
	"math"
	"slices"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
)

// smoothing is the label weight every offered candidate and level starts
// with, as if each had one more thread that sufficed on it: additive
// smoothing, so a few neighbours never rule anything out.
const smoothing = SufficedWeight

// shrink is how many effective neighbours a prior needs to weigh half its
// fitted weight: w = fit * n/(n + shrink).
const shrink = 5.0

// neighbour is a labelled decision a search found, with how similar its
// first prompt is to the one searched for, the best match being 1.
type neighbour struct {
	Similarity float64
	Label      Label
}

// build turns the neighbours into a prior over the offered candidates and
// levels, judged at the time at: each neighbour adds its label weight
// times its similarity times its recency, which halves every halfLife, to
// the model and the level of its label, when offered; every offered one
// starts at smoothing. The effective neighbour count is the sum of
// similarity times recency over the neighbours that bore on each. The
// weights are the fitted ones shrunk by those counts.
func build(neighbours []neighbour, candidates, levels []string, at time.Time, halfLife time.Duration, fit, levelFit float64) decide.Prior {
	prior := decide.Prior{Fit: fit, LevelFit: levelFit}
	models, modelCount := distribution(neighbours, candidates, at, halfLife, func(l Label) string { return l.Model })
	if modelCount > 0 {
		prior.Models, prior.Neighbours = models, modelCount
		prior.Weight = weight(fit, modelCount)
	}
	levelDist, levelCount := distribution(neighbours, levels, at, halfLife, func(l Label) string { return l.Level })
	if levelCount > 0 {
		prior.Levels, prior.LevelNeighbours = levelDist, levelCount
		prior.LevelWeight = weight(levelFit, levelCount)
	}
	for _, n := range neighbours {
		prior.Similar = append(prior.Similar, decide.Precedent{Event: n.Label.Decision, Similarity: round(n.Similarity),
			Model: n.Label.Model, Level: n.Label.Level, Weight: round(n.Label.Weight)})
	}
	return prior
}

// distribution is the smoothed distribution of the neighbours' labels over
// the offered ids, and the effective count behind it; nil when no
// neighbour bore on it.
func distribution(neighbours []neighbour, ids []string, at time.Time, halfLife time.Duration, of func(Label) string) (map[string]float64, float64) {
	if len(ids) == 0 {
		return nil, 0
	}
	mass := make(map[string]float64, len(ids))
	for _, id := range ids {
		mass[id] = smoothing
	}
	count := 0.0
	for _, n := range neighbours {
		id := of(n.Label)
		if !slices.Contains(ids, id) {
			continue
		}
		strength := n.Similarity * recency(at.Sub(n.Label.At), halfLife)
		mass[id] += n.Label.Weight * strength
		count += strength
	}
	if count == 0 {
		return nil, 0
	}
	total := 0.0
	for _, m := range mass {
		total += m
	}
	for id := range mass {
		mass[id] /= total
	}
	return mass, count
}

// recency is how much a label counts after age: half every halfLife,
// whichever way the age runs.
func recency(age, halfLife time.Duration) float64 {
	if halfLife <= 0 {
		return 1
	}
	return math.Exp2(-math.Abs(age.Hours()) / halfLife.Hours())
}

// weight is a fitted weight shrunk by the effective neighbours behind a
// prior.
func weight(fit, neighbours float64) float64 {
	return fit * neighbours / (neighbours + shrink)
}

// round keeps three decimals of a recorded number.
func round(x float64) float64 {
	return math.Round(x*1000) / 1000
}

// Fitting.
const (
	// minLabels is how many replayable labels a fit needs; below it the
	// weight stays zero.
	minLabels = 20
	// maxFit is the most the history may weigh against the deciders.
	maxFit = 0.8
	// fitStep is the grid the weight is searched on, which also keeps the
	// fingerprints of the policies it enters few.
	fitStep = 0.05
	// floor keeps a log loss finite when a blend gives a label no
	// probability at all.
	floor = 1e-3
)

// replay is a labelled decision as a fit replays it: what the deciders
// said, what the prior built without it said, how many neighbours stood
// behind that, and the label with its weight.
type replay struct {
	verdict    map[string]float64
	prior      map[string]float64
	neighbours float64
	label      string
	weight     float64
}

// fitWeight is the weight in [0, maxFit], on the fitStep grid, at which
// blending the prior into the recorded verdicts would have given the
// labels the least weighted log loss; the lighter on a tie. 0 below
// minLabels replays.
func fitWeight(replays []replay) float64 {
	if len(replays) < minLabels {
		return 0
	}
	best, bestLoss := 0.0, math.Inf(1)
	for step := 0; float64(step)*fitStep <= maxFit+1e-9; step++ {
		w := float64(step) * fitStep
		loss := 0.0
		for _, r := range replays {
			loss += r.weight * -math.Log(max(blended(r, w), floor))
		}
		if loss < bestLoss-1e-9 {
			best, bestLoss = w, loss
		}
	}
	return round(best)
}

// blended is the probability a blend at fitted weight w gives a replay's
// label, normalized over the verdict's and the prior's ids.
func blended(r replay, w float64) float64 {
	if r.prior == nil {
		w = 0
	}
	w = weight(w, r.neighbours)
	total := 0.0
	for id, p := range r.verdict {
		total += (1-w)*p + w*r.prior[id]
	}
	for id, p := range r.prior {
		if _, ok := r.verdict[id]; !ok {
			total += w * p
		}
	}
	if total <= 0 {
		return 0
	}
	return ((1-w)*r.verdict[r.label] + w*r.prior[r.label]) / total
}
