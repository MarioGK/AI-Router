// Package learn makes routing improve itself from its own history. Every
// settled, decided thread leaves an implicit label: the model and
// reasoning level its outcome says the task belonged on, weighted by how
// much that outcome says. At decision time the labelled decisions whose
// first prompts read most like the new one form a prior over the offered
// candidates and levels, and the decision blends it with the deciders'
// verdict, by a weight the history earns: on every refresh each labelled
// decision is replayed from its recorded verdict with the prior built
// without it, and the weight that would have predicted the labels best
// is kept. Below enough labels the weight is zero and decisions are
// exactly what the deciders and the economy make of them.
package learn

import (
	"slices"
	"time"

	"github.com/MarioGK/AI-Router/internal/outcome"
	"github.com/MarioGK/AI-Router/internal/pin"
)

// Label weights: how much each kind of outcome says about where a task
// belonged.
const (
	// ReviewedWeight is an operator's review, which names the model and
	// level the task should have had: the strongest evidence there is.
	ReviewedWeight = 1.0
	// RepinDownWeight is an operator moving the thread down to a lighter
	// step: the operator's word that the task needed less, almost as
	// strong as a review.
	RepinDownWeight = 0.9
	// EscalatedWeight is a thread an operator or a redecision moved up to
	// a stronger step: the step it went to is likely where it belonged,
	// though a move can overshoot.
	EscalatedWeight = 0.7
	// SufficedWeight is a thread that stayed where it was decided and was
	// neither long for a light model nor tiny for a heavy one: the decided
	// model sufficed. It is weak because only the chosen model was tried,
	// and a lighter one might have sufficed too.
	SufficedWeight = 0.3
	// LighterWeight is a thread on a heavy step that stayed tiny: one step
	// lighter would probably have done. Size is a poor proxy for
	// difficulty, so it counts least.
	LighterWeight = 0.15
	// maxExplore bounds how much an exploratory decision's label is
	// scaled up for having been unlikely to be made.
	maxExplore = 5
)

// Label kinds.
const (
	Reviewed  = "reviewed"
	RepinDown = "repinned_down"
	Escalated = "escalated"
	Sufficed  = "sufficed"
	Lighter   = "lighter"
)

// Kinds lists the label kinds, strongest first.
var Kinds = []string{Reviewed, RepinDown, Escalated, Sufficed, Lighter}

// Label is where a settled thread's outcome says its task belonged: a
// model and, when the outcome says so, a reasoning level, with how much
// it counts. A label without a level teaches only the model.
type Label struct {
	Decision int64
	Key      string
	At       time.Time
	Kind     string
	Model    string
	Level    string
	Weight   float64
}

// Derive labels a settled thread from its outcome, the operator's review
// of its decision (nil without one) and the ladder, lightest first. ok is false when the outcome teaches
// nothing: a failover, an upgrade or a migration moved it, a redecision
// moved it down, it ran long on a light model, or a heavy thread stayed tiny
// on the lightest step there is.
func Derive(o outcome.Outcome, review *pin.Review, ladder []string) (Label, bool) {
	label := Label{Decision: o.Decision, Key: o.Key}
	switch {
	case review != nil:
		label.Kind, label.Model, label.Level, label.Weight = Reviewed, review.Model, review.Level, ReviewedWeight
	case o.Down != "":
		label.Kind, label.Model, label.Level, label.Weight = RepinDown, o.Down, o.DownLevel, RepinDownWeight
	case o.Escalated:
		label.Kind, label.Model, label.Level, label.Weight = Escalated, o.Up, o.UpLevel, EscalatedWeight
	case o.Moved || o.Has(outcome.RanLong):
		return Label{}, false
	case o.Has(outcome.StayedTiny):
		rank := slices.Index(ladder, o.Model)
		if rank < 1 {
			return Label{}, false
		}
		// Size says nothing about the reasoning the task needed.
		label.Kind, label.Model, label.Weight = Lighter, ladder[rank-1], LighterWeight
	default:
		label.Kind, label.Model, label.Level, label.Weight = Sufficed, o.Model, o.Level, SufficedWeight
	}
	if label.Model == "" {
		return Label{}, false
	}
	label.Weight *= explored(o)
	return label, true
}

// explored is how much an exploratory decision's label is scaled up:
// one over the probability it had of being made, at most maxExplore, so
// the rare explorations of a step stand for the decisions that did not
// explore it. 1 for a decision that did not explore.
func explored(o outcome.Outcome) float64 {
	if !o.Explored || !(o.Propensity > 0) {
		return 1
	}
	return min(1/o.Propensity, maxExplore)
}
