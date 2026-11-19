// Package outcome judges how each settled, decided thread went, from the
// request log and its pin history: how long it was active, what it cost,
// and the signals that its decision was wrong, such as an operator moving
// it to a stronger model or a light model running a marathon. A table
// learns from the outcomes how long a task takes on each model at each
// reasoning level and how often a model sends its tasks to a stronger one,
// which the economy weighs, and ranks the unreviewed decisions most likely
// to have been misrouted.
package outcome

import (
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/usage"
)

// Signal kinds, as /stats counts them.
const (
	// RepinUp and RepinDown: an operator moved the thread to a stronger or
	// a lighter ladder step.
	RepinUp   = "repin_up"
	RepinDown = "repin_down"
	// RedecisionUp and RedecisionDown: the router decided the thread
	// again, onto a stronger or a lighter step.
	RedecisionUp   = "redecision_up"
	RedecisionDown = "redecision_down"
	// Failover: the thread's model failed it over on a server error;
	// QuotaFailover on a 429, when no account had quota left.
	Failover      = "failover"
	QuotaFailover = "quota_failover"
	// CapabilityUpgrade: the thread started carrying images its model
	// could not read.
	CapabilityUpgrade = "capability_upgrade"
	// Migration: the configuration dropped the thread's model.
	Migration = "migration"
	// RanLong: a thread on a light step ran more than Rules.Long turns.
	RanLong = "ran_long"
	// StayedTiny: a thread on a heavy step stayed within Rules.Tiny
	// turns.
	StayedTiny = "stayed_tiny"
)

// Redecision is the event kind of a thread the router decided again.
const Redecision = "redecision"

// Signals lists every signal kind, strongest evidence of a misroute first.
var Signals = []string{RepinUp, RepinDown, RedecisionUp, RedecisionDown, RanLong, Failover, StayedTiny,
	CapabilityUpgrade, QuotaFailover, Migration}

// strength is how strongly each signal says the decision was wrong: an
// operator's word most, the size of the thread least. A quota failover
// and a migration say nothing about the decision.
var strength = map[string]float64{
	RepinUp:           1,
	RepinDown:         0.9,
	RedecisionUp:      0.8,
	RedecisionDown:    0.7,
	RanLong:           0.5,
	Failover:          0.4,
	StayedTiny:        0.3,
	CapabilityUpgrade: 0.2,
}

// Rules are what derivation needs to know: the ladder, lightest first, and
// the counts of regular turns that make a light thread long and a heavy
// one tiny.
type Rules struct {
	Ladder []string
	Long   int
	Tiny   int
}

// rank is a model's place on the ladder; -1 when it is not on it.
func (r Rules) rank(model string) int {
	return slices.Index(r.Ladder, model)
}

// light reports whether a model is on the lighter half of the ladder, and
// heavy whether on the heavier half; a model in the middle of an odd
// ladder is neither.
func (r Rules) light(model string) bool {
	rank := r.rank(model)
	return rank >= 0 && rank < len(r.Ladder)/2
}

func (r Rules) heavy(model string) bool {
	return r.rank(model) >= len(r.Ladder)-len(r.Ladder)/2 && len(r.Ladder) > 1
}

// stronger reports whether to is a stronger ladder step than from; ok is
// false when either is not on the ladder.
func (r Rules) stronger(from, to string) (up, ok bool) {
	a, b := r.rank(from), r.rank(to)
	if a < 0 || b < 0 || a == b {
		return false, false
	}
	return b > a, true
}

// Signal is one sign of how a thread went, in words.
type Signal struct {
	Kind     string
	Strength float64
	Text     string
}

// Outcome is how one settled thread went.
type Outcome struct {
	Key string
	// Model is the decided model and Level the decided reasoning level;
	// Now is the model the thread ended on.
	Model, Level, Now string
	Decision          int64
	Reviewed          bool
	// Explored is set when the decision routed the thread one ladder step
	// cheaper than decided, Propensity the probability of the decision's
	// action (0 when it was not borderline enough to be explored).
	Explored   bool
	Propensity float64
	// Requests counts the thread's regular turns.
	Requests int
	// Active is the sum of the requests' latencies less their wait for a
	// concurrency slot, the wall-clock time the
	// thread kept a model busy; Span is from its first request to its
	// last, the user's time between turns included.
	Active, Span time.Duration
	Tokens       usage.Tokens
	Cost         float64
	Signals      []Signal
	// AtRisk is set when a stronger ladder step than the decided model
	// existed, Escalated when an operator or a redecision moved the thread
	// up to one stronger than the decided model.
	AtRisk    bool
	Escalated bool
	// Up and UpLevel are the model and level of the last move that
	// escalated the thread, where it went when Escalated.
	Up, UpLevel string
	// Down and DownLevel are the model and level an operator last moved
	// the thread down to, when no escalation came after.
	Down, DownLevel string
	// Moved is set when an operator or a redecision moved the thread, or
	// its model changed by failover, upgrade or migration: its time no
	// longer describes the decided model.
	Moved bool
}

// Strength is the outcome's strongest signal plus a tenth of each further
// one, so several signals outrank one of the same strength.
func (o Outcome) Strength() float64 {
	var strongest, rest float64
	for _, s := range o.Signals {
		if s.Strength > strongest {
			rest += strongest
			strongest = s.Strength
		} else {
			rest += s.Strength
		}
	}
	return strongest + rest/10
}

// Derive judges how a thread went.
func Derive(t pin.ThreadActivity, rules Rules) Outcome {
	o := Outcome{Key: t.Key, Model: t.Decided, Level: t.Level, Now: t.Model, Decision: t.Decision,
		Reviewed: t.Reviewed, Explored: t.Explored, Propensity: t.Propensity, Requests: t.Requests, Active: t.Active, Span: t.Last.Sub(t.First), Tokens: t.Tokens,
		Cost: t.Cost}
	o.AtRisk = rules.rank(t.Decided) >= 0 && rules.rank(t.Decided) < len(rules.Ladder)-1
	add := func(kind, text string) {
		o.Signals = append(o.Signals, Signal{Kind: kind, Strength: strength[kind], Text: text})
	}
	for _, e := range t.Changes {
		move := e.Previous + " → " + e.Model
		up, ranked := rules.stronger(e.Previous, e.Model)
		// A move counts as a miss only when it leaves the decision behind
		// for a stronger step: B → A → B is no miss of B.
		beyond, _ := rules.stronger(t.Decided, e.Model)
		// Any change of model splits the thread's time between models.
		if e.Model != e.Previous {
			o.Moved = true
		}
		switch e.Kind {
		case pin.Override:
			o.Moved = true
			if !ranked {
				continue
			}
			if up {
				o.escalate(e, beyond)
				add(RepinUp, "An operator moved it up, "+move)
			} else {
				o.Down, o.DownLevel = e.Model, e.Level
				add(RepinDown, "An operator moved it down, "+move)
			}
		case Redecision:
			o.Moved = true
			if !ranked {
				continue
			}
			if up {
				o.escalate(e, beyond)
				add(RedecisionUp, "The router decided it again, up, "+move)
			} else {
				add(RedecisionDown, "The router decided it again, down, "+move)
			}
		case pin.Failover:
			if quota(e.Detail) {
				add(QuotaFailover, "It failed over when no account had quota left, "+move)
				continue
			}
			// A failover tells of the upstream's health, not of the task:
			// it queues the decision for review but is no miss.
			add(Failover, "It failed over on "+cleanDetail(e.Detail)+", "+move)
		case pin.CapabilityUpgrade:
			add(CapabilityUpgrade, "It started carrying images and moved, "+move)
		case pin.Migration:
			add(Migration, "It migrated, "+move+": "+e.Detail)
		}
	}
	// A thread's size describes the decided model only while it stayed on
	// it: a light thread that failed over and ran long elsewhere did not
	// run long on the light model.
	if o.Moved {
		return o
	}
	if rules.Long > 0 && t.Requests > rules.Long && rules.light(t.Decided) {
		add(RanLong, "It ran "+strconv.Itoa(t.Requests)+" turns on the light "+t.Decided)
	}
	if rules.Tiny > 0 && t.Requests <= rules.Tiny && rules.heavy(t.Decided) {
		add(StayedTiny, "It stayed at "+strconv.Itoa(t.Requests)+" turns on the heavy "+t.Decided)
	}
	return o
}

// escalate records a move up to a stronger step, which undoes any move
// down before it; beyond is set when the step is stronger than the
// decided model.
func (o *Outcome) escalate(e pin.Event, beyond bool) {
	o.Down, o.DownLevel = "", ""
	if beyond {
		o.Escalated, o.Up, o.UpLevel = true, e.Model, e.Level
	}
}

// Has reports whether the outcome shows a signal of the kind.
func (o Outcome) Has(kind string) bool {
	return slices.ContainsFunc(o.Signals, func(s Signal) bool { return s.Kind == kind })
}

// quota reports whether a failover's detail names a 429, which only says
// that no account had quota left.
func quota(detail string) bool {
	return strings.HasSuffix(detail, " 429")
}

// cleanDetail is a failover's detail as part of a sentence.
func cleanDetail(detail string) string {
	if detail == "" {
		return "an error"
	}
	return detail
}
