package outcome

import (
	"cmp"
	"context"
	"log/slog"
	"slices"
	"sync"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
)

// Source reports how the settled, decided threads went; *pin.Store is one.
// A table without one learns nothing and changes no decision.
type Source interface {
	ThreadActivities(ctx context.Context, since, settled time.Time) ([]pin.ThreadActivity, error)
}

// Typical is the wall-clock time a task is assumed to keep its model busy
// before the log has any: about fifty turns of some twenty seconds.
const Typical = 15 * time.Minute

// limit is the most time one thread counts with, as for the cost per task.
const limit = 20 * Typical

// settle is how long after its last request a thread counts as done, as
// for the cost per task.
const settle = time.Hour

// evidence is how many settled threads with a stronger step above them a
// cell needs before its miss rate moves the miss penalty.
const evidence = 5

// The learned miss rate scales the penalty at most this much either way.
const (
	minFactor = 0.25
	maxFactor = 4
)

// Table learns from thread outcomes.
type Table struct {
	source Source
	rules  Rules
	// window is how far back threads teach the table; weight is how many
	// threads an estimate's prior counts as.
	window time.Duration
	weight float64
	now    func() time.Time

	mu       sync.RWMutex
	outcomes []Outcome
	// reviews counts the reviews taken in; reviewed holds the decisions
	// reviewed and the count when each was, so a refresh that read the log
	// before a review was saved does not put its decision back in the
	// queue.
	reviews  int
	reviewed map[int64]int
	cells    map[cell]tally
	level    map[string]tally
	all      tally
	learned  time.Time
}

// cell is one model at one reasoning level.
type cell struct {
	model, level string
}

// tally sums the outcomes of a cell, a level or the whole log: the threads
// timed and their active hours, and the threads that had a stronger step
// above them and how many of those moved up to one.
type tally struct {
	timed  int
	hours  float64
	atRisk int
	missed int
}

// New returns a table that learns from source.
func New(source Source, rules Rules, window time.Duration, weight float64) *Table {
	return &Table{source: source, rules: rules, window: window, weight: weight, now: time.Now,
		cells: map[cell]tally{}, level: map[string]tally{}, reviewed: map[int64]int{}}
}

// Refresh reads the settled threads of the window and judges them.
func (t *Table) Refresh(ctx context.Context) error {
	if t.source == nil {
		return nil
	}
	now := t.now()
	t.mu.RLock()
	started := t.reviews
	t.mu.RUnlock()
	threads, err := t.source.ThreadActivities(ctx, now.Add(-t.window), now.Add(-settle))
	if err != nil {
		return err
	}
	outcomes := make([]Outcome, 0, len(threads))
	cells, levels, all := map[cell]tally{}, map[string]tally{}, tally{}
	for _, thread := range threads {
		o := Derive(thread, t.rules)
		outcomes = append(outcomes, o)
		var add tally
		// A moved thread's time went partly to another model.
		if !o.Moved {
			add.timed, add.hours = 1, min(o.Active, limit).Hours()
		}
		if o.AtRisk {
			add.atRisk = 1
			if o.Escalated {
				add.missed = 1
			}
		}
		cells[cell{o.Model, o.Level}] = cells[cell{o.Model, o.Level}].plus(add)
		levels[o.Level] = levels[o.Level].plus(add)
		all = all.plus(add)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	// Reviews taken in while the log was read may be missing from it; those
	// from before were read and need no keeping.
	for decision, at := range t.reviewed {
		if at <= started {
			delete(t.reviewed, decision)
			continue
		}
		for i := range outcomes {
			if outcomes[i].Decision == decision {
				outcomes[i].Reviewed = true
			}
		}
	}
	t.outcomes, t.cells, t.level, t.all, t.learned = outcomes, cells, levels, all, now
	return nil
}

// Run refreshes the table every interval until ctx ends; after each
// refresh that succeeded it runs then, which learn from the fresh
// outcomes.
func (t *Table) Run(ctx context.Context, interval time.Duration, log *slog.Logger, then ...func(context.Context) error) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		if err := t.Refresh(ctx); err != nil && ctx.Err() == nil {
			log.Warn("thread outcome refresh failed", "error", err)
		} else if err == nil {
			for _, next := range then {
				if err := next(ctx); err != nil && ctx.Err() == nil {
					log.Warn("learning from thread outcomes failed", "error", err)
				}
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// Estimate is what the table learned about a task on a model at a level.
type Estimate struct {
	Model string `json:"model"`
	Level string `json:"level,omitempty"`
	// Hours is the expected wall-clock time per task, learned from Timed
	// threads of the cell.
	Hours float64 `json:"hours"`
	Timed int     `json:"timed"`
	// MissRate is the share of the cell's threads an operator or a
	// redecision moved up to a stronger step, shrunk towards the fleet's; Missed of AtRisk threads
	// did. Factor is how it scales the miss penalty: the rate against the
	// fleet's, 1 until the cell has enough evidence.
	MissRate float64 `json:"miss_rate"`
	Missed   int     `json:"missed"`
	AtRisk   int     `json:"at_risk"`
	Factor   float64 `json:"factor"`
}

// Estimate is what the table expects of a task on model at level.
func (t *Table) Estimate(model, level string) Estimate {
	t.mu.RLock()
	defer t.mu.RUnlock()
	seen := t.cells[cell{model, level}]
	e := Estimate{Model: model, Level: level, Hours: t.hours(model, level), Timed: seen.timed,
		Missed: seen.missed, AtRisk: seen.atRisk, Factor: 1}
	e.MissRate, e.Factor = t.miss(seen)
	return e
}

// Estimates are the estimates of every model at every level, in the order
// given.
func (t *Table) Estimates(models, levels []string) []Estimate {
	if len(levels) == 0 {
		levels = []string{""}
	}
	var list []Estimate
	for _, model := range models {
		for _, level := range levels {
			list = append(list, t.Estimate(model, level))
		}
	}
	return list
}

// hours is the expected active time of a task on model at level: the
// time the level's tasks take times how slow the model is.
func (t *Table) hours(model, level string) float64 {
	return t.size(level) * t.slowness(model)
}

// size is the mean active hours of a level's tasks, shrunk towards every
// thread's mean, and that towards the typical task.
func (t *Table) size(level string) float64 {
	return t.level[level].shrink(t.all.shrink(Typical.Hours(), t.weight), t.weight)
}

// slowness is how long a model's threads took against what the same
// numbers of threads of the same levels took on every model, shrunk
// towards 1 as if weight more threads had taken just as long.
func (t *Table) slowness(model string) float64 {
	threads, used, expected := 0, 0.0, 0.0
	for c, seen := range t.cells {
		if c.model == model {
			threads += seen.timed
			used += seen.hours
			expected += float64(seen.timed) * t.size(c.level)
		}
	}
	if threads == 0 || expected <= 0 {
		return 1
	}
	prior := t.weight * expected / float64(threads)
	return (used + prior) / (expected + prior)
}

// miss is a cell's miss rate, shrunk towards the fleet's as if weight more
// threads had missed at that rate, and the factor it scales the miss
// penalty by: 1 until the cell has enough evidence and the fleet has missed
// at all.
func (t *Table) miss(seen tally) (rate, factor float64) {
	fleet := 0.0
	if t.all.atRisk > 0 {
		fleet = float64(t.all.missed) / float64(t.all.atRisk)
	}
	rate = fleet
	if n := float64(seen.atRisk) + t.weight; n > 0 {
		rate = (float64(seen.missed) + t.weight*fleet) / n
	}
	if seen.atRisk < evidence || fleet <= 0 {
		return rate, 1
	}
	return rate, min(max(rate/fleet, minFactor), maxFactor)
}

// Queue is up to limit unreviewed decisions of settled threads with a
// sign of a misroute, strongest first, then newest.
func (t *Table) Queue(limit int) []Outcome {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var queue []Outcome
	for _, o := range t.outcomes {
		if !o.Reviewed && o.Strength() > 0 {
			queue = append(queue, o)
		}
	}
	slices.SortStableFunc(queue, func(a, b Outcome) int {
		return cmp.Or(cmp.Compare(b.Strength(), a.Strength()), cmp.Compare(b.Decision, a.Decision))
	})
	return queue[:min(limit, len(queue))]
}

// Outcomes are the outcomes of the settled threads the latest refresh
// judged.
func (t *Table) Outcomes() []Outcome {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return slices.Clone(t.outcomes)
}

// Reviewed takes a decision an operator just judged off the queue, before
// the next refresh would.
func (t *Table) Reviewed(decision int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.reviews++
	t.reviewed[decision] = t.reviews
	for i := range t.outcomes {
		if t.outcomes[i].Decision == decision {
			t.outcomes[i].Reviewed = true
		}
	}
}

// Counts are how many settled threads the table judged, how many of those
// with a stronger step above them moved up to one, how many decisions with
// a sign of a misroute wait for a review, and how many threads showed
// each signal.
type Counts struct {
	Threads   int            `json:"threads"`
	AtRisk    int            `json:"at_risk"`
	Escalated int            `json:"escalated"`
	Queued    int            `json:"queued"`
	Signals   map[string]int `json:"signals"`
	Learned   time.Time      `json:"learned"`
}

// Counts tallies the signals of the settled threads.
func (t *Table) Counts() Counts {
	t.mu.RLock()
	defer t.mu.RUnlock()
	counts := Counts{Threads: len(t.outcomes), AtRisk: t.all.atRisk, Escalated: t.all.missed,
		Signals: map[string]int{}, Learned: t.learned}
	for _, kind := range Signals {
		counts.Signals[kind] = 0
	}
	for _, o := range t.outcomes {
		seen := map[string]bool{}
		for _, s := range o.Signals {
			if !seen[s.Kind] {
				seen[s.Kind] = true
				counts.Signals[s.Kind]++
			}
		}
		if !o.Reviewed && o.Strength() > 0 {
			counts.Queued++
		}
	}
	return counts
}

func (a tally) plus(b tally) tally {
	return tally{timed: a.timed + b.timed, hours: a.hours + b.hours, atRisk: a.atRisk + b.atRisk,
		missed: a.missed + b.missed}
}

// shrink is the tally's mean hours per task, pulled towards prior as if
// weight more threads had taken it.
func (a tally) shrink(prior, weight float64) float64 {
	n := float64(a.timed) + weight
	if n <= 0 {
		return prior
	}
	return (a.hours + weight*prior) / n
}
