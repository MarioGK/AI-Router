// Package costs learns what a task costs on each model at each reasoning
// level. A task's size is the tokens the settled threads of its level used
// (shrunk towards every thread's, and those towards a typical coding task);
// a model's verbosity is how many tokens its threads used against what
// threads of the same levels used on every model (shrunk towards 1); their
// product is priced at the model's list price times its cost factor.
// Levels measure how hard a task is, so a model that gets the hard ones is
// not taken for a wasteful one, and counting tokens rather than dollars
// keeps the estimates right when a price changes.
package costs

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/usage"
)

// Source reports the usage of settled threads, each cut down to at most
// limit tokens; *pin.Store is one. A table without one prices every task
// as a typical one.
type Source interface {
	TaskUsages(ctx context.Context, since, settled time.Time, limit int64) ([]pin.TaskUsage, error)
}

// Typical is the task assumed before the log has any: an agentic coding
// session of about fifty turns over a mostly cached context.
var Typical = usage.Tokens{Input: 200_000, CacheRead: 2_000_000, Output: 40_000}

// limit is the most tokens one thread counts with: a marathon session of
// twenty typical tasks must not set a model's cost for a whole window.
var limit = 20 * Typical.Total()

// settle is how long after its last request a thread counts as done.
const settle = time.Hour

// Table estimates the cost per task of each model at each reasoning level.
type Table struct {
	source Source
	price  func(model string) (usage.Price, bool)
	factor func(model string) float64
	// window is how far back threads teach the table; weight is how many
	// threads an estimate's prior counts as.
	window time.Duration
	weight float64
	now    func() time.Time

	mu    sync.RWMutex
	cells map[cell]tally
	level map[string]tally
	all   tally
	// learned is when the table last read the log.
	learned time.Time
}

// cell is one model at one reasoning level.
type cell struct {
	model, level string
}

// tally sums the threads of a cell, a level or the whole log.
type tally struct {
	threads int
	tokens  usage.Tokens
	cost    float64
}

// New returns a table that learns from source and prices tasks with
// price times factor.
func New(source Source, price func(model string) (usage.Price, bool), factor func(model string) float64,
	window time.Duration, weight float64) *Table {
	return &Table{source: source, price: price, factor: factor, window: window, weight: weight, now: time.Now,
		cells: map[cell]tally{}, level: map[string]tally{}}
}

// Refresh reads the settled threads of the window.
func (t *Table) Refresh(ctx context.Context) error {
	if t.source == nil {
		return nil
	}
	now := t.now()
	usages, err := t.source.TaskUsages(ctx, now.Add(-t.window), now.Add(-settle), limit)
	if err != nil {
		return err
	}
	cells, levels, all := map[cell]tally{}, map[string]tally{}, tally{}
	for _, u := range usages {
		add := tally{threads: u.Threads, tokens: u.Tokens, cost: u.Cost}
		cells[cell{u.Model, u.Level}] = cells[cell{u.Model, u.Level}].plus(add)
		levels[u.Level] = levels[u.Level].plus(add)
		all = all.plus(add)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.cells, t.level, t.all, t.learned = cells, levels, all, now
	return nil
}

// Run refreshes the table every interval until ctx ends.
func (t *Table) Run(ctx context.Context, interval time.Duration, log *slog.Logger) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		if err := t.Refresh(ctx); err != nil && ctx.Err() == nil {
			log.Warn("task cost refresh failed", "error", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// Task is the expected cost of a task on model at level, in US dollars;
// ok is false for a model without a price.
func (t *Table) Task(model, level string) (float64, bool) {
	price, ok := t.price(model)
	if !ok {
		return 0, false
	}
	t.mu.RLock()
	tokens := t.tokens(model, level)
	t.mu.RUnlock()
	return price.Cost(tokens) * t.factor(model), true
}

// tokens is the expected usage of a task on model at level: the size of
// the level's tasks times the model's verbosity.
func (t *Table) tokens(model, level string) usage.Tokens {
	return scale(t.size(level), t.verbosity(model))
}

// size is the mean task of a level, shrunk towards every thread's mean,
// and that towards the typical task.
func (t *Table) size(level string) usage.Tokens {
	return t.level[level].shrink(t.all.shrink(Typical, t.weight), t.weight)
}

// verbosity is how many tokens a model's threads used against what the
// same numbers of threads of the same levels used on every model, shrunk
// towards 1 as if weight more threads had used just as many.
func (t *Table) verbosity(model string) float64 {
	threads, used, expected := 0, 0.0, 0.0
	for c, seen := range t.cells {
		if c.model == model {
			threads += seen.threads
			used += float64(seen.tokens.Total())
			expected += float64(seen.threads) * float64(t.size(c.level).Total())
		}
	}
	if threads == 0 || expected <= 0 {
		return 1
	}
	prior := t.weight * expected / float64(threads)
	return (used + prior) / (expected + prior)
}

// Estimate is what the table expects a task on a model at a level to cost,
// and what the threads it learned from cost, both at list price times the
// model's cost factor.
type Estimate struct {
	Model string
	Level string
	// Cost is the expected cost per task; Threads the settled threads it
	// learned from, which cost Observed per task on average.
	Cost     float64
	Threads  int
	Observed float64
}

// Estimates are the expected costs of every model at every level, in the
// order given.
func (t *Table) Estimates(models, levels []string) []Estimate {
	if len(levels) == 0 {
		levels = []string{""}
	}
	var list []Estimate
	for _, model := range models {
		for _, level := range levels {
			cost, ok := t.Task(model, level)
			if !ok {
				continue
			}
			t.mu.RLock()
			seen := t.cells[cell{model, level}]
			t.mu.RUnlock()
			estimate := Estimate{Model: model, Level: level, Cost: cost, Threads: seen.threads}
			if seen.threads > 0 {
				estimate.Observed = seen.cost / float64(seen.threads) * t.factor(model)
			}
			list = append(list, estimate)
		}
	}
	return list
}

// Learned is how many settled threads the table learned from, and when.
func (t *Table) Learned() (int, time.Time) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.all.threads, t.learned
}

// scale multiplies every count of a usage by factor.
func scale(tokens usage.Tokens, factor float64) usage.Tokens {
	times := func(n int64) int64 { return int64(float64(n) * factor) }
	return usage.Tokens{Input: times(tokens.Input), Output: times(tokens.Output),
		CacheRead: times(tokens.CacheRead), CacheWrite: times(tokens.CacheWrite)}
}

func (a tally) plus(b tally) tally {
	return tally{threads: a.threads + b.threads, tokens: a.tokens.Add(b.tokens), cost: a.cost + b.cost}
}

// shrink is the tally's mean task, pulled towards prior as if weight more
// threads had used it.
func (a tally) shrink(prior usage.Tokens, weight float64) usage.Tokens {
	n := float64(a.threads) + weight
	if n <= 0 {
		return prior
	}
	mean := func(sum, prior int64) int64 {
		return int64((float64(sum) + weight*float64(prior)) / n)
	}
	return usage.Tokens{
		Input:      mean(a.tokens.Input, prior.Input),
		Output:     mean(a.tokens.Output, prior.Output),
		CacheRead:  mean(a.tokens.CacheRead, prior.CacheRead),
		CacheWrite: mean(a.tokens.CacheWrite, prior.CacheWrite),
	}
}
