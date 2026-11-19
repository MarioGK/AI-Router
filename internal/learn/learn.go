package learn

import (
	"context"
	"log/slog"
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/outcome"
	"github.com/MarioGK/AI-Router/internal/pin"
)

// Store reads the decisions to learn from and searches their first
// prompts; *pin.Store is one.
type Store interface {
	Precedents(ctx context.Context, ids []int64) ([]pin.Precedent, error)
	SimilarPrompts(ctx context.Context, match string, from int64, exclude string, limit, offset int) ([]pin.Similar, error)
	PromptTerms(ctx context.Context) (map[string]int, int, error)
}

// Outcomes are the settled, decided threads the outcome table judged;
// *outcome.Table is one.
type Outcomes interface {
	Outcomes() []outcome.Outcome
}

// Config is what a learner needs to know: the ladder, lightest first, the
// reasoning level ids, the window the outcomes come from, whose quarter
// is the half-life of a label, and how many neighbours a prior is built
// from.
type Config struct {
	Ladder     []string
	Levels     []string
	Window     time.Duration
	Neighbours int
}

// searchTimeout bounds a decision's similarity search, so a slow database
// delays no thread for long: without a prior the deciders decide alone.
const searchTimeout = 2 * time.Second

// A refresh gives each search refreshSearch and itself refreshTimeout: a
// search that runs out replays nothing, and a refresh that runs out fits
// on the replays it made.
const (
	refreshSearch  = 50 * time.Millisecond
	refreshTimeout = 2 * time.Minute
)

// A search reads its matches in batches of overfetch per neighbour wanted,
// since some are unlabelled, until it has its neighbours, runs out of
// matches or has read maxScanned of them.
const (
	overfetch  = 5
	maxScanned = 1000
)

// maxReplays bounds the labelled decisions a fit replays, the newest, each
// with one search: a few thousand replays fit a weight well enough, and
// every label still serves as a neighbour.
const maxReplays = 2000

// Learner labels settled threads, builds priors from similar labelled
// prompts and fits how much they weigh.
type Learner struct {
	store    Store
	outcomes Outcomes
	config   Config
	log      *slog.Logger
	now      func() time.Time
	// searchBound is how long a refresh gives each search.
	searchBound time.Duration

	mu sync.RWMutex
	// labels holds the labels by decision; a refresh replaces the map
	// whole, so a reader may keep it. from is the oldest labelled
	// decision, where searches start.
	labels map[int64]Label
	from   int64
	// documents counts the prompts of the index holding each word, which
	// with the postings budget picks the words a search asks for.
	documents map[string]int
	budget    int
	// fit and levelFit are the weights the history earned for the model
	// and the level, over replays and levelReplays labelled decisions.
	fit, levelFit         float64
	replays, levelReplays int
	refreshed             time.Time
	took                  time.Duration
}

// New returns a learner that labels the outcomes and searches store.
func New(store Store, outcomes Outcomes, config Config, log *slog.Logger) *Learner {
	if config.Neighbours <= 0 {
		config.Neighbours = 20
	}
	return &Learner{store: store, outcomes: outcomes, config: config, log: log, now: time.Now, searchBound: refreshSearch,
		labels: map[int64]Label{}}
}

// halfLife is how long a label takes to count half as much.
func (l *Learner) halfLife() time.Duration {
	return l.config.Window / 4
}

// Refresh labels the settled threads the outcome table judged and fits
// the weights again: each labelled decision with a recorded verdict, up to
// maxReplays of the newest, is replayed with the prior its neighbours,
// itself and its thread left out, would have given it. It costs one pass
// over the index's vocabulary and one bounded, indexed search per replay.
func (l *Learner) Refresh(ctx context.Context) error {
	start := time.Now()
	ctx, cancel := context.WithTimeout(ctx, refreshTimeout)
	defer cancel()
	outcomes := l.outcomes.Outcomes()
	ids := make([]int64, 0, len(outcomes))
	for _, o := range outcomes {
		ids = append(ids, o.Decision)
	}
	precedents, err := l.store.Precedents(ctx, ids)
	if err != nil {
		return err
	}
	byID := make(map[int64]pin.Precedent, len(precedents))
	for _, p := range precedents {
		byID[p.ID] = p
	}
	counts, indexed, err := l.store.PromptTerms(ctx)
	if err != nil {
		return err
	}
	budget := Postings(indexed)
	labels := map[int64]Label{}
	from := int64(0)
	for _, o := range outcomes {
		p, ok := byID[o.Decision]
		if !ok {
			continue
		}
		label, ok := Derive(o, p.Review, l.config.Ladder)
		if !ok {
			continue
		}
		label.At = p.At
		labels[label.Decision] = label
		if from == 0 || label.Decision < from {
			from = label.Decision
		}
	}
	var models, levels []replay
	ids = slices.Sorted(maps.Keys(labels))
	// The newest replayable labels, so labels without a recorded verdict
	// take none of the replays.
	slices.Reverse(ids)
	replayed := 0
	for _, id := range ids {
		label, p := labels[id], byID[id]
		verdict, offered := deciding(p)
		if verdict == nil {
			continue
		}
		if ctx.Err() != nil || replayed == maxReplays {
			break
		}
		replayed++
		searchCtx, cancel := context.WithTimeout(ctx, l.searchBound)
		neighbours, err := l.search(searchCtx, Match(p.Prompt, counts, budget), p.Key, labels, from)
		timedOut := searchCtx.Err() != nil
		cancel()
		if timedOut {
			continue
		}
		if err != nil {
			return err
		}
		prior := build(neighbours, offered, l.config.Levels, p.At, l.halfLife(), 0, 0)
		if slices.Contains(offered, label.Model) {
			models = append(models, replay{verdict: verdict.Probabilities, prior: prior.Models,
				neighbours: prior.Neighbours, label: label.Model, weight: label.Weight})
		}
		if slices.Contains(l.config.Levels, label.Level) && len(verdict.Levels) > 0 {
			levels = append(levels, replay{verdict: verdict.Levels, prior: prior.Levels,
				neighbours: prior.LevelNeighbours, label: label.Level, weight: label.Weight})
		}
	}
	fit, levelFit := fitWeight(models), fitWeight(levels)
	l.mu.Lock()
	defer l.mu.Unlock()
	l.labels, l.from, l.documents, l.budget, l.fit, l.levelFit = labels, from, counts, budget, fit, levelFit
	l.replays, l.levelReplays = len(models), len(levels)
	l.refreshed, l.took = l.now(), time.Since(start)
	return nil
}

// deciding is the verdict of a decision's deciding attempt, the first
// that answered the decision question, and the candidates it was offered;
// nil when no decider answered.
func deciding(p pin.Precedent) (*decide.Attempt, []string) {
	i := slices.IndexFunc(p.Attempts, func(a decide.Attempt) bool {
		return len(a.Pair) == 0 && (a.Outcome == decide.Decided || a.Outcome == decide.Doubted) && len(a.Probabilities) > 0
	})
	if i < 0 {
		return nil, nil
	}
	offered := p.Basis.Candidates
	if len(offered) == 0 {
		offered = slices.Sorted(maps.Keys(p.Attempts[i].Probabilities))
	}
	return &p.Attempts[i], offered
}

// search finds the labelled neighbours a match finds, at most the
// configured number, best first, leaving out the thread key. Similarity is
// each match's bm25 score against the best match's, the thread's own
// decisions aside.
func (l *Learner) search(ctx context.Context, match, key string, labels map[int64]Label, from int64) ([]neighbour, error) {
	if match == "" || len(labels) == 0 {
		return nil, nil
	}
	best := 0.0
	var neighbours []neighbour
	batch := l.config.Neighbours * overfetch
	for offset := 0; offset < maxScanned && len(neighbours) < l.config.Neighbours; offset += batch {
		found, err := l.store.SimilarPrompts(ctx, match, from, key, batch, offset)
		if err != nil {
			return nil, err
		}
		for _, similar := range found {
			best = max(best, similar.Score)
			if label, ok := labels[similar.Event]; ok && len(neighbours) < l.config.Neighbours {
				neighbours = append(neighbours, neighbour{Similarity: similar.Score, Label: label})
			}
		}
		if len(found) < batch {
			break
		}
	}
	for i := range neighbours {
		if best > 0 {
			neighbours[i].Similarity /= best
		} else {
			neighbours[i].Similarity = 1
		}
	}
	return neighbours, nil
}

// Prior is what the history suggests for a new thread's prompt among the
// offered candidates and levels, its own earlier decisions (key) left
// out. Without labels, or a prompt, it carries only the fitted weights;
// when the search fails the deciders decide alone.
func (l *Learner) Prior(ctx context.Context, prompt, key string, candidates, levels []string) *decide.Prior {
	l.mu.RLock()
	labels, from, counts, budget, fit, levelFit := l.labels, l.from, l.documents, l.budget, l.fit, l.levelFit
	l.mu.RUnlock()
	prior := &decide.Prior{Fit: fit, LevelFit: levelFit}
	if prompt == "" || len(labels) == 0 || len(candidates) == 0 {
		return prior
	}
	searchCtx, cancel := context.WithTimeout(ctx, searchTimeout)
	defer cancel()
	neighbours, err := l.search(searchCtx, Match(prompt, counts, budget), key, labels, from)
	if err != nil {
		if ctx.Err() == nil && l.log != nil {
			l.log.Warn("similar prompt search failed", "error", err)
		}
		return prior
	}
	built := build(neighbours, candidates, levels, l.now(), l.halfLife(), fit, levelFit)
	return &built
}

// Summary is what the learner holds: whether it is on, the labels by
// kind, the weights the history earned for the model and the level and
// how many labelled decisions each fit replayed, and when it last
// refreshed and how long that took.
type Summary struct {
	Enabled      bool           `json:"enabled"`
	Labels       int            `json:"labels"`
	Kinds        map[string]int `json:"kinds"`
	Fit          float64        `json:"fit"`
	LevelFit     float64        `json:"level_fit"`
	Replays      int            `json:"replays"`
	LevelReplays int            `json:"level_replays"`
	MinLabels    int            `json:"min_labels"`
	Neighbours   int            `json:"neighbours"`
	Refreshed    time.Time      `json:"refreshed,omitzero"`
	TookMS       int64          `json:"took_ms"`
}

// Summary reports what the learner holds; a nil learner is off.
func (l *Learner) Summary() Summary {
	if l == nil {
		return Summary{Kinds: map[string]int{}, MinLabels: minLabels}
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	summary := Summary{Enabled: true, Labels: len(l.labels), Kinds: map[string]int{}, Fit: l.fit, LevelFit: l.levelFit,
		Replays: l.replays, LevelReplays: l.levelReplays, MinLabels: minLabels, Neighbours: l.config.Neighbours,
		Refreshed: l.refreshed, TookMS: l.took.Milliseconds()}
	for _, kind := range Kinds {
		summary.Kinds[kind] = 0
	}
	for _, label := range l.labels {
		summary.Kinds[label.Kind]++
	}
	return summary
}
