package learn

import (
	"context"
	"fmt"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/outcome"
	"github.com/MarioGK/AI-Router/internal/pin"
)

// judged is the outcome table's view of the settled threads.
type judged []outcome.Outcome

func (j judged) Outcomes() []outcome.Outcome { return j }

// guess is what a decider that cannot tell the topics apart answers.
var guess = map[string]float64{"glm-5.3": 0.5, "gpt-6.1-sol": 0.3, "claude-opus-5-5": 0.2}

// topics are the first prompts of two kinds of task: chores that sufficed
// on glm-5.3 and concurrency bugs operators sent to claude-opus-5-5.
var topics = []struct {
	prompt   string
	model    string
	reviewed bool
}{
	{"Rename the variable %d in the docs and fix the typo in the readme", "glm-5.3", false},
	{"Find the deadlock %d in the concurrency code, a race between the scheduler goroutines", "claude-opus-5-5", true},
}

// history records n decisions of each topic, all decided by the guessing
// decider, and how their threads went.
func history(t testing.TB, store *pin.Store, n int) judged {
	t.Helper()
	return historyOf(t, store, n, false)
}

// historyOf is history; with legacy set the decisions keep only the
// prompt as judged, as decisions from before first prompts were kept do.
func historyOf(t testing.TB, store *pin.Store, n int, legacy bool) judged {
	t.Helper()
	ctx := context.Background()
	var outcomes judged
	id := int64(0)
	for i := range n {
		for _, topic := range topics {
			id++
			key := fmt.Sprintf("thread-%d", id)
			event := pin.Event{Key: key,
				Kind: pin.Decision, Model: "glm-5.3", Level: "low", Choice: "glm-5.3", Probabilities: guess,
				FirstPrompt: fmt.Sprintf(topic.prompt, i),
				Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "glm-5.3", Probabilities: guess,
					Level: "low", Levels: map[string]float64{"low": 0.6, "medium": 0.3, "high": 0.1}}},
				Basis: pin.Basis{Candidates: models}}
			if legacy {
				event.Prompt, event.FirstPrompt = event.FirstPrompt, ""
			}
			if _, err := store.Create(ctx, pin.Pin{Key: key, Model: "glm-5.3", Level: "low"}, event); err != nil {
				t.Fatal(err)
			}
			if topic.reviewed {
				if err := store.SaveReview(ctx, pin.Review{Event: id, Model: topic.model, Level: "high"}); err != nil {
					t.Fatal(err)
				}
			}
			outcomes = append(outcomes, outcome.Outcome{Key: key, Model: "glm-5.3", Level: "low", Decision: id, Requests: 40})
		}
	}
	return outcomes
}

func open(t testing.TB) *pin.Store {
	t.Helper()
	store, err := pin.Open(filepath.Join(t.TempDir(), "router.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	return store
}

// patient gives a learner's searches time enough on a slow test machine,
// under the race detector.
func patient(l *Learner) *Learner {
	l.searchBound = time.Minute
	return l
}

func TestATaskThatRecursOftenStillFindsItsHistory(t *testing.T) {
	store := open(t)
	ctx := t.Context()
	var outcomes judged
	for i := range 150 {
		key := fmt.Sprintf("bump-%d", i)
		if _, err := store.Create(ctx, pin.Pin{Key: key, Model: "glm-5.3"}, pin.Event{Key: key, Kind: pin.Decision,
			Model: "glm-5.3", FirstPrompt: "Bump the dependency versions and regenerate the lockfile",
			Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "glm-5.3", Probabilities: guess}},
			Basis:    pin.Basis{Candidates: models}}); err != nil {
			t.Fatal(err)
		}
		outcomes = append(outcomes, outcome.Outcome{Key: key, Model: "glm-5.3", Decision: int64(i + 1), Requests: 40})
	}
	learner := patient(New(store, outcomes, config, nil))
	if err := learner.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	// Every word is in 150 prompts, past the budget of 100: the search
	// still asks for the rarest few.
	counts, indexed, err := store.PromptTerms(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if match := Match("Bump the dependency versions again", counts, Postings(indexed)); match == "" {
		t.Fatal("a recurring task searches for nothing")
	}
	prior := learner.Prior(ctx, "Bump the dependency versions again", "", models, levels)
	if len(prior.Similar) == 0 || prior.Similar[0].Model != "glm-5.3" {
		t.Fatalf("prior = %+v", prior)
	}
}

func TestUnlabelledAndOwnMatchesDoNotCrowdOutNeighbours(t *testing.T) {
	store := open(t)
	ctx := t.Context()
	create := func(key, prompt string) {
		t.Helper()
		if _, err := store.Create(ctx, pin.Pin{Key: key, Model: "glm-5.3"}, pin.Event{Key: key, Kind: pin.Decision,
			Model: "glm-5.3", FirstPrompt: prompt}); err != nil {
			t.Fatal(err)
		}
	}
	// Five labelled decisions match less well than 150 unlabelled ones and
	// 150 of the asking thread's own.
	var outcomes judged
	for i := range 5 {
		key := fmt.Sprintf("labelled-%d", i)
		create(key, "A deadlock somewhere in the scheduler, with many other words around it to dilute the match")
		outcomes = append(outcomes, outcome.Outcome{Key: key, Model: "claude-opus-5-5", Decision: int64(i + 1),
			Requests: 40})
	}
	for i := range 150 {
		create(fmt.Sprintf("unlabelled-%d", i), "deadlock scheduler deadlock scheduler")
		create("own", "deadlock scheduler deadlock scheduler")
	}
	learner := patient(New(store, outcomes, config, nil))
	if err := learner.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	prior := learner.Prior(ctx, "deadlock scheduler", "own", models, levels)
	if len(prior.Similar) != 5 || prior.Similar[0].Model != "claude-opus-5-5" || prior.Similar[0].Similarity >= 1 {
		t.Fatalf("prior = %+v", prior)
	}
}

func TestASearchThatRunsOutReplaysNothing(t *testing.T) {
	store := open(t)
	learner := New(store, history(t, store, 15), config, nil)
	learner.searchBound = time.Nanosecond
	if err := learner.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	if summary := learner.Summary(); summary.Labels != 30 || summary.Replays != 0 || summary.Fit != 0 {
		t.Fatalf("summary = %+v", summary)
	}
}

var config = Config{Ladder: models, Levels: levels, Window: 30 * 24 * time.Hour, Neighbours: 20}

func TestTheHistoryLearnsWhereSimilarPromptsBelong(t *testing.T) {
	store := open(t)
	learner := patient(New(store, history(t, store, 15), config, nil))
	if err := learner.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	summary := learner.Summary()
	if summary.Labels != 30 || summary.Kinds[Reviewed] != 15 || summary.Kinds[Sufficed] != 15 || summary.Replays != 30 ||
		summary.Fit <= 0 || summary.LevelFit <= 0 {
		t.Fatalf("summary = %+v", summary)
	}
	prior := learner.Prior(t.Context(), "There is a deadlock in the scheduler concurrency code", "", models, levels)
	if prior.Weight <= 0 || prior.Models["claude-opus-5-5"] <= prior.Models["glm-5.3"] || len(prior.Similar) == 0 ||
		prior.Similar[0].Model != "claude-opus-5-5" || prior.Similar[0].Similarity != 1 {
		t.Fatalf("prior = %+v", prior)
	}
	// A thread's own decisions are no precedent for it.
	own := learner.Prior(t.Context(), fmt.Sprintf(topics[1].prompt, 0), "thread-2", models, levels)
	for _, p := range own.Similar {
		if p.Event == 2 {
			t.Fatalf("the thread's own decision is its neighbour: %+v", own.Similar)
		}
	}
	// Only offered candidates, and nothing for a prompt with no words.
	offered := learner.Prior(t.Context(), "deadlock race", "", []string{"glm-5.3", "gpt-6.1-sol"}, nil)
	if _, ok := offered.Models["claude-opus-5-5"]; ok || len(offered.Levels) != 0 {
		t.Fatalf("prior over the offered = %+v", offered)
	}
	if none := learner.Prior(t.Context(), "?!", "", models, levels); none.Similar != nil || none.Fit != summary.Fit {
		t.Fatalf("prior without words = %+v", none)
	}
}

func TestDecisionsWithoutAFirstPromptStillTeach(t *testing.T) {
	store := open(t)
	learner := patient(New(store, historyOf(t, store, 15, true), config, nil))
	if err := learner.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	if summary := learner.Summary(); summary.Replays != 30 || summary.Fit <= 0 {
		t.Fatalf("summary over legacy decisions = %+v", summary)
	}
	prior := learner.Prior(t.Context(), "There is a deadlock in the scheduler concurrency code", "", models, levels)
	if len(prior.Similar) == 0 || prior.Similar[0].Model != "claude-opus-5-5" || prior.Weight <= 0 {
		t.Fatalf("prior over legacy decisions = %+v", prior)
	}
}

func TestWithoutLabelsTheHistoryChangesNothing(t *testing.T) {
	store := open(t)
	outcomes := history(t, store, 3)
	// Too few labels to fit: neighbours are found but weigh nothing.
	learner := patient(New(store, outcomes, config, nil))
	if err := learner.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	prior := learner.Prior(t.Context(), "deadlock in the concurrency code", "", models, levels)
	if summary := learner.Summary(); summary.Labels != 6 || summary.Fit != 0 || summary.LevelFit != 0 {
		t.Fatalf("summary = %+v", summary)
	}
	if prior.Weight != 0 || prior.LevelWeight != 0 || len(prior.Similar) == 0 {
		t.Fatalf("prior = %+v", prior)
	}
	// No outcome, no label, no neighbour.
	empty := New(store, judged{}, config, nil)
	if err := empty.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	if prior := empty.Prior(t.Context(), "deadlock", "", models, levels); prior.Similar != nil || prior.Weight != 0 {
		t.Fatalf("prior without labels = %+v", prior)
	}
}

// corpus records n decisions whose first prompts draw some forty words
// each from a vocabulary of a few thousand, with Zipf's law, as prompts
// do, and labels each after its first word.
func corpus(b *testing.B, store *pin.Store, n int) judged {
	b.Helper()
	ctx := context.Background()
	r := rand.New(rand.NewPCG(5, 6))
	zipf := rand.NewZipf(r, 1.1, 1, 4999)
	var outcomes judged
	for i := range n {
		var prompt strings.Builder
		for range 40 {
			fmt.Fprintf(&prompt, "word%c%c%c ", 'a'+zipf.Uint64()%26, 'a'+zipf.Uint64()/26%26, 'a'+zipf.Uint64()/676%26)
		}
		key := fmt.Sprintf("thread-%d", i)
		if _, err := store.Create(ctx, pin.Pin{Key: key, Model: "glm-5.3"}, pin.Event{Key: key, Kind: pin.Decision,
			Model: "glm-5.3", Choice: "glm-5.3", Probabilities: guess, FirstPrompt: prompt.String(),
			Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "glm-5.3", Probabilities: guess}},
			Basis:    pin.Basis{Candidates: models}}); err != nil {
			b.Fatal(err)
		}
		outcomes = append(outcomes, outcome.Outcome{Key: key, Model: models[len(prompt.String())%3], Decision: int64(i + 1),
			Requests: 40})
	}
	return outcomes
}

// BenchmarkRefresh times a refresh over 5,000 labelled decisions, one
// search per label.
func BenchmarkRefresh(b *testing.B) {
	store := open(b)
	learner := New(store, corpus(b, store, 5000), config, nil)
	b.ResetTimer()
	for range b.N {
		if err := learner.Refresh(b.Context()); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(learner.Summary().Labels), "labels")
}

// BenchmarkPrior times the search a decision makes among 5,000 labelled
// decisions.
func BenchmarkPrior(b *testing.B) {
	store := open(b)
	learner := New(store, corpus(b, store, 5000), config, nil)
	if err := learner.Refresh(b.Context()); err != nil {
		b.Fatal(err)
	}
	prompt := "worda wordb wordc wordbaa wordcab wordzzz wordqqb wordkma wordbbb wordaab"
	b.ResetTimer()
	for range b.N {
		learner.Prior(b.Context(), prompt, "", models, levels)
	}
}

func TestTheExplorationIsReadFromTheRecordedBasis(t *testing.T) {
	store := open(t)
	ctx := t.Context()
	now := time.Now()
	for _, key := range []string{"explored", "plain"} {
		basis := pin.Basis{Candidates: models}
		if key == "explored" {
			basis.ExploreEligible, basis.Explore, basis.Propensity, basis.ExploredFrom = true, true, 0.25, "gpt-6.1-sol"
		}
		if _, err := store.Create(ctx, pin.Pin{Key: key, Model: "glm-5.3", Level: "low"}, pin.Event{Key: key,
			Kind: pin.Decision, Model: "glm-5.3", Level: "low", Basis: basis}); err != nil {
			t.Fatal(err)
		}
		if err := store.LogRequest(ctx, pin.RequestLog{At: now, Key: key, Kind: "messages", Model: "glm-5.3",
			Reason: "decided", Status: 200, Latency: time.Second}); err != nil {
			t.Fatal(err)
		}
	}
	threads, err := store.ThreadActivities(ctx, now.Add(-time.Hour), now.Add(time.Hour))
	if err != nil || len(threads) != 2 {
		t.Fatalf("threads = %+v, %v", threads, err)
	}
	weights := map[string]float64{}
	for _, thread := range threads {
		label, ok := Derive(outcome.Derive(thread, rules), nil, ladder)
		if !ok {
			t.Fatalf("no label for %+v", thread)
		}
		weights[thread.Key] = label.Weight
	}
	if weights["explored"] != SufficedWeight*4 || weights["plain"] != SufficedWeight {
		t.Fatalf("weights = %v", weights)
	}
}
