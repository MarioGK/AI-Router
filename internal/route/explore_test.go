package route

import (
	"cmp"
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"log/slog"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/thread"
	"github.com/MarioGK/AI-Router/internal/wire"
)

// verdictDecider answers every question with the same verdict.
type verdictDecider struct{ verdict decide.Verdict }

func (d *verdictDecider) Name() string { return "verdict" }

func (d *verdictDecider) Info() decide.DeciderInfo {
	return decide.DeciderInfo{Name: d.Name(), Kind: decide.KindSystemOne, Model: "test"}
}

func (d *verdictDecider) Decide(context.Context, string, []decide.Candidate, []decide.Level) (decide.Verdict, error) {
	return d.verdict, nil
}

// spillLoad spills a saturated model to another.
type spillLoad map[string]string

func (l spillLoad) Spill(model string) (string, bool) {
	spill, ok := l[model]
	return spill, ok
}

// exploreTasks are the costs per task of the explore ladder: with no miss
// penalty, a verdict of 40% light and 60% middle scores middle at 2 and
// light at 1 + 0.6 × 2 = 2.2, a borderline call 10% apart.
var exploreTasks = map[string]float64{"light": 1, "middle": 2, "heavy": 10}

// borderline is a verdict for middle that the economy keeps on middle,
// with the step below it 10% dearer.
func borderline() decide.Verdict {
	return decide.Verdict{Choice: "middle", Probabilities: map[string]float64{"light": 0.4, "middle": 0.6}}
}

type exploreHarness struct {
	r       *Resolver
	decider *verdictDecider
	path    string
	draws   int
}

// exploreResolver routes over a three-step ladder and a specialist, with
// the cost economy on and exploration at rate; every draw is draw.
func exploreResolver(t *testing.T, rate, draw float64) *exploreHarness {
	t.Helper()
	path := filepath.Join(t.TempDir(), "router.db")
	store, err := pin.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	auto := config.Auto{
		DefaultModel: "middle", AuxModel: "light", DecideTimeout: time.Second, StateMaxChars: 1000,
		EscalationThreshold: 0.9, RecordPrompts: new(false),
		Candidates: []config.Candidate{
			{ID: "light"}, {ID: "middle", Vision: true}, {ID: "heavy", Vision: true},
			{ID: "special", Vision: true, Specialist: true},
		},
		Learn: config.Learn{Explore: rate},
	}
	h := &exploreHarness{decider: &verdictDecider{verdict: borderline()}, path: path}
	levels := []decide.Level{{ID: "high"}, {ID: "xhigh"}}
	chain := decide.NewChain([]decide.Step{{Decider: h.decider, Timeout: time.Second}}, auto.EscalationThreshold,
		decide.DoubtEscalate, levels).Price(decide.Economy{Miss: 0,
		Task: func(model, _ string, _ []decide.Candidate) (float64, bool) {
			cost, ok := exploreTasks[model]
			return cost, ok
		}})
	h.r = New(auto, "test", store, chain, served(), nil, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	h.r.random = func() float64 {
		h.draws++
		return draw
	}
	t.Cleanup(h.r.Close)
	return h
}

// decide evaluates a new thread's first prompt the way a decision does,
// under the current routing settings.
func (h *exploreHarness) decide(t *testing.T, images bool) pin.Event {
	t.Helper()
	view, err := h.r.snapshot(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	view.random = h.r.random
	conversation := wire.Conversation{LatestUser: "Fix the flaky retry test.", HasImage: images, HasTools: true}
	event, _ := view.evaluateAmong(t.Context(), wire.Messages, thread.Identity{}, conversation, view.candidates(images), true)
	return event
}

// basis reads the decision log's raw basis of a thread's events, oldest
// first, as the outcome learner reads it.
func (h *exploreHarness) basis(t *testing.T, key string) []map[string]any {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+h.path+"?_pragma=busy_timeout(5000)")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	rows, err := db.QueryContext(t.Context(), `SELECT basis FROM events WHERE key = ? ORDER BY id`, key)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var all []map[string]any
	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			t.Fatal(err)
		}
		var basis map[string]any
		if err := json.Unmarshal([]byte(raw), &basis); err != nil {
			t.Fatal(err)
		}
		all = append(all, basis)
	}
	return all
}

func TestExploreRoutesABorderlineDecisionOneStepCheaper(t *testing.T) {
	h := exploreResolver(t, 0.1, 0.05)
	id := thread.Identity{Session: "explored"}
	target := h.r.Resolve(t.Context(), wire.Messages, id, wire.Conversation{LatestUser: "Fix the flaky retry test.",
		HasTools: true, Prompted: true})
	if target.Model != "light" || target.Reason != ReasonDecided {
		t.Fatalf("explored decision = %+v, want light", target)
	}
	got := h.basis(t, target.Key)
	if len(got) != 1 || got[0]["explore"] != true || got[0]["propensity"] != 0.1 ||
		got[0]["explored_from"] != "middle" || got[0]["explore_eligible"] != true {
		t.Fatalf("explored basis = %v", got)
	}
}

func TestExploreRecordsThePropensityOfAKeptStep(t *testing.T) {
	h := exploreResolver(t, 0.1, 0.5)
	id := thread.Identity{Session: "kept"}
	target := h.r.Resolve(t.Context(), wire.Messages, id, wire.Conversation{LatestUser: "Fix the flaky retry test.",
		HasTools: true, Prompted: true})
	if target.Model != "middle" {
		t.Fatalf("kept decision = %+v, want middle", target)
	}
	got := h.basis(t, target.Key)
	if len(got) != 1 || got[0]["explore"] != nil || got[0]["explore_eligible"] != true ||
		got[0]["propensity"] != 0.9 || got[0]["explored_from"] != nil {
		t.Fatalf("kept basis = %v", got)
	}
}

func TestExploreDrawsFromTheInjectedSource(t *testing.T) {
	// The draw explores below the rate and keeps the step from it on.
	for draw, want := range map[float64]string{0: "light", 0.0999: "light", 0.1: "middle", 0.99: "middle"} {
		h := exploreResolver(t, 0.1, draw)
		if got := h.decide(t, false); got.Model != want || h.draws != 1 {
			t.Errorf("draw %v: model %s after %d draws, want %s after one", draw, got.Model, h.draws, want)
		}
	}
}

func TestExploreOnlyBorderlineDecisions(t *testing.T) {
	h := exploreResolver(t, 0.2, 0)
	// light scores 10% above middle: within a margin of 0.1, not of 0.05.
	h.r.auto.Learn.ExploreMargin = new(0.1)
	if got := h.decide(t, false); !got.Basis.Explore || got.Model != "light" {
		t.Fatalf("within the margin: %+v", got.Basis)
	}
	h.r.auto.Learn.ExploreMargin = new(0.05)
	if got := h.decide(t, false); got.Basis.ExploreEligible || got.Model != "middle" || h.draws != 1 {
		t.Fatalf("beyond the margin: %s, %+v after %d draws", got.Model, got.Basis, h.draws)
	}
	// A clear call for middle is no borderline one under the default margin.
	h.r.auto.Learn.ExploreMargin = nil
	h.decider.verdict = decide.Verdict{Choice: "middle", Probabilities: map[string]float64{"light": 0.1, "middle": 0.9}}
	exploreTasks["light"] = 1.9
	t.Cleanup(func() { exploreTasks["light"] = 1 })
	if got := h.decide(t, false); got.Basis.ExploreEligible {
		t.Fatalf("1.9 + 0.9 × 2 against 2 is no borderline call: %+v", got.Basis)
	}
}

func TestExploreEligibility(t *testing.T) {
	for name, c := range map[string]struct {
		setup  func(t *testing.T, h *exploreHarness)
		images bool
		want   string
	}{
		"images the step below cannot read": {images: true},
		"the step below outside the model group": {setup: func(t *testing.T, h *exploreHarness) {
			if _, err := h.r.SetSettings(t.Context(), Settings{Mode: ModeAuto, JevEnabled: true,
				Models: []string{"middle", "heavy"}}); err != nil {
				t.Fatal(err)
			}
		}},
		"the step below without quota": {setup: func(_ *testing.T, h *exploreHarness) {
			h.r.quota = fakeQuota{"light": true}
		}},
		"the step below retired": {setup: func(_ *testing.T, h *exploreHarness) {
			h.r.auto.Retired = []string{"light"}
		}},
		"the step below saturated": {setup: func(_ *testing.T, h *exploreHarness) {
			h.r.load = spillLoad{"light": "heavy"}
		}},
		"an xhigh task": {setup: func(_ *testing.T, h *exploreHarness) {
			h.decider.verdict.Levels = map[string]float64{"xhigh": 1}
		}},
		"a specialist": {want: "special", setup: func(_ *testing.T, h *exploreHarness) {
			h.decider.verdict = decide.Verdict{Choice: "special", Probabilities: map[string]float64{"special": 1}}
		}},
		"the lightest step": {want: "light", setup: func(_ *testing.T, h *exploreHarness) {
			h.decider.verdict = decide.Verdict{Choice: "light", Probabilities: map[string]float64{"light": 1}}
		}},
		"fixed mode": {setup: func(t *testing.T, h *exploreHarness) {
			if _, err := h.r.SetSettings(t.Context(), Settings{Mode: ModeFixed, FixedModel: "middle"}); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(name, func(t *testing.T) {
			h := exploreResolver(t, 0.2, 0)
			if c.setup != nil {
				c.setup(t, h)
			}
			got := h.decide(t, c.images)
			if got.Basis.ExploreEligible || got.Basis.Explore || got.Basis.Propensity != 0 || h.draws != 0 {
				t.Fatalf("explored %s: %+v after %d draws", got.Model, got.Basis, h.draws)
			}
			if want := cmp.Or(c.want, "middle"); got.Model != want {
				t.Fatalf("the decision went to %s, want %s", got.Model, want)
			}
		})
	}
}

func TestATrialNeverExplores(t *testing.T) {
	h := exploreResolver(t, 0.2, 0)
	event, _ := h.r.Try(t.Context(), "Fix the flaky retry test.", false)
	if event.Model != "middle" || event.Basis.ExploreEligible || h.draws != 0 {
		t.Fatalf("trial = %s, %+v after %d draws", event.Model, event.Basis, h.draws)
	}
}

func TestExploreOffChangesNothing(t *testing.T) {
	on := exploreResolver(t, 0.2, 0)
	off := exploreResolver(t, 0, 0)
	for _, verdict := range []decide.Verdict{
		borderline(),
		{Choice: "light", Probabilities: map[string]float64{"light": 0.5, "middle": 0.5}},
		{Choice: "middle", Probabilities: map[string]float64{"middle": 0.7, "heavy": 0.3}},
		{Choice: "special", Probabilities: map[string]float64{"special": 1}},
	} {
		on.decider.verdict, off.decider.verdict = verdict, verdict
		h := off
		view, err := h.r.snapshot(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		conversation := wire.Conversation{LatestUser: "Fix the flaky retry test.", HasTools: true}
		// The same decision without exploration, as before it existed.
		before, _ := view.evaluateAmong(t.Context(), wire.Messages, thread.Identity{}, conversation, view.candidates(false), false)
		after := off.decide(t, false)
		before.Latency, after.Latency = 0, 0
		for i := range before.Attempts {
			before.Attempts[i].LatencyMS = 0
		}
		for i := range after.Attempts {
			after.Attempts[i].LatencyMS = 0
		}
		if !reflect.DeepEqual(before, after) {
			t.Errorf("explore 0 changed the decision for %s:\n%+v\nwant\n%+v", verdict.Choice, after, before)
		}
		if off.draws != 0 {
			t.Errorf("explore 0 drew %d times", off.draws)
		}
	}
	// The same decision with exploration on does explore.
	on.decider.verdict = borderline()
	if got := on.decide(t, false); got.Model != "light" || !got.Basis.Explore {
		t.Fatalf("explore 0.2 kept %s: %+v", got.Model, got.Basis)
	}
}

func TestRedecisionExplores(t *testing.T) {
	h := exploreResolver(t, 0.2, 0)
	h.r.auto.Redecide.Idle = new(5 * time.Minute)
	// middle scores 2 + 10 against heavy's 10, 20% more: beyond this margin.
	h.r.auto.Learn.ExploreMargin = new(0.15)
	h.decider.verdict = decide.Verdict{Choice: "heavy", Probabilities: map[string]float64{"heavy": 1}}
	id := thread.Identity{Session: "redecided"}
	prompted := wire.Conversation{LatestUser: "Fix the flaky retry test.", HasTools: true, Prompted: true}
	first := h.r.Resolve(t.Context(), wire.Messages, id, prompted)
	if first.Model != "heavy" {
		t.Fatalf("first decision = %+v", first)
	}
	h.decider.verdict = borderline()
	h.r.now = func() time.Time { return time.Now().Add(time.Hour) }
	again := h.r.Resolve(t.Context(), wire.Messages, id, prompted)
	got := h.basis(t, again.Key)
	if again.Model != "light" || len(got) != 2 || got[1]["explore"] != true || got[1]["explored_from"] != "middle" {
		t.Fatalf("redecision = %+v, basis %v", again, got)
	}
}

func TestRedecisionNeverExploresTowardsTheFailedModel(t *testing.T) {
	h := exploreResolver(t, 0.2, 0)
	h.r.auto.Redecide.Idle = new(5 * time.Minute)
	h.r.auto.Fallback = map[string]string{"light": "middle"}
	key := h.r.key("failed", "")
	if _, err := h.r.store.Create(t.Context(), pin.Pin{Key: key, Model: "light"}, pin.Event{Key: key, Kind: pin.Decision, Model: "light"}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.r.store.Failover(t.Context(), key, "light", "middle", true, pin.Event{Key: key, Kind: pin.Failover,
		Model: "middle", Previous: "light"}); err != nil {
		t.Fatal(err)
	}
	h.r.now = func() time.Time { return time.Now().Add(time.Hour) }
	got := h.r.Resolve(t.Context(), wire.Messages, thread.Identity{Session: "failed"},
		wire.Conversation{LatestUser: "Fix the flaky retry test.", HasTools: true, Prompted: true})
	basis := h.basis(t, key)
	if got.Model != "middle" || len(basis) != 3 || basis[2]["explore_eligible"] != nil || h.draws != 0 {
		t.Fatalf("redecision of a thread that failed over from light = %+v, basis %v", got, basis)
	}
}

func TestRedecisionNeverExploresAThreadThatMovedUp(t *testing.T) {
	h := exploreResolver(t, 0.2, 0)
	h.r.auto.Redecide.Idle = new(5 * time.Minute)
	key := h.r.key("moved-up", "")
	if _, err := h.r.store.Create(t.Context(), pin.Pin{Key: key, Model: "light"}, pin.Event{Key: key, Kind: pin.Decision, Model: "light"}); err != nil {
		t.Fatal(err)
	}
	p, err := h.r.store.Get(t.Context(), key)
	if err != nil {
		t.Fatal(err)
	}
	// A redecision moved the thread up from light to middle.
	if _, err := h.r.store.Redecide(t.Context(), *p, "middle", "", true, pin.Event{Key: key, Kind: pin.Redecision,
		Model: "middle", Previous: "light"}); err != nil {
		t.Fatal(err)
	}
	h.r.now = func() time.Time { return time.Now().Add(time.Hour) }
	got := h.r.Resolve(t.Context(), wire.Messages, thread.Identity{Session: "moved-up"},
		wire.Conversation{LatestUser: "Fix the flaky retry test.", HasTools: true, Prompted: true})
	basis := h.basis(t, key)
	if got.Model != "middle" || len(basis) != 3 || basis[2]["explore_eligible"] != nil || h.draws != 0 {
		t.Fatalf("redecision of a thread that moved up = %+v, basis %v", got, basis)
	}
}
