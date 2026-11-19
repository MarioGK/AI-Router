package route

import (
	"context"
	"slices"
	"testing"

	"github.com/MarioGK/AI-Router/internal/catalog"
	"github.com/MarioGK/AI-Router/internal/config"
)

// served is a catalog that has not loaded yet, which serves every model.
func served() *catalog.Catalog {
	return catalog.New("http://cliproxy.invalid", "", nil)
}

func TestVisionTargetPrefersTheLadder(t *testing.T) {
	auto := config.Auto{Candidates: []config.Candidate{
		{ID: "deepseek-flash", Vision: true},
		{ID: "gpt-6.1-sol", Vision: true},
		{ID: "glm-5.3"},
		{ID: "claude-opus-5-5", Vision: true, Specialist: true},
	}}
	r := &Resolver{auto: auto, catalog: served()}
	if got := r.visionTarget(auto.Rank("glm-5.3"), ""); got != "gpt-6.1-sol" {
		t.Fatalf("top-of-ladder text model upgrades to %q, want the nearest ladder step below", got)
	}
	if got := r.visionTarget(auto.Rank("glm-5.3"), "gpt-6.1-sol"); got != "deepseek-flash" {
		t.Fatalf("with the failed model excluded the upgrade goes to %q", got)
	}
	r.auto.Candidates = auto.Candidates[2:]
	if got := r.visionTarget(0, ""); got != "claude-opus-5-5" {
		t.Fatalf("only the specialist left: got %q", got)
	}
	r.auto.Candidates = auto.Candidates[2:3]
	if got := r.visionTarget(0, ""); got != "" {
		t.Fatalf("no vision candidate should mean no upgrade, got %q", got)
	}
}

// fakeQuota marks models no account can serve.
type fakeQuota map[string]bool

func (q fakeQuota) Available(model string) bool { return !q[model] }

func TestModelsNoAccountCanServeAreWithheld(t *testing.T) {
	auto := config.Auto{
		DefaultModel: "glm-5.3",
		Candidates:   []config.Candidate{{ID: "deepseek-flash"}, {ID: "glm-5.3"}, {ID: "gpt-6.1-sol"}, {ID: "claude-opus-5-5"}},
		Fallback:     map[string]string{"glm-5.3": "gpt-6.1-sol", "claude-opus-5-5": "glm-5.3"},
	}
	r := &Resolver{auto: auto, catalog: served(), quota: fakeQuota{"gpt-6.1-sol": true}}
	var offered []string
	for _, c := range r.candidates(false) {
		offered = append(offered, c.ID)
	}
	if !slices.Equal(offered, []string{"deepseek-flash", "glm-5.3", "claude-opus-5-5"}) {
		t.Fatalf("offered %q", offered)
	}
	if _, ok := r.Failover(context.Background(), Target{Model: "glm-5.3"}, 429, false); ok {
		t.Fatal("a thread failed over to a model no account can serve")
	}
	if next, ok := r.Failover(context.Background(), Target{Model: "claude-opus-5-5"}, 429, false); !ok || next.Model != "glm-5.3" {
		t.Fatalf("failover = %+v, %v", next, ok)
	}
	// A fallback no account can serve gives way to the default model.
	r.auto.Fallback = map[string]string{"claude-opus-5-5": "gpt-6.1-sol"}
	if next, ok := r.Failover(context.Background(), Target{Model: "claude-opus-5-5"}, 429, false); !ok || next.Model != "glm-5.3" {
		t.Fatalf("failover past a spent fallback = %+v, %v", next, ok)
	}
	r.auto.Fallback = auto.Fallback

	// With every candidate withheld there is no eligible new pin.
	r.quota = fakeQuota{"deepseek-flash": true, "glm-5.3": true, "gpt-6.1-sol": true, "claude-opus-5-5": true}
	if got := r.candidates(false); len(got) != 0 {
		t.Fatalf("offered %d candidates", len(got))
	}
}

func TestOnlyARunOfServerErrorsFailsOver(t *testing.T) {
	trip := 3
	r := &Resolver{auto: config.Auto{FailoverStatuses: []int{429, 500, 503, 529}, FailoverTrip: &trip}}
	for i, step := range []struct {
		model  string
		status int
		want   bool
	}{
		{"claude-opus-5-5", 529, false},
		{"claude-opus-5-5", 529, false},
		{"claude-opus-5-5", 200, false}, // an answer ends the run
		{"claude-opus-5-5", 529, false},
		{"gpt-6.1-sol", 503, false}, // runs are per model
		{"claude-opus-5-5", 503, false},
		{"claude-opus-5-5", 500, true},
		{"claude-opus-5-5", 529, true},
		{"claude-opus-5-5", 502, false}, // not a failover status
		{"gpt-6.1-sol", 429, true},      // a spent quota fails over at once
		{"gpt-6.1-sol", 400, false},
	} {
		if got := r.Answered(step.model, step.status); got != step.want {
			t.Errorf("step %d: Answered(%s, %d) = %v, want %v", i, step.model, step.status, got, step.want)
		}
	}
}

func TestImagesFailOverOnlyToModelsThatReadThem(t *testing.T) {
	auto := config.Auto{
		Candidates: []config.Candidate{
			{ID: "deepseek-flash", Vision: true}, {ID: "glm-5.3"}, {ID: "gpt-6.1-sol", Vision: true},
			{ID: "claude-opus-5-5", Vision: true},
		},
		Fallback: map[string]string{"gpt-6.1-sol": "glm-5.3", "deepseek-flash": "glm-5.3"},
	}
	r := &Resolver{auto: auto, catalog: served()}
	if next, ok := r.Failover(context.Background(), Target{Model: "gpt-6.1-sol"}, 429, false); !ok || next.Model != "glm-5.3" {
		t.Fatalf("text failover = %+v, %v", next, ok)
	}
	if next, ok := r.Failover(context.Background(), Target{Model: "gpt-6.1-sol"}, 429, true); !ok || next.Model != "claude-opus-5-5" {
		t.Fatalf("image failover = %+v, %v; want the nearest vision step", next, ok)
	}
	r.auto.Candidates = auto.Candidates[1:3]
	if next, ok := r.Failover(context.Background(), Target{Model: "gpt-6.1-sol"}, 429, true); ok {
		t.Fatalf("an image thread failed over to %+v, which cannot read images", next)
	}
}

func TestTheDefaultIsALadderStep(t *testing.T) {
	auto := config.Auto{DefaultModel: "glm-5.3", Candidates: []config.Candidate{
		{ID: "deepseek-flash", Vision: true}, {ID: "glm-5.3"}, {ID: "ui", Vision: true, Specialist: true},
		{ID: "gpt-6.1-sol", Vision: true},
	}}
	r := &Resolver{auto: auto, catalog: served()}
	if got := r.defaultFor(false); got != "glm-5.3" {
		t.Fatalf("default = %q", got)
	}
	if got := r.defaultFor(true); got != "gpt-6.1-sol" {
		t.Fatalf("default for images = %q, want the next vision ladder step, not the specialist", got)
	}
}

func TestPinKeysStayStable(t *testing.T) {
	// sha256("scope\x00claude-code\x00cc-1"): the key Claude Code threads
	// were pinned under while the router served other clients too.
	r := &Resolver{scope: "scope"}
	if got := r.key("cc-1", ""); got != "a3a69dfb3287dc4fe3f438224b06ceb8d77022869876c02c62b4187a120611b7" {
		t.Fatalf("key = %s", got)
	}
}
