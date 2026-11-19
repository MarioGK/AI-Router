package route

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/thread"
	"github.com/MarioGK/AI-Router/internal/wire"
)

func capabilityResolver(t *testing.T) *Resolver {
	t.Helper()
	store, err := pin.Open(filepath.Join(t.TempDir(), "router.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	auto := config.Auto{
		DefaultModel: "glm-5.3", AuxModel: "aux", DecideTimeout: time.Second,
		StateMaxChars: 1000, EscalationThreshold: 0.9, RecordPrompts: new(false),
		Candidates: []config.Candidate{
			{ID: "deepseek-flash", Vision: true}, {ID: "glm-5.3"},
			{ID: "gpt-6.1-sol", Vision: true}, {ID: "claude-opus-5-5", Vision: true},
		},
		Fallback: map[string]string{"gpt-6.1-sol": "glm-5.3"},
	}
	r := New(auto, "test", store, decide.NewChain(nil, 0.9, decide.DoubtEscalate, nil), served(), nil, nil,
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Cleanup(r.Close)
	return r
}

func capabilitySeed(t *testing.T, r *Resolver, model string, images bool) Target {
	t.Helper()
	key := r.key("capability-thread", "")
	if _, err := r.store.Create(t.Context(), pin.Pin{Key: key, Model: model, Upgraded: images},
		pin.Event{Key: key, Kind: pin.Decision, Model: model}); err != nil {
		t.Fatal(err)
	}
	return Target{Model: model, Key: key, Thread: key}
}

func capabilityID() thread.Identity {
	return thread.Identity{Session: "capability-thread", Class: "interactive"}
}

func TestImageCandidatesNeverRestoreUnavailableOrTextModels(t *testing.T) {
	r := capabilityResolver(t)
	r.quota = fakeQuota{"deepseek-flash": true, "gpt-6.1-sol": true, "claude-opus-5-5": true}
	if got := r.candidates(true); len(got) != 0 {
		t.Fatalf("unavailable vision candidates = %+v", got)
	}
	if got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{HasImage: true}); got.Model != "" || got.Pinned() || got.Reason != ReasonUnavailable {
		t.Fatalf("unavailable image request = %+v", got)
	}
	if p, err := r.store.Get(t.Context(), r.key(capabilityID().Session, "")); err != nil || p != nil {
		t.Fatalf("unavailable request created pin = %+v, %v", p, err)
	}
	if next, ok := r.Failover(t.Context(), Target{Model: "gpt-6.1-sol"}, 429, true); ok {
		t.Fatalf("unavailable vision fallback = %+v", next)
	}
}

func TestExistingPinSurvivesUnavailableNewCandidates(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "previous-model", false)
	r.quota = fakeQuota{"deepseek-flash": true, "glm-5.3": true, "gpt-6.1-sol": true, "claude-opus-5-5": true}
	got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{})
	if got.Model != seed.Model || got.Key != seed.Key || got.Reason != ReasonPinned {
		t.Fatalf("valid existing pin was withheld = %+v", got)
	}
}

func TestImageFirstPinKeepsCapabilityThroughTextFailover(t *testing.T) {
	r := capabilityResolver(t)
	first := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{HasImage: true})
	p, err := r.store.Get(t.Context(), first.Key)
	if err != nil || p == nil || !p.Upgraded || !r.sees(p.Model) {
		t.Fatalf("image-first pin = %+v, %v", p, err)
	}
	if _, err := r.Override(t.Context(), first.Key, "glm-5.3", ""); err == nil {
		t.Fatal("operator moved an image-first pin to a text-only model")
	}
	next, ok := r.Failover(t.Context(), first, 429, false)
	if !ok || !r.sees(next.Model) || !next.FailedOver {
		t.Fatalf("text failover lost image capability = %+v, %v", next, ok)
	}
	if _, ok := r.Failover(t.Context(), next, 429, false); ok {
		t.Fatal("image pin spent failover twice")
	}
}

func TestAlreadyVisionPinRecordsFirstImageDurably(t *testing.T) {
	r := capabilityResolver(t)
	capabilitySeed(t, r, "gpt-6.1-sol", false)
	first := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{HasImage: true})
	p, err := r.store.Get(t.Context(), first.Key)
	if err != nil || p == nil || !p.Upgraded || p.Model != "gpt-6.1-sol" {
		t.Fatalf("first image on an existing vision pin = %+v, %v", p, err)
	}
	next, ok := r.Failover(t.Context(), first, 429, false)
	if !ok || !r.sees(next.Model) {
		t.Fatalf("text failover after first image = %+v, %v", next, ok)
	}
}

func TestUpgradedPinKeepsVisionThroughTextMigration(t *testing.T) {
	r := capabilityResolver(t)
	capabilitySeed(t, r, "retired-vision", true)
	r.auto.Retired = []string{"retired-vision"}
	r.auto.Fallback["retired-vision"] = "glm-5.3"
	got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{})
	if !got.Pinned() || !r.sees(got.Model) {
		t.Fatalf("migration lost durable image capability = %+v", got)
	}
	p, err := r.store.Get(t.Context(), got.Key)
	if err != nil || p == nil || !p.Upgraded {
		t.Fatalf("migration cleared image marker = %+v, %v", p, err)
	}
}

func TestInvalidUpgradedPinRepairsWithoutAnotherUpgrade(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "glm-5.3", true)
	got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{})
	if !r.sees(got.Model) {
		t.Fatalf("legacy capability corruption was not repaired = %+v", got)
	}
	detail, err := r.store.Thread(t.Context(), seed.Key)
	if err != nil || len(detail.Events) != 2 || detail.Events[1].Kind != pin.Migration {
		t.Fatalf("repair event trail = %+v, %v", detail, err)
	}
}

func TestSideImagesNeverFollowTextOnlyPin(t *testing.T) {
	for _, request := range []struct {
		name  string
		kind  wire.Kind
		class string
	}{
		{"compaction", wire.Messages, thread.Compaction},
		{"count tokens", wire.CountTokens, ""},
		{"background", wire.Messages, ""},
		{"auxiliary", wire.Messages, thread.Auxiliary},
	} {
		t.Run(request.name, func(t *testing.T) {
			r := capabilityResolver(t)
			seed := capabilitySeed(t, r, "glm-5.3", false)
			id := capabilityID()
			id.Class = request.class
			got := r.Resolve(t.Context(), request.kind, id, wire.Conversation{HasImage: true})
			if !r.sees(got.Model) || got.Pinned() {
				t.Fatalf("side image request = %+v", got)
			}
			p, err := r.store.Get(t.Context(), seed.Key)
			if err != nil || p == nil || p.Model != seed.Model || p.Upgraded {
				t.Fatalf("side request changed pin = %+v, %v", p, err)
			}
		})
	}
}

func TestCapabilityUpgradeRechecksConcurrentWinner(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "glm-5.3", false)
	stale, err := r.store.Get(t.Context(), seed.Key)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.store.Failover(t.Context(), seed.Key, seed.Model, "gpt-6.1-sol", true,
		pin.Event{Key: seed.Key, Kind: pin.Failover, Model: "gpt-6.1-sol", Previous: seed.Model}); err != nil {
		t.Fatal(err)
	}
	p, err := r.upgrade(t.Context(), stale)
	if err != nil || p == nil || !p.Upgraded || !r.sees(p.Model) {
		t.Fatalf("concurrent vision winner did not record images = %+v, %v", p, err)
	}
}

type hookQuota func(string) bool

func (q hookQuota) Available(model string) bool { return q(model) }

func TestFailoverRevalidatesConcurrentTextWinner(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "gpt-6.1-sol", false)
	change := sync.OnceFunc(func() {
		if _, err := r.store.Failover(t.Context(), seed.Key, seed.Model, "glm-5.3", false,
			pin.Event{Key: seed.Key, Kind: pin.Failover, Model: "glm-5.3", Previous: seed.Model}); err != nil {
			t.Fatal(err)
		}
	})
	r.quota = hookQuota(func(model string) bool {
		if model == "glm-5.3" {
			change()
		}
		return true
	})
	got, ok := r.Failover(t.Context(), seed, 429, true)
	if !ok || !r.sees(got.Model) || !got.FailedOver {
		t.Fatalf("image failover accepted incompatible winner = %+v, %v", got, ok)
	}
	p, err := r.store.Get(t.Context(), seed.Key)
	required, capErr := r.store.ImagesRequired(t.Context(), seed.Key)
	if err != nil || capErr != nil || p == nil || !required || !r.sees(p.Model) {
		t.Fatalf("concurrent winner image requirement not durable = %+v, %v", p, err)
	}
}

func TestFailoverRetriesAfterConcurrentSameModelImageMarker(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "gpt-6.1-sol", false)
	mark := sync.OnceFunc(func() {
		if _, err := r.store.Upgrade(t.Context(), seed.Key, seed.Model, seed.Model,
			pin.Event{Key: seed.Key, Kind: pin.CapabilityUpgrade, Model: seed.Model, Previous: seed.Model}); err != nil {
			t.Fatal(err)
		}
	})
	r.quota = hookQuota(func(model string) bool {
		if model == "glm-5.3" {
			mark()
		}
		return true
	})
	got, ok := r.Failover(t.Context(), seed, 429, false)
	if !ok || !r.sees(got.Model) || !got.FailedOver {
		t.Fatalf("same-model capability race stopped compatible failover = %+v, %v", got, ok)
	}
}

func TestNoConfiguredVisionModelCreatesNoPin(t *testing.T) {
	r := capabilityResolver(t)
	r.auto.Candidates = []config.Candidate{{ID: "glm-5.3"}}
	got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{HasImage: true})
	if got.Model != "" || got.Pinned() || got.Reason != ReasonUnavailable {
		t.Fatalf("request without any configured vision model = %+v", got)
	}
	if p, err := r.store.Get(t.Context(), r.key(capabilityID().Session, "")); err != nil || p != nil {
		t.Fatalf("unsupported image request created pin = %+v, %v", p, err)
	}
}

func TestMigrationDoesNotReturnToFailedSource(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "gpt-6.1-sol", false)
	if _, err := r.store.Failover(t.Context(), seed.Key, seed.Model, "glm-5.3", false,
		pin.Event{Key: seed.Key, Kind: pin.Failover, Model: "glm-5.3", Previous: seed.Model}); err != nil {
		t.Fatal(err)
	}
	r.auto.Retired = []string{"glm-5.3"}
	r.auto.DefaultModel = "gpt-6.1-sol"
	r.auto.Fallback["glm-5.3"] = "gpt-6.1-sol"
	got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{})
	if !got.Pinned() || got.Model == seed.Model || got.Model == "glm-5.3" || !got.FailedOver {
		t.Fatalf("migration returned to exhausted failover source = %+v", got)
	}
}

func TestImagesNeverReturnToTheOnlyExhaustedVisionSource(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "gpt-6.1-sol", false)
	if _, err := r.store.Failover(t.Context(), seed.Key, seed.Model, "glm-5.3", false,
		pin.Event{Key: seed.Key, Kind: pin.Failover, Model: "glm-5.3", Previous: seed.Model}); err != nil {
		t.Fatal(err)
	}
	r.auto.Candidates = r.auto.Candidates[1:3]
	got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{HasImage: true})
	if got.Model != "" || got.Reason != ReasonUnavailable {
		t.Fatalf("request returned to exhausted vision source = %+v", got)
	}
	p, err := r.store.Get(t.Context(), seed.Key)
	if err != nil || p == nil || p.Model != "glm-5.3" || !p.FailedOver || p.FailedFrom != seed.Model {
		t.Fatalf("unsupported image request changed spent pin = %+v, %v", p, err)
	}
}

type privacyDecider struct{ fail bool }

func (d privacyDecider) Name() string { return "privacy" }
func (d privacyDecider) Info() decide.DeciderInfo {
	return decide.DeciderInfo{Name: d.Name(), Kind: decide.KindSystemOne, Model: "test"}
}
func (d privacyDecider) Decide(context.Context, string, []decide.Candidate, []decide.Level) (decide.Verdict, error) {
	v := decide.Verdict{Choice: "glm-5.3", Probabilities: map[string]float64{"glm-5.3": 1},
		Response: "private-task secret-token", Reason: "private-task secret-token", Digest: "safe-hash", InputTokens: 5}
	if d.fail {
		return v, errors.New("private-task secret-token")
	}
	return v, nil
}

func TestRecordPromptsFalseRedactsUntrustedAttemptText(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "failure"}[fail], func(t *testing.T) {
			r := capabilityResolver(t)
			r.chain = decide.NewChain([]decide.Step{{Decider: privacyDecider{fail: fail}, Timeout: time.Second}},
				0.9, decide.DoubtEscalate, nil)
			got := r.Resolve(t.Context(), wire.Messages, capabilityID(),
				wire.Conversation{LatestUser: "private-task secret-token", HasTools: true})
			detail, err := r.store.Thread(t.Context(), got.Key)
			if err != nil || detail == nil || len(detail.Events) != 1 {
				t.Fatalf("decision trail = %+v, %v", detail, err)
			}
			event := detail.Events[0]
			if event.Prompt != "" || event.FirstPrompt != "" || len(event.Attempts) != 1 {
				t.Fatalf("private prompt persisted = %+v", event)
			}
			attempt := event.Attempts[0]
			if attempt.Response != "" || attempt.Reason != "" || strings.Contains(attempt.Error, "secret-token") || attempt.Digest != "safe-hash" {
				t.Fatalf("untrusted attempt text persisted = %+v", attempt)
			}
		})
	}
}
