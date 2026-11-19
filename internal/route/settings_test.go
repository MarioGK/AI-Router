package route

import (
	"context"
	"log/slog"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/thread"
	"github.com/MarioGK/AI-Router/internal/wire"
)

func saveControls(t *testing.T, r *Resolver, settings Settings) Settings {
	t.Helper()
	got, err := r.SetSettings(t.Context(), settings)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func TestSettingsValidateNormalizeAndPersist(t *testing.T) {
	r := capabilityResolver(t)
	initial, err := r.Settings(t.Context())
	if err != nil || initial.Mode != ModeAuto || !initial.JevEnabled || len(initial.Models) != 0 {
		t.Fatalf("initial controls = %+v, %v", initial, err)
	}
	got := saveControls(t, r, Settings{Mode: ModeAuto, FixedModel: "gpt-6.1-sol",
		Models: []string{"claude-opus-5-5", "glm-5.3", "glm-5.3"}, JevEnabled: false})
	if !slices.Equal(got.Models, []string{"glm-5.3", "claude-opus-5-5"}) {
		t.Fatalf("canonical model group = %v", got.Models)
	}
	got.Models[0] = "deepseek-flash"
	persisted, err := r.Settings(t.Context())
	if err != nil || !slices.Equal(persisted.Models, []string{"glm-5.3", "claude-opus-5-5"}) {
		t.Fatalf("returned settings mutation persisted = %+v, %v", persisted, err)
	}
	r.auto.Retired = []string{"deepseek-flash"}
	for _, invalid := range []Settings{
		{Mode: "unknown"},
		{Mode: ModeFixed},
		{Mode: ModeFixed, FixedModel: "unknown"},
		{Mode: ModeAuto, FixedModel: "unknown"},
		{Mode: ModeAuto, Models: []string{"unknown"}},
		{Mode: ModeAuto, Models: []string{"deepseek-flash"}},
		{Mode: ModeAuto, Models: make([]string, 129)},
		{Mode: ModeAuto, Preferences: decide.Preferences{Priority: "unknown"}},
		{Mode: ModeAuto, Preferences: decide.Preferences{Ranking: []string{"unknown"}}},
		{Mode: ModeAuto, Preferences: decide.Preferences{Ranking: []string{"glm-5.3", "glm-5.3"}}},
	} {
		if _, err := r.SetSettings(t.Context(), invalid); err == nil {
			t.Errorf("invalid settings accepted = %+v", invalid)
		}
	}
	again, err := r.Settings(t.Context())
	if err != nil || !reflect.DeepEqual(again, persisted) {
		t.Fatalf("rejected setting changed persistence = %+v, %v", again, err)
	}
}

func TestFixedModePreservesAutomaticPinAndAllSideKinds(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "glm-5.3", false)
	before, err := r.store.Get(t.Context(), seed.Key)
	if err != nil {
		t.Fatal(err)
	}
	saveControls(t, r, Settings{Mode: ModeFixed, FixedModel: "gpt-6.1-sol"})
	for _, request := range []struct {
		kind  wire.Kind
		class string
	}{
		{wire.Messages, "interactive"}, {wire.Messages, thread.Auxiliary},
		{wire.Messages, thread.Compaction}, {wire.CountTokens, ""}, {wire.Messages, ""},
	} {
		id := capabilityID()
		id.Class = request.class
		got := r.Resolve(t.Context(), request.kind, id, wire.Conversation{})
		if got.Model != "gpt-6.1-sol" || got.Pinned() || got.Level != "" || got.Thread != seed.Key || got.Reason != ReasonFixed {
			t.Fatalf("fixed request = %+v", got)
		}
		if _, ok := r.Failover(t.Context(), got, 429, false); ok {
			t.Fatal("fixed request failed over")
		}
	}
	after, err := r.store.Get(t.Context(), seed.Key)
	if err != nil || !reflect.DeepEqual(after, before) {
		t.Fatalf("fixed requests changed automatic pin = %+v, %v", after, err)
	}
	saveControls(t, r, Settings{Mode: ModeAuto, JevEnabled: true})
	if got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{}); got.Model != seed.Model || got.Key != seed.Key {
		t.Fatalf("restored automatic mode lost original pin = %+v", got)
	}
}

func TestFixedOnlyThreadHasCapabilityWithoutPhantomPinLink(t *testing.T) {
	r := capabilityResolver(t)
	saveControls(t, r, Settings{Mode: ModeFixed, FixedModel: "gpt-6.1-sol"})
	got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{HasImage: true})
	if got.Model != "gpt-6.1-sol" || got.Thread != "" || got.Pinned() {
		t.Fatalf("fixed-only request = %+v", got)
	}
	key := r.key(capabilityID().Session, "")
	if required, err := r.store.ImagesRequired(t.Context(), key); err != nil || !required {
		t.Fatalf("fixed-only image requirement = %v, %v", required, err)
	}
	if p, err := r.store.Get(t.Context(), key); err != nil || p != nil {
		t.Fatalf("fixed-only image created automatic pin = %+v, %v", p, err)
	}
	saveControls(t, r, Settings{Mode: ModeAuto, Models: []string{"glm-5.3"}, JevEnabled: true})
	if got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{}); got.Model != "" || got.Reason != ReasonUnavailable {
		t.Fatalf("automatic mode forgot fixed images = %+v", got)
	}
}

func TestFixedModeRejectsExistingDurableImageRequirement(t *testing.T) {
	r := capabilityResolver(t)
	capabilitySeed(t, r, "gpt-6.1-sol", true)
	saveControls(t, r, Settings{Mode: ModeFixed, FixedModel: "glm-5.3"})
	if got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{}); got.Model != "" || got.Reason != ReasonUnavailable {
		t.Fatalf("fixed text model ignored durable images = %+v", got)
	}
}

func TestAutomaticGroupMigratesPinsWithoutChangingReasoning(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "gpt-6.1-sol", true)
	if _, err := r.store.Move(t.Context(), seed.Key, seed.Model, seed.Model, "high", true,
		pin.Event{Key: seed.Key, Kind: pin.Override, Model: seed.Model, Level: "high"}); err != nil {
		t.Fatal(err)
	}
	saveControls(t, r, Settings{Mode: ModeAuto, Models: []string{"glm-5.3", "claude-opus-5-5"}, JevEnabled: true})
	got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{})
	if got.Model != "claude-opus-5-5" || got.Level != "high" || got.Key != seed.Key {
		t.Fatalf("model group migration lost capability or reasoning = %+v", got)
	}
	if _, err := r.Override(t.Context(), seed.Key, "gpt-6.1-sol", ""); err == nil {
		t.Fatal("override escaped selected pool")
	}
}

func TestTargetKeepsSettingsSnapshotAcrossFailover(t *testing.T) {
	r := capabilityResolver(t)
	capabilitySeed(t, r, "gpt-6.1-sol", false)
	saveControls(t, r, Settings{Mode: ModeAuto, Models: []string{"glm-5.3", "gpt-6.1-sol"}, JevEnabled: true})
	old := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{})
	copyOfOld := old
	saveControls(t, r, Settings{Mode: ModeAuto, Models: []string{"claude-opus-5-5"}, JevEnabled: false})
	next, ok := r.Failover(t.Context(), copyOfOld, 429, false)
	if !ok || next.Model != "glm-5.3" {
		t.Fatalf("in-flight request mixed new settings = %+v, %v", next, ok)
	}
	if got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{}); got.Model != "claude-opus-5-5" {
		t.Fatalf("subsequent request escaped new pool = %+v", got)
	}
}

func TestFixedTargetStillCannotFailoverAfterAutomaticModeReturns(t *testing.T) {
	r := capabilityResolver(t)
	saveControls(t, r, Settings{Mode: ModeFixed, FixedModel: "gpt-6.1-sol"})
	old := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{})
	saveControls(t, r, Settings{Mode: ModeAuto, JevEnabled: true})
	if next, ok := r.Failover(t.Context(), old, 429, false); ok || next.Model != old.Model {
		t.Fatalf("old fixed request changed mode during failover = %+v, %v", next, ok)
	}
}

type settingsDecider struct {
	info   decide.DeciderInfo
	choice string
	calls  atomic.Int32
}

func (d *settingsDecider) Name() string             { return d.info.Name }
func (d *settingsDecider) Info() decide.DeciderInfo { return d.info }
func (d *settingsDecider) Decide(context.Context, string, []decide.Candidate, []decide.Level) (decide.Verdict, error) {
	d.calls.Add(1)
	return decide.Verdict{Choice: d.choice, Probabilities: map[string]float64{d.choice: 1}}, nil
}

func TestJevToggleRetainsOtherSystemOneAndLLMDeciders(t *testing.T) {
	for _, retainSystemOne := range []bool{true, false} {
		t.Run(map[bool]string{true: "other systemone", false: "llm fallback"}[retainSystemOne], func(t *testing.T) {
			r := capabilityResolver(t)
			jev := &settingsDecider{info: decide.DeciderInfo{Name: "first", Kind: decide.KindSystemOne, Model: "jev-1.13"}, choice: "deepseek-flash"}
			other := &settingsDecider{info: decide.DeciderInfo{Name: "other", Kind: decide.KindSystemOne, Model: "classifier-v1"}, choice: "glm-5.3"}
			judge := &settingsDecider{info: decide.DeciderInfo{Name: "fallback", Kind: decide.KindLLM, Model: "judge"}, choice: "claude-opus-5-5"}
			steps := []decide.Step{{Decider: jev, Timeout: time.Second}}
			if retainSystemOne {
				steps = append(steps, decide.Step{Decider: other, Timeout: time.Second})
			}
			steps = append(steps, decide.Step{Decider: judge, Timeout: time.Second})
			r.chain = decide.NewChain(steps, 0.9, decide.DoubtEscalate, nil)
			saveControls(t, r, Settings{Mode: ModeAuto, JevEnabled: false})
			got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{LatestUser: "route this task", HasTools: true})
			want := "claude-opus-5-5"
			if retainSystemOne {
				want = "glm-5.3"
			}
			if got.Model != want || jev.calls.Load() != 0 {
				t.Fatalf("Jev toggle = %+v; Jev calls %d", got, jev.calls.Load())
			}
			for _, info := range r.Policy(false).Deciders {
				if isJev(info) {
					t.Fatalf("disabled Jev present in runtime policy = %+v", info)
				}
			}
		})
	}
}

func TestAutomaticModeIgnoresRetiredInactiveFixedChoiceWithoutBroadeningGroup(t *testing.T) {
	r := capabilityResolver(t)
	saveControls(t, r, Settings{Mode: ModeAuto, FixedModel: "gpt-6.1-sol", Models: []string{"glm-5.3"}, JevEnabled: true})
	r.auto.Retired = []string{"gpt-6.1-sol"}
	got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{})
	if got.Model != "glm-5.3" || len(r.Policy(false).Candidates) != 1 {
		t.Fatalf("inactive retired fixed choice broke automatic mode = %+v", got)
	}
	if err := r.store.SaveRoutingSettings(t.Context(), Settings{Mode: ModeAuto, Models: []string{"gpt-6.1-sol"}, JevEnabled: true}); err != nil {
		t.Fatal(err)
	}
	if got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{}); got.Model != "" || got.Reason != ReasonUnavailable {
		t.Fatalf("invalid active group was silently broadened = %+v", got)
	}
}

func TestFixedModeIgnoresRetiredInactiveGroupWithoutChangingSavedGroup(t *testing.T) {
	r := capabilityResolver(t)
	saveControls(t, r, Settings{Mode: ModeFixed, FixedModel: "claude-opus-5-5", Models: []string{"gpt-6.1-sol"}, JevEnabled: true})
	r.auto.Retired = []string{"gpt-6.1-sol"}
	if got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{}); got.Model != "claude-opus-5-5" || got.Reason != ReasonFixed {
		t.Fatalf("inactive retired group broke fixed mode = %+v", got)
	}
	saved, err := r.Settings(t.Context())
	if err != nil || !slices.Equal(saved.Models, []string{"gpt-6.1-sol"}) {
		t.Fatalf("inactive group was rewritten = %+v, %v", saved, err)
	}
	saved.Mode = ModeAuto
	if _, err := r.SetSettings(t.Context(), saved); err == nil {
		t.Fatal("inactive retired group was allowed to become active")
	}
}

func TestFailoverRechecksIndependentRequirementOnConcurrentWinner(t *testing.T) {
	for _, changedModel := range []bool{false, true} {
		t.Run(map[bool]string{false: "same model", true: "changed model"}[changedModel], func(t *testing.T) {
			r := capabilityResolver(t)
			seed := capabilitySeed(t, r, "gpt-6.1-sol", false)
			change := sync.OnceFunc(func() {
				if changedModel {
					if _, err := r.store.Failover(t.Context(), seed.Key, seed.Model, "glm-5.3", false,
						pin.Event{Key: seed.Key, Kind: pin.Failover, Model: "glm-5.3", Previous: seed.Model}); err != nil {
						t.Fatal(err)
					}
				}
				if err := r.store.RequireImages(t.Context(), seed.Key); err != nil {
					t.Fatal(err)
				}
			})
			r.quota = hookQuota(func(model string) bool {
				if model == "glm-5.3" {
					change()
				}
				return true
			})
			got, ok := r.Failover(t.Context(), seed, 429, false)
			if !ok || !r.sees(got.Model) || !got.FailedOver {
				t.Fatalf("independent image observation lost in CAS = %+v, %v", got, ok)
			}
		})
	}
}

type settingsLogHook struct{ callback func() }

func (h settingsLogHook) Enabled(context.Context, slog.Level) bool { return true }
func (h settingsLogHook) Handle(_ context.Context, record slog.Record) error {
	if record.Message == "thread re-pinned by an operator" {
		h.callback()
	}
	return nil
}
func (h settingsLogHook) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h settingsLogHook) WithGroup(string) slog.Handler      { return h }

func TestOverrideRejectsUnexpectedConcurrentWinnerOutsideGroup(t *testing.T) {
	r := capabilityResolver(t)
	seed := capabilitySeed(t, r, "deepseek-flash", false)
	saveControls(t, r, Settings{Mode: ModeAuto, Models: []string{"glm-5.3"}, JevEnabled: true})
	r.log = slog.New(settingsLogHook{callback: sync.OnceFunc(func() {
		if _, err := r.store.Move(t.Context(), seed.Key, seed.Model, "gpt-6.1-sol", "", true,
			pin.Event{Key: seed.Key, Kind: pin.Migration, Model: "gpt-6.1-sol", Previous: seed.Model}); err != nil {
			t.Fatal(err)
		}
	})})
	if p, err := r.Override(t.Context(), seed.Key, "glm-5.3", ""); err == nil {
		t.Fatalf("override accepted concurrent outside-group pin = %+v", p)
	}
}

func TestSettingsReadFailureNeverRoutesUsingDefaultGroup(t *testing.T) {
	r := capabilityResolver(t)
	saveControls(t, r, Settings{Mode: ModeFixed, FixedModel: "gpt-6.1-sol"})
	if err := r.store.Close(); err != nil {
		t.Fatal(err)
	}
	if got := r.Resolve(t.Context(), wire.Messages, capabilityID(), wire.Conversation{}); got.Model != "" || got.Reason != ReasonUnavailable {
		t.Fatalf("settings failure routed with defaults = %+v", got)
	}
	if event, policy := r.Try(t.Context(), "test", false); event.Model != "" || len(policy.Candidates) != 0 {
		t.Fatalf("settings failure exposed automatic trial = %+v, %+v", event, policy)
	}
}
