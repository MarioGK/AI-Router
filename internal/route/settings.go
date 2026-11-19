package route

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
)

const (
	ModeAuto  = "auto"
	ModeFixed = "fixed"
)

// Settings controls subsequent virtual-model requests. A fixed request
// bypasses all deciders without changing the thread's automatic pin. Models
// restricts automatic routing; empty retains the complete configured policy.
type Settings = pin.RoutingSettings

func defaultSettings() Settings {
	return Settings{Mode: ModeAuto, JevEnabled: true}
}

// Settings reads the persisted policy, using the original automatic behavior
// only when no settings have been saved.
func (r *Resolver) Settings(ctx context.Context) (Settings, error) {
	if r.store == nil {
		return defaultSettings(), nil
	}
	stored, err := r.store.RoutingSettings(ctx)
	if err != nil {
		return Settings{}, err
	}
	if stored == nil {
		return defaultSettings(), nil
	}
	return *stored, nil
}

// SetSettings validates and persists a replacement, in configured ladder
// order. Its result has independent ownership of the selected model slice.
func (r *Resolver) SetSettings(ctx context.Context, settings Settings) (Settings, error) {
	settings, err := r.validateSettings(settings, false)
	if err != nil {
		return Settings{}, err
	}
	if r.store == nil {
		return Settings{}, fmt.Errorf("routing settings have no persistence store")
	}
	if err := r.store.SaveRoutingSettings(ctx, settings); err != nil {
		return Settings{}, err
	}
	return settings, nil
}

func (r *Resolver) validateSettings(settings Settings, activeOnly bool) (Settings, error) {
	if settings.Mode != ModeAuto && settings.Mode != ModeFixed {
		return Settings{}, fmt.Errorf("routing mode must be %q or %q", ModeAuto, ModeFixed)
	}
	if err := settings.Preferences.Validate(); err != nil {
		return Settings{}, err
	}
	if settings.Preferences.Priority == decide.PriorityAuto {
		settings.Preferences.Priority = ""
	}
	settings.Preferences.Ranking = slices.Clone(settings.Preferences.Ranking)
	for _, model := range settings.Preferences.Ranking {
		if (!activeOnly || settings.Mode != ModeFixed) && (r.auto.Rank(model) < 0 || r.auto.IsRetired(model)) {
			return Settings{}, fmt.Errorf("ranked model %q is not an active configured candidate", model)
		}
	}
	if len(settings.Models) > 128 {
		return Settings{}, fmt.Errorf("routing group has more than 128 entries")
	}
	valid := func(model string) bool { return r.auto.Rank(model) >= 0 && !r.auto.IsRetired(model) }
	if settings.FixedModel != "" && (!activeOnly || settings.Mode == ModeFixed) && !valid(settings.FixedModel) {
		return Settings{}, fmt.Errorf("fixed model %q is not an active configured candidate", settings.FixedModel)
	}
	if settings.Mode == ModeFixed && settings.FixedModel == "" {
		return Settings{}, fmt.Errorf("fixed mode needs a model")
	}
	if activeOnly && settings.Mode == ModeFixed {
		return settings, nil
	}
	for _, model := range settings.Models {
		if !valid(model) {
			return Settings{}, fmt.Errorf("routing model %q is not an active configured candidate", model)
		}
	}
	var selected []string
	for _, candidate := range r.auto.Candidates {
		if slices.Contains(settings.Models, candidate.ID) {
			selected = append(selected, candidate.ID)
		}
	}
	settings.Models = selected
	return settings, nil
}

// snapshot owns its settings and selected deciders while sharing immutable
// configuration, chooser health, singleflight and persistence with the service.
func (r *Resolver) snapshot(ctx context.Context) (*Resolver, error) {
	readCtx, cancel := context.WithTimeout(ctx, pinTimeout)
	settings, err := r.Settings(readCtx)
	cancel()
	if err != nil {
		return nil, err
	}
	// A previously used fixed model is inactive in automatic mode. Config
	// changes must not invalidate an otherwise valid automatic model group.
	if settings.Mode == ModeAuto && settings.FixedModel != "" &&
		(r.auto.Rank(settings.FixedModel) < 0 || r.auto.IsRetired(settings.FixedModel)) {
		settings.FixedModel = ""
	}
	settings, err = r.validateSettings(settings, true)
	if err != nil {
		return nil, err
	}
	view := &Resolver{
		auto: r.auto, scope: r.scope, store: r.store, flight: r.flight,
		chain: r.chain, state: r.state, catalog: r.catalog, quota: r.quota, load: r.load, log: r.log,
		lifetime: r.lifetime, policies: r.policies, controls: &settings, learner: r.learner, now: r.now,
		random: r.random,
	}
	if r.chain != nil {
		view.chooser = r.chain.Select(func(info decide.DeciderInfo) bool {
			return settings.Mode != ModeFixed && (settings.JevEnabled || !isJev(info))
		}).Prefer(settings.Preferences)
	}
	return view, nil
}

func isJev(info decide.DeciderInfo) bool {
	return info.Kind == decide.KindSystemOne && (info.Model == "jev" || strings.HasPrefix(info.Model, "jev-"))
}

func (r *Resolver) fixed() bool { return r.controls != nil && r.controls.Mode == ModeFixed }

func (r *Resolver) allowed(model string) bool {
	if r.controls == nil || r.controls.Mode == ModeAuto && len(r.controls.Models) == 0 {
		return true
	}
	if r.fixed() {
		return model == r.controls.FixedModel
	}
	return slices.Contains(r.controls.Models, model)
}

func (r *Resolver) excluded(model string) bool {
	return r.auto.IsRetired(model) || !r.allowed(model)
}

func (r *Resolver) capability(ctx context.Context, key string, images bool) (bool, error) {
	if key == "" {
		return images, nil
	}
	capCtx, cancel := context.WithTimeout(ctx, pinTimeout)
	defer cancel()
	required, err := r.store.ImagesRequired(capCtx, key)
	if err != nil {
		return false, err
	}
	if images && !required {
		if err := r.store.RequireImages(capCtx, key); err != nil {
			return false, err
		}
	}
	return images || required, nil
}

func (r *Resolver) auxFor(images bool) string {
	if !images && !r.excluded(r.auto.AuxModel) && r.usable(r.auto.AuxModel) {
		return r.auto.AuxModel
	}
	return r.defaultFor(images)
}

// fixedTarget is the target of a request in fixed mode. Its thread, for the
// request log, is the first of keys that is pinned: a subagent's own and
// then its session's.
func (r *Resolver) fixedTarget(ctx context.Context, images bool, keys ...string) Target {
	target := Target{Reason: ReasonFixed}
	for _, key := range keys {
		if key == "" {
			continue
		}
		p, err := r.store.Get(ctx, key)
		if err != nil {
			return Target{Reason: ReasonUnavailable}
		}
		if p != nil {
			target.Thread = key
			break
		}
	}
	model := r.controls.FixedModel
	if r.excluded(model) || !r.usable(model) || images && !r.sees(model) {
		target.Reason = ReasonUnavailable
		return target
	}
	target.Model = model
	return target
}

func (r *Resolver) selection() *decide.Selection {
	if r.chooser != nil {
		return r.chooser
	}
	return r.chain.Select(nil)
}
