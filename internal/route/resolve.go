// Package route turns a Claude Code request for the virtual model into the
// concrete model its thread is pinned to, and the reasoning level it runs
// at: it decides on a thread's first prompt and again on a new prompt where
// the thread's prompt cache is cold, keeps the pin otherwise, upgrades it
// once for vision, fails it over once, and moves it off a retired model.
package route

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/MarioGK/AI-Router/internal/catalog"
	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/thread"
	"github.com/MarioGK/AI-Router/internal/wire"
)

// Reasons a target was chosen, reported in logs and stats.
const (
	ReasonPinned      = "pinned"
	ReasonDecided     = "decided"
	ReasonRedecided   = "redecided"
	ReasonAuxiliary   = "auxiliary"
	ReasonNoSession   = "no_session"
	ReasonSide        = "side_request"
	ReasonStoreError  = "store_error"
	ReasonFailover    = "failover"
	ReasonUnavailable = "unavailable"
	ReasonFixed       = "fixed"
)

// Target is the model a request is sent to.
type Target struct {
	Model string
	// Level is the thread's reasoning level; empty when the chooser picked
	// none and the model runs at its fixed effort.
	Level string
	// Key is the pin key; empty when the request is not pinned.
	Key string
	// Thread is the key of the thread the request belongs to, for the
	// request log: the pin key, and also for side requests, which follow a
	// pin without being pinned requests.
	Thread     string
	FailedOver bool
	Reason     string
	// routing holds the immutable policy used to resolve this request, so
	// a settings change cannot alter its later failover.
	routing *Resolver
	// compacts is the pin key of the thread a compaction request belongs to
	// when it follows the thread's own pin; empty for any other request.
	compacts string
}

// Pinned reports whether the target belongs to a pinned thread.
func (t Target) Pinned() bool {
	return t.Key != ""
}

// Compacts reports whether the target is a compaction of its own pinned
// thread, which the caller reports to Compacted once it succeeded.
func (t Target) Compacts() bool {
	return t.compacts != ""
}

// Resolver resolves virtual-model requests.
type Resolver struct {
	auto     config.Auto
	scope    string
	store    *pin.Store
	flight   *pin.Flight
	chain    *decide.Chain
	state    *decide.StateBuilder
	catalog  *catalog.Catalog
	quota    Quota
	load     Load
	log      *slog.Logger
	lifetime context.Context
	cancel   context.CancelFunc
	// policies holds the fingerprints of the policies already recorded.
	policies *sync.Map
	outages  outages
	controls *Settings
	chooser  *decide.Selection
	// learner, when set, suggests from similar past prompts what a new
	// decision blends in.
	learner Learner
	now     func() time.Time
	// random draws whether a borderline decision is explored, uniformly
	// from [0, 1).
	random func() float64
}

// Learner suggests, from the labelled decisions whose first prompts read
// most like a new one, a prior over the offered candidates and levels,
// leaving out the thread's own decisions (key); *learn.Learner is one.
type Learner interface {
	Prior(ctx context.Context, prompt, key string, candidates, levels []string) *decide.Prior
}

// Quota tells whether any upstream account can serve a model;
// *accounts.Watcher is one.
type Quota interface {
	Available(model string) bool
}

// Load tells where a new thread decided for a busy model goes instead;
// *limit.Limiter is one.
type Load interface {
	Spill(model string) (string, bool)
}

// client is part of every pin key, so the keys of the threads pinned when
// the router served other clients too still match.
const client = "claude-code"

// maxPriorWait bounds the search for similar past prompts before a
// decision; it never takes more than a quarter of decide_timeout either.
const maxPriorWait = 2 * time.Second

// pinTimeout bounds each persistence phase independently of the chooser.
const pinTimeout = 5 * time.Second

var errUnavailable = errors.New("no configured model can serve this request")

// New returns a resolver. scope separates pins of different client keys;
// quota, when not nil, withholds models no account can serve; load, when
// not nil, spills new threads off saturated models.
func New(auto config.Auto, scope string, store *pin.Store, chain *decide.Chain, models *catalog.Catalog, quota Quota, load Load, log *slog.Logger) *Resolver {
	lifetime, cancel := context.WithCancel(context.Background())
	return &Resolver{
		auto: auto, scope: scope, store: store, chain: chain, catalog: models, quota: quota, load: load, log: log,
		state:    decide.NewStateBuilder(auto.StripTags, auto.StateMaxChars),
		lifetime: lifetime, cancel: cancel,
		flight: new(pin.Flight), policies: new(sync.Map), now: time.Now, random: rand.Float64,
	}
}

// Learn makes the resolver blend what the learner suggests into every
// decision.
func (r *Resolver) Learn(learner Learner) *Resolver {
	r.learner = learner
	return r
}

// Close cancels shared decisions and waits for them before the store closes.
func (r *Resolver) Close() {
	if r.cancel != nil {
		r.cancel()
	}
	if r.flight != nil {
		r.flight.Close()
	}
}

// Resolve returns the target for a request whose model is the virtual one.
func (r *Resolver) Resolve(ctx context.Context, kind wire.Kind, id thread.Identity, conversation wire.Conversation) Target {
	view, err := r.snapshot(ctx)
	if err != nil {
		r.log.Error("routing settings unavailable", "error", err)
		return Target{Reason: ReasonUnavailable}
	}
	target := view.resolve(ctx, kind, id, conversation)
	target.routing = view
	return target
}

func (r *Resolver) resolve(ctx context.Context, kind wire.Kind, id thread.Identity, conversation wire.Conversation) Target {
	// A subagent is a thread of its own, pinned under its session and agent
	// id; parent is then the pin key of its session's main thread.
	key, parent := "", ""
	if id.Session != "" {
		key = r.key(id.Session, id.Agent)
		if id.Agent != "" {
			parent = r.key(id.Session, "")
		}
	}
	images, err := r.capability(ctx, key, conversation.HasImage)
	if err != nil {
		r.log.Error("thread capability unavailable", "error", err)
		return Target{Thread: key, Reason: ReasonUnavailable}
	}
	conversation.HasImage = images
	if r.fixed() {
		target := r.fixedTarget(ctx, images, key, parent)
		// A compaction replaces the history of the thread's own automatic
		// pin in fixed mode too, which matters once automatic mode resumes.
		if kind != wire.CountTokens && id.Class == thread.Compaction && key != "" {
			if p, err := r.store.Get(ctx, key); err == nil && p != nil {
				target.compacts = key
			}
		}
		return target
	}
	switch {
	case kind == wire.CountTokens || id.Class == thread.Compaction:
		target := r.side(ctx, Target{Model: r.defaultFor(conversation.HasImage), Reason: ReasonSide},
			conversation.HasImage, key, parent)
		if kind != wire.CountTokens && target.Thread != "" && target.Thread == key {
			target.compacts = key
		}
		return target
	case id.Class == thread.Auxiliary:
		return Target{Model: r.auxFor(conversation.HasImage), Reason: ReasonAuxiliary}
	case key == "":
		return Target{Model: r.defaultFor(conversation.HasImage), Reason: ReasonNoSession}
	case id.Class == "" && !conversation.HasTools:
		// Claude Code's background calls (titles, summaries) offer no
		// tools, and carry no class without its gateway hint headers: they
		// must not make a thread's decision.
		model := r.auxFor(conversation.HasImage)
		return r.side(ctx, Target{Model: model, Reason: ReasonAuxiliary}, conversation.HasImage, key, parent)
	}
	reason := ReasonPinned
	p, err := r.store.Get(ctx, key)
	switch {
	case err == nil && p == nil:
		p, err = r.flight.DoContext(ctx, key, func() (*pin.Pin, error) {
			return r.decide(r.lifetime, key, parent, kind, id, conversation)
		})
		reason = ReasonDecided
	case err == nil && r.cold(p, conversation) != "":
		// Concurrent requests that find the cache cold share one
		// redecision, under a flight key of its own.
		before := *p
		p, err = r.flight.DoContext(ctx, "redecide\x00"+key, func() (*pin.Pin, error) {
			return r.redecide(r.lifetime, key, kind, id, conversation)
		})
		if err == nil && p != nil && (p.Model != before.Model || p.Level != before.Level) {
			reason = ReasonRedecided
		}
	}
	images, capErr := r.capability(ctx, key, conversation.HasImage)
	if capErr != nil {
		return Target{Thread: key, Reason: ReasonUnavailable}
	}
	conversation.HasImage = images
	if err == nil && p != nil && r.excluded(p.Model) {
		p, err = r.migrate(ctx, p, conversation.HasImage)
	}
	if err == nil && p != nil && (conversation.HasImage || p.Upgraded) {
		p, err = r.upgrade(ctx, p)
	}
	if err != nil || p == nil {
		if errors.Is(err, errUnavailable) {
			return Target{Thread: key, Reason: ReasonUnavailable}
		}
		r.log.Error("pin store failed; routing without a pin", "error", err)
		return Target{Model: r.defaultFor(conversation.HasImage), Reason: ReasonStoreError}
	}
	if r.excluded(p.Model) || (conversation.HasImage || p.Upgraded) && !r.sees(p.Model) {
		return Target{Thread: key, Reason: ReasonUnavailable}
	}
	if err := r.store.Seen(ctx, key); err != nil {
		r.log.Warn("pin touch failed", "error", err)
	}
	return Target{Model: p.Model, Level: p.Level, Key: key, Thread: key, FailedOver: p.FailedOver, Reason: reason}
}

// Compacted records that a compaction of its thread's own pin completed:
// it replaced the thread's history, so the thread's next request finds the
// prompt cache cold. A compaction that failed or was cut leaves the client
// with its old history and warm cache, and must not be recorded.
func (r *Resolver) Compacted(ctx context.Context, target Target) {
	if !target.Compacts() {
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), pinTimeout)
	defer cancel()
	if err := r.store.Compacted(ctx, target.compacts); err != nil {
		r.log.Warn("pin compaction mark failed", "error", err)
	}
}

// side is the target of a request that follows the pin of the first of
// keys that is pinned, a subagent's own and then its session's, without
// ever creating one; a thread without a pin gets unpinned.
func (r *Resolver) side(ctx context.Context, unpinned Target, images bool, keys ...string) Target {
	var p *pin.Pin
	for _, key := range keys {
		if key == "" {
			continue
		}
		var err error
		if p, err = r.store.Get(ctx, key); err != nil {
			return unpinned
		}
		if p != nil {
			break
		}
	}
	if p == nil {
		return unpinned
	}
	target := Target{Model: unpinned.Model, Level: p.Level, Thread: p.Key, Reason: ReasonSide}
	if !r.excluded(p.Model) && (!images || r.sees(p.Model)) {
		target.Model = p.Model
	}
	return target
}

// Answered records the status a model answered a request with and reports
// whether it fails the request's thread over: a status among the failover
// statuses does at once, except a server error, which does once the model
// answered failover_trip of them in a row. Until then the error reaches
// the client, which retries on the same model and account.
func (r *Resolver) Answered(model string, status int) bool {
	run := r.outages.answered(model, status)
	return slices.Contains(r.auto.FailoverStatuses, status) && (status < 500 || run >= *r.auto.FailoverTrip)
}

// Failover moves the target to its fallback model after the upstream
// failed with status. A pinned thread switches at most once, permanently;
// ok is false when no switch is allowed. A request carrying images only
// fails over to a model that can read them.
func (r *Resolver) Failover(ctx context.Context, target Target, status int, images bool) (Target, bool) {
	view := target.routing
	if view == nil {
		var err error
		view, err = r.snapshot(ctx)
		if err != nil {
			return target, false
		}
	}
	if view.fixed() {
		return target, false
	}
	next, ok := view.failover(ctx, target, status, images)
	next.routing = view
	return next, ok
}

func (r *Resolver) failover(ctx context.Context, target Target, status int, images bool) (Target, bool) {
	if target.FailedOver {
		return target, false
	}
	key := cmp.Or(target.Key, target.Thread)
	var err error
	images, err = r.capability(ctx, key, images)
	if err != nil {
		return target, false
	}
	for range 8 {
		if target.Pinned() {
			var err error
			images, err = r.capability(ctx, target.Key, images)
			if err != nil {
				return target, false
			}
			p, err := r.store.Get(ctx, target.Key)
			if err != nil || p == nil {
				r.log.Error("pin failover read failed", "error", err)
				return target, false
			}
			images = images || p.Upgraded
			if p.Model != target.Model {
				return r.followChanged(ctx, target, p, images)
			}
			if p.FailedOver {
				return target, false
			}
		}
		next := r.fallback(target.Model, images, "")
		if next == "" {
			return target, false
		}
		if !target.Pinned() {
			return Target{Model: next, Level: target.Level, Thread: target.Thread, Reason: ReasonFailover}, true
		}
		p, err := r.store.Failover(ctx, target.Key, target.Model, next, r.sees(next), pin.Event{
			Key: target.Key, Kind: pin.Failover, Model: next, Previous: target.Model,
			Detail: "upstream status " + strconv.Itoa(status),
		})
		if err != nil {
			r.log.Error("pin failover failed", "error", err)
			return target, false
		}
		if p == nil {
			return target, false
		}
		images, err = r.capability(ctx, target.Key, images)
		if err != nil {
			return target, false
		}
		if p.Model == target.Model {
			required, err := r.capability(ctx, target.Key, false)
			if err != nil {
				return target, false
			}
			if required && !p.FailedOver && !r.sees(next) {
				images = true
				continue
			}
			return target, false
		}
		images = images || p.Upgraded
		if r.excluded(p.Model) || images && !r.sees(p.Model) {
			return r.followChanged(ctx, target, p, images)
		}
		r.log.Warn("thread failed over", "pin", short(target.Key), "from", target.Model, "to", p.Model, "status", status)
		return Target{Model: p.Model, Level: p.Level, Key: target.Key, Thread: target.Key, FailedOver: p.FailedOver,
			Reason: ReasonFailover}, true
	}
	return target, false
}

// followChanged validates the pin that won a concurrent transition before
// retrying this request on it.
func (r *Resolver) followChanged(ctx context.Context, target Target, p *pin.Pin, images bool) (Target, bool) {
	images = images || p.Upgraded
	var err error
	images, err = r.capability(ctx, p.Key, images)
	if err != nil {
		return target, false
	}
	if r.excluded(p.Model) {
		p, err = r.migrate(ctx, p, images)
	}
	if err == nil && p != nil && images {
		p, err = r.upgrade(ctx, p)
	}
	if err != nil || p == nil || p.Model == target.Model || r.excluded(p.Model) || images && !r.sees(p.Model) {
		return target, false
	}
	return Target{Model: p.Model, Level: p.Level, Key: target.Key, Thread: target.Key,
		FailedOver: p.FailedOver, Reason: ReasonFailover}, true
}

// fallback is the model a target fails over to: its configured fallback
// when that is served and, for a request carrying images, able to read
// them; else, for such a request, the nearest other vision-capable
// candidate, and for any other the default model. It is empty when there
// is none.
func (r *Resolver) fallback(model string, images bool, exclude string) string {
	next, ok := r.auto.Fallback[model]
	ok = ok && next != exclude && !r.excluded(next) && r.usable(next)
	switch {
	case ok && (!images || r.sees(next)):
		return next
	case images:
		return r.visionTarget(r.auto.Rank(model), model, exclude)
	}
	if next := r.defaultFor(false); next != model && next != exclude && !r.excluded(next) && r.usable(next) {
		return next
	}
	return ""
}

// sees reports whether a model is a candidate that reads images.
func (r *Resolver) sees(model string) bool {
	candidate, ok := r.auto.Candidate(model)
	return ok && candidate.Vision
}

// decide decides a thread's first prompt and pins the thread; parent is the
// pin key of the session when the thread is a subagent's.
func (r *Resolver) decide(ctx context.Context, key, parent string, kind wire.Kind, id thread.Identity, conversation wire.Conversation) (*pin.Pin, error) {
	readCtx, cancel := context.WithTimeout(ctx, pinTimeout)
	p, err := r.store.Get(readCtx, key)
	cancel()
	if err != nil || p != nil {
		return p, err
	}
	event, policy := r.evaluateAmong(ctx, kind, id, conversation, r.candidates(conversation.HasImage), true)
	if event.Model == "" {
		return nil, errUnavailable
	}
	event.Key = key
	event.Basis.Parent = parent
	writeCtx, cancel := context.WithTimeout(ctx, pinTimeout)
	defer cancel()
	r.keep(writeCtx, &event, policy)
	r.log.Info("thread pinned", "pin", short(key), "model", event.Model,
		"level", event.Level, "decider", event.Decider, "choice", event.Choice, "probabilities", event.Probabilities,
		"escalated", event.Escalated, "latency_ms", event.Latency.Milliseconds(),
		"failures", decide.Failures(event.Attempts), "consulted", event.Consulted(), "policy", event.Basis.Policy,
		"parent", short(parent))
	return r.store.Create(writeCtx, pin.Pin{Key: key, Model: event.Model, Level: event.Level, Upgraded: conversation.HasImage,
		Parent: parent}, event)
}

// keep readies a decision for the log: without record_prompts it drops the
// prompts and the deciders' replies, and it records the policy the decision
// was made under, once per policy.
func (r *Resolver) keep(ctx context.Context, event *pin.Event, policy decide.Policy) {
	if !*r.auto.RecordPrompts {
		event.Prompt, event.FirstPrompt = "", ""
		for i := range event.Attempts {
			event.Attempts[i].Response, event.Attempts[i].Reason = "", ""
			if event.Attempts[i].Error != "" {
				event.Attempts[i].Error = "decider request failed"
			}
		}
	}
	if _, saved := r.policies.Load(event.Basis.Policy); !saved {
		if err := r.store.SavePolicy(ctx, policy); err != nil {
			r.log.Warn("policy record failed", "error", err)
		} else {
			r.policies.Store(event.Basis.Policy, true)
		}
	}
}

// cold is why a pinned thread is decided again on this request, or empty
// when it is not: the request is a new prompt from the user, not the next
// turn of a tool loop, and the thread's prompt cache is cold anyway, after
// the idle time without a request or after a compaction. With idle
// redecision off, only a thread that outgrew a pin below the top of the
// ladder is decided again, after a compaction.
func (r *Resolver) cold(p *pin.Pin, conversation wire.Conversation) string {
	if !conversation.Prompted || r.state.Build(conversation.LatestUser, false).Text == "" {
		return ""
	}
	idle, grow := r.auto.Redecide.IdleAfter(), r.auto.Redecide.GrowAfter()
	var why string
	switch gap := r.now().Sub(p.LastRequest); {
	case p.Compacted:
		why = "after a compaction"
	case idle > 0 && gap >= idle:
		why = "idle for " + gap.Truncate(time.Second).String()
	default:
		return ""
	}
	if grow > 0 && p.Turns >= grow && r.belowTop(p.Model, conversation.HasImage || p.Upgraded) {
		return why + ", " + strconv.Itoa(p.Turns) + " requests below the top of the ladder"
	}
	if idle == 0 {
		return ""
	}
	return why
}

// belowTop reports whether a model is a ladder step with a stronger step
// above it that the thread could be decided onto.
func (r *Resolver) belowTop(model string, images bool) bool {
	candidate, ok := r.auto.Candidate(model)
	if !ok || candidate.Specialist {
		return false
	}
	return slices.ContainsFunc(r.candidates(images), func(c decide.Candidate) bool {
		return !c.Specialist && r.auto.Rank(c.ID) > r.auto.Rank(model)
	})
}

// redecide decides a pinned thread again on its new prompt, when the shared
// flight still finds its prompt cache cold: among the candidates its images
// allow, other than the model it failed over from, up or down the ladder.
// A thread an operator re-pinned keeps its pin, and so does a thread whose
// new prompt no decider answered: an outage must not move it to the
// default model.
func (r *Resolver) redecide(ctx context.Context, key string, kind wire.Kind, id thread.Identity, conversation wire.Conversation) (*pin.Pin, error) {
	readCtx, cancel := context.WithTimeout(ctx, pinTimeout)
	p, err := r.store.Get(readCtx, key)
	overridden := false
	if err == nil && p != nil {
		overridden, err = r.store.Overridden(readCtx, key)
	}
	escalated := false
	if err == nil && p != nil && !overridden {
		escalated, err = r.escalated(readCtx, key)
	}
	cancel()
	if err != nil || p == nil || overridden {
		return p, err
	}
	why := r.cold(p, conversation)
	if why == "" {
		return p, nil
	}
	images := conversation.HasImage || p.Upgraded
	candidates := slices.DeleteFunc(r.candidates(images), func(c decide.Candidate) bool { return c.ID == p.FailedFrom })
	// The new prompt alone is judged, never an older one.
	conversation.HasImage, conversation.UserTexts = images, []string{conversation.LatestUser}
	// A thread that already moved up the ladder is no place to try a
	// cheaper step.
	event, policy := r.evaluateAmong(ctx, kind, id, conversation, candidates, !escalated)
	if event.Model == "" || event.Decider == config.DefaultDecider {
		r.log.Warn("thread kept its pin: no decider judged its new prompt", "pin", short(key), "cold", why)
		return p, nil
	}
	event.Kind, event.Key, event.Previous = pin.Redecision, key, p.Model
	event.Basis.Parent = p.Parent
	event.Detail = why + "; " + modelLevel(p.Model, p.Level) + " → " + modelLevel(event.Model, event.Level)
	writeCtx, cancel := context.WithTimeout(ctx, pinTimeout)
	defer cancel()
	r.keep(writeCtx, &event, policy)
	r.log.Info("thread decided again", "pin", short(key), "from", p.Model, "from_level", p.Level,
		"model", event.Model, "level", event.Level, "decider", event.Decider, "cold", why,
		"latency_ms", event.Latency.Milliseconds(), "failures", decide.Failures(event.Attempts))
	return r.store.Redecide(writeCtx, *p, event.Model, event.Level, r.sees(event.Model), event)
}

// modelLevel names a model and the reasoning level it runs at.
func modelLevel(model, level string) string {
	if level == "" {
		return model
	}
	return model + " (" + level + ")"
}

// Try decides a prompt the way a new thread's first request is decided,
// without pinning anything, and returns the decision with the policy it
// was made under.
func (r *Resolver) Try(ctx context.Context, prompt string, images bool) (pin.Event, decide.Policy) {
	view, err := r.snapshot(ctx)
	if err != nil {
		return pin.Event{Kind: pin.Decision}, r.chain.Select(func(decide.DeciderInfo) bool { return false }).Policy(nil)
	}
	return view.evaluate(ctx, wire.Messages, thread.Identity{UserAgent: "ai-router dashboard"},
		wire.Conversation{LatestUser: prompt, HasImage: images})
}

// Health is the decider chain's health.
func (r *Resolver) Health() []decide.DeciderHealth {
	return r.chain.Health()
}

// Policy is the policy a new thread is decided under.
func (r *Resolver) Policy(images bool) decide.Policy {
	ctx, cancel := context.WithTimeout(context.Background(), pinTimeout)
	defer cancel()
	view, err := r.snapshot(ctx)
	if err != nil {
		return r.chain.Select(func(decide.DeciderInfo) bool { return false }).Policy(nil)
	}
	return view.selection().Learned(view.prior(ctx, "", "", nil)).Policy(view.candidates(images))
}

// prior is what the history suggests for a prompt among the candidates;
// nil without a learner and in fixed mode, where the history decides
// nothing, so the policy advertised and the one a trial runs under agree.
func (r *Resolver) prior(ctx context.Context, prompt, key string, candidates []decide.Candidate) *decide.Prior {
	if r.learner == nil || r.fixed() {
		return nil
	}
	ids := make([]string, len(candidates))
	for i, candidate := range candidates {
		ids[i] = candidate.ID
	}
	return r.learner.Prior(ctx, prompt, key, ids, r.auto.Reasoning.IDs())
}

// evaluate runs the decider chain on a first request without pinning it,
// as a trial does, and returns the decision event, with the prompt as
// judged, and the policy in force. A trial never explores.
func (r *Resolver) evaluate(ctx context.Context, kind wire.Kind, id thread.Identity, conversation wire.Conversation) (pin.Event, decide.Policy) {
	return r.evaluateAmong(ctx, kind, id, conversation, r.candidates(conversation.HasImage), false)
}

// evaluateAmong runs the decider chain on the given candidates; with
// explore set, a borderline decision may be explored one step cheaper.
func (r *Resolver) evaluateAmong(ctx context.Context, kind wire.Kind, id thread.Identity, conversation wire.Conversation, candidates []decide.Candidate, explore bool) (pin.Event, decide.Policy) {
	state := r.state.Build(conversation.LatestUser, conversation.HasImage)
	if conversation.UserTexts != nil {
		state = r.state.BuildLatest(conversation.UserTexts, conversation.HasImage)
	}
	var decision decide.Decision
	var prior *decide.Prior
	selection := r.selection()
	decided := false
	if r.fixed() {
		decision.Model = r.fixedTarget(ctx, conversation.HasImage).Model
		decision.Decider = ReasonFixed
		decided = true
	} else if state.Text != "" && len(candidates) > 0 {
		key := ""
		if id.Session != "" {
			key = r.key(id.Session, id.Agent)
		}
		// The history has its own bound, and the deciders their whole
		// timeout after it: a slow search must not leave them no time.
		priorCtx, cancel := context.WithTimeout(ctx, min(maxPriorWait, r.auto.DecideTimeout/4))
		prior = r.prior(priorCtx, cmp.Or(state.Prompt, state.Text), key, candidates)
		cancel()
		selection = selection.Learned(prior)
		chainCtx, cancel := context.WithTimeout(ctx, r.auto.DecideTimeout)
		decision, decided = selection.Decide(chainCtx, state.Text, candidates)
		cancel()
	} else if !r.fixed() {
		// No prompt to search for: the policy still records the weights.
		selection = selection.Learned(r.prior(ctx, "", "", nil))
	}
	if !decided {
		decision.Model = r.defaultAmong(candidates)
		decision.Decider = config.DefaultDecider
	}
	offered := make([]string, len(candidates))
	for i, candidate := range candidates {
		offered[i] = candidate.ID
	}
	var exploration pin.Basis
	if explore && decided && !r.fixed() {
		decision.Model, exploration = r.explore(candidates, decision)
	}
	spilled := ""
	if !r.fixed() {
		if spill, ok := Spill(r.load, decision.Model, candidates); ok {
			spilled, decision.Model = decision.Model, spill
		}
	}
	var withheld []string
	for _, candidate := range r.auto.Candidates {
		if !r.usable(candidate.ID) {
			withheld = append(withheld, candidate.ID)
		}
	}
	policy := selection.Policy(candidates)
	// The prior is kept when it found neighbours: what they suggested,
	// and how much it weighed.
	if prior != nil && len(prior.Similar) == 0 {
		prior = nil
	}
	var costs *decide.Costs
	if len(decision.Costs.Done) > 0 {
		costs = &decision.Costs
	}
	return pin.Event{
		Kind: pin.Decision, Model: decision.Model,
		Decider: decision.Decider, Choice: decision.Choice, Probabilities: decision.Probabilities,
		Escalated: decision.Escalated, Level: decision.Level, Levels: decision.Levels, Latency: decision.Latency,
		Attempts: decision.Attempts, Prompt: state.Text, FirstPrompt: state.Prompt,
		Basis: pin.Basis{
			Kind: kind.String(), UserAgent: id.UserAgent, Candidates: offered, Withheld: withheld, Spilled: spilled,
			Costs: costs, Prior: prior,
			Threshold: r.auto.EscalationThreshold, Doubt: r.auto.Doubt, Images: conversation.HasImage,
			PromptChars: state.Chars, Truncated: state.Truncated, Policy: policy.Fingerprint(), Version: version,
			ExploreEligible: exploration.ExploreEligible, Explore: exploration.Explore,
			Propensity: exploration.Propensity, ExploredFrom: exploration.ExploredFrom,
		},
	}, policy
}

// explore routes a borderline automatic decision one ladder step cheaper
// with the probability auto.learn.explore, so that the outcomes show
// whether the cheaper step would have done. A decision is borderline when
// the step just below its model is one of the candidates (which already
// respect images, the model group, availability, retirement and the model
// a thread failed over from), would not spill, and the economy scored it
// within auto.learn.explore_margin of the model. A specialist, an xhigh
// decision and a decision the economy did not price are never explored.
// It returns the model to pin and what the basis records: nothing when the
// decision was not eligible, else the propensity of the action taken.
func (r *Resolver) explore(candidates []decide.Candidate, decision decide.Decision) (string, pin.Basis) {
	rate := r.auto.Learn.Explore
	if rate <= 0 || decision.Level == "xhigh" || r.chain == nil {
		return decision.Model, pin.Basis{}
	}
	lighter, excess, ok := r.selection().Lighter(candidates, decision)
	if !ok || excess > r.auto.Learn.Margin()+1e-12 || r.excluded(lighter) || !r.usable(lighter) {
		return decision.Model, pin.Basis{}
	}
	if _, spills := Spill(r.load, lighter, candidates); spills {
		return decision.Model, pin.Basis{}
	}
	if r.random() < rate {
		return lighter, pin.Basis{ExploreEligible: true, Explore: true, Propensity: rate, ExploredFrom: decision.Model}
	}
	return decision.Model, pin.Basis{ExploreEligible: true, Propensity: 1 - rate}
}

// escalated reports whether a thread ever moved up the ladder, by a
// redecision or an operator's re-pin.
func (r *Resolver) escalated(ctx context.Context, key string) (bool, error) {
	moves, err := r.store.Moves(ctx, key)
	if err != nil {
		return false, err
	}
	ladder := r.auto.Ladder()
	return slices.ContainsFunc(moves, func(e pin.Event) bool {
		from, to := slices.Index(ladder, e.Previous), slices.Index(ladder, e.Model)
		return from >= 0 && to > from
	}), nil
}

// Override re-pins a thread on an operator's word, to a candidate at a
// reasoning level (empty for the model's fixed effort), whatever the pin
// holds; its failover and capability upgrade stay as they were, so a
// thread upgraded for images only moves to a model that reads them.
func (r *Resolver) Override(ctx context.Context, key, model, level string) (*pin.Pin, error) {
	view, err := r.snapshot(ctx)
	if err != nil {
		return nil, err
	}
	if view.excluded(model) {
		return nil, fmt.Errorf("%q is outside the active routing model group", model)
	}
	return view.override(ctx, key, model, level)
}

func (r *Resolver) override(ctx context.Context, key, model, level string) (*pin.Pin, error) {
	switch {
	case r.auto.Rank(model) < 0:
		return nil, fmt.Errorf("%q is not a candidate", model)
	case level != "" && r.auto.Reasoning.Rank(level) < 0:
		return nil, fmt.Errorf("%q is not a reasoning level", level)
	}
	p, err := r.store.Get(ctx, key)
	if err != nil || p == nil {
		return nil, cmp.Or(err, fmt.Errorf("no thread is pinned under %s", short(key)))
	}
	images, err := r.capability(ctx, key, false)
	if err != nil {
		return nil, err
	}
	if images && !r.sees(model) {
		return nil, fmt.Errorf("the thread carries images, which %s cannot read", model)
	}
	r.log.Info("thread re-pinned by an operator", "pin", short(key), "from", p.Model, "to", model, "level", level)
	moved, err := r.store.Move(ctx, key, p.Model, model, level, r.sees(model), pin.Event{
		Key: key, Kind: pin.Override, Model: model, Level: level, Previous: p.Model, Detail: "re-pinned by an operator",
	})
	if err == nil && moved != nil {
		images, err = r.capability(ctx, key, false)
		if err != nil {
			return nil, err
		}
		if images && !r.sees(model) {
			return nil, fmt.Errorf("the thread carries images, which %s cannot read", model)
		}
		if r.excluded(moved.Model) || moved.Model != model || moved.Level != level {
			return nil, fmt.Errorf("the thread changed while overriding; try again")
		}
	}
	return moved, err
}

// migrate moves a pin off a retired model, keeping its level: to the
// model's fallback while one is configured and usable, else to the
// default.
func (r *Resolver) migrate(ctx context.Context, p *pin.Pin, images bool) (*pin.Pin, error) {
	images = images || p.Upgraded
	candidates := slices.DeleteFunc(r.candidates(images), func(c decide.Candidate) bool { return c.ID == p.FailedFrom })
	target := cmp.Or(r.fallback(p.Model, images, p.FailedFrom), r.defaultAmong(candidates))
	if target == "" {
		return p, nil
	}
	r.log.Info("thread moved off a retired model", "pin", short(p.Key), "from", p.Model, "to", target)
	detail := p.Model + " is retired"
	if !r.auto.IsRetired(p.Model) {
		detail = "routing model group changed"
	}
	return r.store.Move(ctx, p.Key, p.Model, target, p.Level, r.sees(target), pin.Event{
		Key: p.Key, Kind: pin.Migration, Model: target, Level: p.Level, Previous: p.Model,
		Detail: detail,
	})
}

// upgrade moves a pin whose model cannot read images to a vision-capable
// candidate, once: the nearest that costs no more (see upgradeTarget), then
// the nearest ladder step above, else the nearest below, else a
// specialist. A pin that failed over never goes back to the model it
// failed over from, which it could not leave again.
func (r *Resolver) upgrade(ctx context.Context, p *pin.Pin) (*pin.Pin, error) {
	// A competing transition may win the model comparison. Recheck that
	// winner so a successful image request always leaves a durable marker.
	for range 8 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if r.sees(p.Model) && p.Upgraded && !r.excluded(p.Model) {
			return p, nil
		}
		target := p.Model
		detail := "images; current model already supports them"
		if !r.sees(target) || r.excluded(target) {
			target = r.upgradeTarget(p.Model, p.FailedFrom)
			detail = "images"
		}
		if target == "" {
			return p, nil
		}
		var err error
		if p.Upgraded {
			p, err = r.store.Move(ctx, p.Key, p.Model, target, p.Level, true, pin.Event{
				Key: p.Key, Kind: pin.Migration, Model: target, Level: p.Level, Previous: p.Model,
				Detail: "restore image capability",
			})
		} else {
			p, err = r.store.Upgrade(ctx, p.Key, p.Model, target, pin.Event{
				Key: p.Key, Kind: pin.CapabilityUpgrade, Model: target, Previous: p.Model, Detail: detail,
			})
		}
		if err != nil || p == nil {
			return p, err
		}
	}
	return nil, fmt.Errorf("pin kept changing while recording image capability")
}

// upgradeTarget is the vision candidate a pin on model moves to for
// images, other than exclude: the nearest that costs no more per dollar of
// list price, so that images alone do not move an included-plan thread
// (GLM 5.3) to a paid tier when an included vision model (GLM 5.3 Flash)
// serves them; else the nearest vision candidate.
func (r *Resolver) upgradeTarget(model string, exclude string) string {
	ceiling := r.auto.CostFactor(model)
	var pricier []string
	for _, candidate := range r.auto.Candidates {
		if candidate.ID != model && r.auto.CostFactor(candidate.ID) > ceiling {
			pricier = append(pricier, candidate.ID)
		}
	}
	return cmp.Or(r.visionTarget(r.auto.Rank(model), append(pricier, exclude)...), r.visionTarget(r.auto.Rank(model), exclude))
}

// visionTarget is the vision-capable candidate nearest to the given rank,
// other than exclude: the nearest ladder step above, else the nearest
// below, else a specialist.
func (r *Resolver) visionTarget(rank int, exclude ...string) string {
	above, below, specialist := "", "", ""
	for _, candidate := range r.candidates(true) {
		c, _ := r.auto.Candidate(candidate.ID)
		if slices.Contains(exclude, c.ID) || !r.usable(c.ID) {
			continue
		}
		switch candidateRank := r.auto.Rank(c.ID); {
		case c.Specialist:
			specialist = cmp.Or(specialist, c.ID)
		case candidateRank > rank:
			above = cmp.Or(above, c.ID)
		case candidateRank < rank:
			below = c.ID
		}
	}
	return cmp.Or(above, below, specialist)
}

// usable reports whether CLIProxy serves a model and some account can.
func (r *Resolver) usable(model string) bool {
	return r.catalog.Has(model) && (r.quota == nil || r.quota.Available(model))
}

// candidates are the candidates restricted to usable models and, when the
// thread carries images, to vision-capable ones.
func (r *Resolver) candidates(hasImage bool) []decide.Candidate {
	eligible := slices.DeleteFunc(slices.Clone(r.auto.Candidates), func(c config.Candidate) bool {
		return r.excluded(c.ID) || hasImage && !c.Vision
	})
	served := slices.DeleteFunc(slices.Clone(eligible), func(c config.Candidate) bool { return !r.usable(c.ID) })
	candidates := make([]decide.Candidate, len(served))
	for i, c := range served {
		candidates[i] = c.Criterion()
	}
	if r.chain != nil {
		return r.selection().Candidates(candidates)
	}
	return candidates
}

// defaultFor is the default model for a request no decision covers.
func (r *Resolver) defaultFor(hasImage bool) string {
	return r.defaultAmong(r.candidates(hasImage))
}

// defaultAmong is the default model when it is a candidate, else the
// cheapest ladder candidate ranked above it, else the strongest ladder
// candidate; a specialist only when there is nothing else.
func (r *Resolver) defaultAmong(candidates []decide.Candidate) string {
	if len(candidates) == 0 {
		return ""
	}
	rank := r.auto.Rank(r.auto.DefaultModel)
	ladder := slices.DeleteFunc(slices.Clone(candidates), func(c decide.Candidate) bool { return c.Specialist })
	if len(ladder) == 0 {
		ladder = candidates
	}
	for _, c := range ladder {
		if r.auto.Rank(c.ID) >= rank {
			return c.ID
		}
	}
	return ladder[len(ladder)-1].ID
}

// Spill reports the destination of a saturated model only when both models
// were offered to this decision, including its capability restrictions.
func Spill(load Load, model string, offered []decide.Candidate) (string, bool) {
	if load == nil || !slices.ContainsFunc(offered, func(c decide.Candidate) bool { return c.ID == model }) {
		return "", false
	}
	spill, ok := load.Spill(model)
	if !ok || spill == model || !slices.ContainsFunc(offered, func(c decide.Candidate) bool { return c.ID == spill }) {
		return "", false
	}
	return spill, true
}

// key is the pin key of a session's main thread, or of one of its
// subagents when agent is set.
func (r *Resolver) key(session, agent string) string {
	name := session
	if agent != "" {
		name += "\x00agent\x00" + agent
	}
	sum := sha256.Sum256([]byte(r.scope + "\x00" + client + "\x00" + name))
	return hex.EncodeToString(sum[:])
}

func short(key string) string {
	return key[:min(12, len(key))]
}
