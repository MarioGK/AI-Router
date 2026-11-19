// Package app wires the router together and serves it.
package app

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/MarioGK/AI-Router/internal/accounts"
	"github.com/MarioGK/AI-Router/internal/auth"
	"github.com/MarioGK/AI-Router/internal/catalog"
	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/costs"
	"github.com/MarioGK/AI-Router/internal/dashboard"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/learn"
	"github.com/MarioGK/AI-Router/internal/limit"
	"github.com/MarioGK/AI-Router/internal/outcome"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/proxy"
	"github.com/MarioGK/AI-Router/internal/route"
	"github.com/MarioGK/AI-Router/internal/stats"
	"github.com/MarioGK/AI-Router/internal/thread"
	"github.com/MarioGK/AI-Router/internal/usage"
	"github.com/MarioGK/AI-Router/internal/wire"
)

// costRefresh is how often the cost per task is learned again from the
// request log.
const costRefresh = 10 * time.Minute

// App is the router.
type App struct {
	cfg      *config.Config
	log      *slog.Logger
	auth     *auth.Checker
	forward  *proxy.Forwarder
	catalog  *catalog.Catalog
	accounts *accounts.Watcher
	limits   *limit.Limiter
	store    *pin.Store
	costs    *costs.Table
	outcomes *outcome.Table
	// learner learns from the outcomes; nil with auto.learn off.
	learner  *learn.Learner
	resolver *route.Resolver
	counters *stats.Counters
	stats    stats.Source
	history  *history
	board    *dashboard.Dashboard
	hops     hops
	// draining is set once the router stops taking new work, and cut once
	// the shutdown grace ran out; inflight counts the requests it still
	// serves.
	draining atomic.Bool
	cut      atomic.Bool
	inflight inflight
}

// inflight counts the requests being served, so a stopping router can wait
// for them; unlike a WaitGroup, a request may start while it waits.
type inflight struct {
	mu   sync.Mutex
	n    int
	idle sync.Cond
}

func (f *inflight) add() {
	f.mu.Lock()
	f.n++
	f.mu.Unlock()
}

func (f *inflight) done() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.n--; f.n == 0 && f.idle.L != nil {
		f.idle.Broadcast()
	}
}

// wait returns once no request is being served.
func (f *inflight) wait() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.idle.L = &f.mu
	for f.n > 0 {
		f.idle.Wait()
	}
}

// New opens the pin store and builds the router.
func New(cfg *config.Config, log *slog.Logger) (*App, error) {
	return newWithClients(cfg, log, proxy.NewClient(), &http.Client{})
}

func newWithClients(cfg *config.Config, log *slog.Logger, client, external *http.Client) (*App, error) {
	base, err := url.Parse(cfg.Upstream.BaseURL)
	if err != nil {
		return nil, err
	}
	dashboardAuth, err := auth.NewPassword(cfg.Dashboard.Username, cfg.DashboardPasswordHash())
	if err != nil {
		return nil, fmt.Errorf("dashboard authentication: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(cfg.Database), 0o700); err != nil {
		return nil, err
	}
	store, err := pin.Open(cfg.Database)
	if err != nil {
		return nil, err
	}
	// Requests the metering repair corrected await a price; a failure only
	// leaves them unpriced.
	if n, err := store.RepriceRepaired(context.Background(), cfg.Price); err != nil {
		log.Warn("pricing the repaired requests failed", "error", err)
	} else if n > 0 {
		log.Info("priced requests the metering repair corrected", "requests", n)
	}
	forward := proxy.New(base, client)
	checker := auth.New(cfg.UpstreamKey())
	models := catalog.New(cfg.Upstream.BaseURL, cfg.UpstreamKey(), client)
	limits := limiter(cfg.Limits)
	tasks := costs.New(store, cfg.Price, cfg.CostFactor, cfg.Auto.Cost.Window, *cfg.Auto.Cost.Weight)
	outcomes := outcome.New(store, outcome.Rules{Ladder: cfg.Auto.Ladder(), Long: *cfg.Auto.Outcomes.Long,
		Tiny: *cfg.Auto.Outcomes.Tiny}, cfg.Auto.Cost.Window, *cfg.Auto.Cost.Weight)
	chain := decide.NewChain(steps(cfg, client, external), cfg.Auto.EscalationThreshold, cfg.Auto.Doubt, cfg.Auto.Reasoning.Criteria()).
		Rest(*cfg.Auto.DeciderTrip, cfg.Auto.DeciderRest)
	if cfg.Auto.Cost.Enabled {
		// A thread decided for a saturated model spills: it costs what the
		// spill model costs, and takes its time, when that destination is
		// offered.
		destination := func(model string, offered []decide.Candidate) string {
			if spill, ok := route.Spill(limits, model, offered); ok {
				return spill
			}
			return model
		}
		chain.Price(decide.Economy{Miss: *cfg.Auto.Cost.Miss, TimeValue: cfg.Auto.Cost.TimeValue,
			Task: func(model, level string, offered []decide.Candidate) (float64, bool) {
				return tasks.Task(destination(model, offered), level)
			},
			Learn: func(model, level string, offered []decide.Candidate) decide.Learned {
				e := outcomes.Estimate(destination(model, offered), level)
				return decide.Learned{Hours: e.Hours, MissRate: e.MissRate, Factor: e.Factor}
			}})
	}
	counters := &stats.Counters{}
	var watcher *accounts.Watcher
	var quota route.Quota
	var accountView dashboard.Accounts
	if key := cfg.ManagementKey(); key != "" {
		watcher = accounts.New(cfg.Upstream.BaseURL, key, cfg.Accounts.Providers, cfg.Accounts.LimitedPriority, client, log)
		quota, accountView = watcher, watcher
	}
	resolver := route.New(cfg.Auto, checker.Fingerprint(), store, chain, models, quota, limits, log)
	var learner *learn.Learner
	if *cfg.Auto.Learn.Enabled {
		learner = learn.New(store, outcomes, learn.Config{Ladder: cfg.Auto.Ladder(), Levels: cfg.Auto.Reasoning.IDs(),
			Window: cfg.Auto.Cost.Window, Neighbours: *cfg.Auto.Learn.Neighbours}, log)
		resolver.Learn(learner)
	}
	return &App{
		cfg: cfg, log: log, auth: checker, forward: forward, catalog: models, accounts: watcher, limits: limits,
		store: store, costs: tasks, outcomes: outcomes, learner: learner, counters: counters, resolver: resolver,
		stats: stats.Source{
			Store: store, Catalog: models, Required: cfg.Auto.Required(), Counters: counters, Deciders: chain.Health,
			Costs: func() []costs.Estimate { return tasks.Estimates(cfg.Auto.CandidateIDs(), cfg.Auto.Reasoning.IDs()) },
			Outcomes: func() stats.Outcomes {
				return stats.Outcomes{Counts: outcomes.Counts(),
					Estimates: outcomes.Estimates(cfg.Auto.CandidateIDs(), cfg.Auto.Reasoning.IDs())}
			},
			Learn:   learner.Summary,
			Started: time.Now(),
		},
		history: newHistory(store.LogRequest, historyBuffer, counters, log),
		board: dashboard.New(store, models, resolver, accountView, limits, tasks, outcomes, cfg.Auto, dashboardAuth,
			log).Learning(learner.Summary),
	}, nil
}

// Steps builds the decider chain. LLM judges share the CLIProxy client;
// systemone deciders call out through the default transport.
func Steps(cfg *config.Config, upstream *http.Client) []decide.Step {
	return steps(cfg, upstream, &http.Client{})
}

func steps(cfg *config.Config, upstream, external *http.Client) []decide.Step {
	var steps []decide.Step
	for _, d := range cfg.Deciders {
		var decider decide.Decider
		switch d.Type {
		case config.SystemOne:
			decider = decide.NewSystemOne(d.Name, d.URL, d.Model, cfg.Secret(d.APIKeyEnv), external)
		case config.LLM:
			decider = decide.NewJudge(d.Name, d.URL, d.Model, cfg.Secret(d.APIKeyEnv), upstream)
		}
		steps = append(steps, decide.Step{Decider: decider, Timeout: d.Timeout})
	}
	return steps
}

// Close revokes dashboard sessions, stops shared decisions, writes the queued
// request log and releases the pin store.
func (a *App) Close() error {
	a.board.Close()
	a.resolver.Close()
	a.history.Close()
	return a.store.Close()
}

// Run serves until ctx ends, then drains in-flight requests briefly.
func (a *App) Run(ctx context.Context) error {
	refreshCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	if err := a.catalog.Refresh(refreshCtx); err != nil {
		a.log.Warn("initial catalog refresh failed; every model counts as served until one succeeds", "error", err)
	} else if missing := a.catalog.Check(a.cfg.Auto.Required()).Missing; len(missing) > 0 {
		a.log.Warn("CLIProxy does not serve configured models", "missing", missing)
	}
	cancel()
	go a.catalog.Run(ctx, a.cfg.CatalogInterval, a.log)
	go a.costs.Run(ctx, costRefresh, a.log)
	// The learner labels the outcomes as each refresh judges them.
	if a.learner != nil {
		go a.outcomes.Run(ctx, costRefresh, a.log, a.learner.Refresh)
	} else {
		go a.outcomes.Run(ctx, costRefresh, a.log)
	}
	if a.accounts != nil {
		go a.accounts.Run(ctx, a.cfg.Accounts.Interval)
	}
	go a.retain(ctx)

	server := &http.Server{
		Addr:              a.cfg.Listen,
		Handler:           a,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       2 * time.Minute,
		ErrorLog:          slog.NewLogLogger(a.log.Handler(), slog.LevelWarn),
	}
	server.RegisterOnShutdown(a.board.Close)
	errs := make(chan error, 1)
	go func() { errs <- server.ListenAndServe() }()
	a.log.Info("ai-router listening", "listen", a.cfg.Listen, "upstream", a.cfg.Upstream.BaseURL,
		"model", a.cfg.Auto.ModelID, "candidates", len(a.cfg.Auto.Candidates), "deciders", len(a.cfg.Deciders))
	select {
	case err := <-errs:
		return err
	case <-ctx.Done():
	}
	a.draining.Store(true)
	a.log.Info("ai-router draining", "grace", a.cfg.ShutdownGrace)
	shutdownCtx, cancel := context.WithTimeout(context.Background(), a.cfg.ShutdownGrace)
	defer cancel()
	var err error
	if server.Shutdown(shutdownCtx) != nil {
		a.log.Warn("responses still streaming after the shutdown grace were cut")
		a.cut.Store(true)
		err = server.Close()
	}
	a.resolver.Close()
	// Cut requests still record themselves before the request log closes.
	a.inflight.wait()
	return err
}

// upstreamCall is one client request on its way to CLIProxy: the context
// its upstream calls run in, and the watchdog that ends them when CLIProxy
// goes silent.
type upstreamCall struct {
	r     *http.Request
	ctx   context.Context
	kind  wire.Kind
	watch *proxy.Watchdog
	// queued sums the time the request waited for concurrency slots; nil
	// for a request that is not logged.
	queued *queueClock
}

// queueClock sums how long one request waited for a concurrency slot,
// over every gate it queued at (a failover queues again).
type queueClock struct{ total time.Duration }

// add counts a wait; it does nothing on a nil clock.
func (q *queueClock) add(d time.Duration) {
	if q != nil {
		q.total += d
	}
}

// waited is the summed wait; zero on a nil clock.
func (q *queueClock) waited() time.Duration {
	if q == nil {
		return 0
	}
	return q.total
}

// ServeHTTP routes one request.
func (a *App) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	kind := wire.KindOf(r.Method, r.URL.Path)
	switch {
	case r.URL.Path == "/healthz":
		a.stats.Health(w, r, a.draining.Load())
		return
	case strings.HasPrefix(r.URL.Path, "/dashboard"):
		// The dashboard signs its operators in itself.
		a.board.ServeHTTP(w, r)
		return
	case !a.auth.Valid(r.Header):
		a.counters.Add("rejected.unauthorized")
		proxy.Error(w, http.StatusUnauthorized, "authentication_error", "invalid API key")
		return
	case r.URL.Path == "/stats":
		a.stats.Stats(w, r)
		return
	case kind == wire.Other:
		a.counters.Add("rejected.path")
		proxy.Error(w, http.StatusNotFound, "not_found_error", "ai-router serves only Claude Code's Messages API")
		return
	case kind == wire.Models:
		a.models(w, r)
		return
	}

	a.inflight.add()
	defer a.inflight.done()
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, a.cfg.MaxBodyBytes))
	if err != nil {
		status := http.StatusBadRequest
		if _, tooLarge := errors.AsType[*http.MaxBytesError](err); tooLarge {
			status = http.StatusRequestEntityTooLarge
		}
		proxy.Error(w, status, "invalid_request_error", "cannot read request body")
		return
	}
	field, found := wire.FindModel(body)
	chosen, virtual := a.virtual(field.Value)
	ctx, watch := proxy.Watch(r.Context(), a.cfg.Upstream.IdleTimeout)
	defer watch.Stop()
	call := upstreamCall{r: r, ctx: ctx, kind: kind, watch: watch, queued: &queueClock{}}
	if !found || !virtual {
		// Explicit models pass through untouched and stay out of the
		// request log, which holds only the virtual model's traffic.
		a.counters.Add("requests.explicit")
		a.relay(w, call, field.Value, body)
		return
	}

	conversation := wire.Parse(kind, body)
	id := thread.Identify(r.Header, conversation)
	target := a.resolver.Resolve(r.Context(), kind, id, conversation)
	if target.Model == "" {
		a.counters.Add("rejected.model_unavailable")
		proxy.Error(w, http.StatusServiceUnavailable, "model_unavailable_error", "no configured model can serve this request")
		return
	}
	a.counters.Add("requests.auto." + target.Reason)
	// A failover keeps the request a compaction of the same thread.
	resolved := target
	effortOf := func(model string) string {
		return a.effort(model, target.Level, chosen)
	}
	outgoing := func(model string) []byte {
		routed := field.Replace(body, config.Suffixed(model, effortOf(model)))
		if kind == wire.Messages && !reachesAnthropic(model) {
			if announced, ok := wire.AppendSystem(routed, modelNote(model)); ok {
				routed = announced
			}
		}
		return routed
	}
	failedOver := false
	response, release, err := a.send(call, target.Model, outgoing(target.Model))
	if err == nil && a.resolver.Answered(target.Model, response.StatusCode) {
		if next, ok := a.resolver.Failover(r.Context(), target, response.StatusCode, conversation.HasImage); ok {
			proxy.Discard(response)
			release()
			a.counters.Add("failover." + target.Model + "->" + next.Model)
			target, failedOver = next, true
			response, release, err = a.send(call, target.Model, outgoing(target.Model))
			if err == nil {
				a.resolver.Answered(target.Model, response.StatusCode)
			}
		}
	}
	effort := effortOf(target.Model)
	entry := pin.RequestLog{At: start, Key: target.Thread, Kind: kind.String(),
		Requested: field.Value, Model: target.Model, Effort: effort, Reason: target.Reason, Failover: failedOver}
	if err != nil {
		a.upstreamFailed(w, call, err)
		a.record(call, entry, nil, nil, err)
		return
	}
	defer release()
	a.counters.Add("upstream." + target.Model + "." + strconv.Itoa(response.StatusCode))
	if reachesAnthropic(target.Model) {
		a.claudeLimit(r.Context(), target.Model, response)
	}
	a.follow(r, target.Model, response)
	w.Header().Set("X-Ts-Router-Model", target.Model)
	if effort != "" {
		w.Header().Set("X-Ts-Router-Effort", effort)
	}
	meter := usage.NewMeter(response.Header.Get("Content-Type"))
	err = proxy.Copy(w, response, io.MultiWriter(meter, watch))
	if err != nil && r.Context().Err() == nil {
		a.counters.Add("stream_errors")
		a.log.Warn("response stream interrupted", "model", target.Model, "idle", watch.Fired(), "error", err)
	}
	if err == nil && response.StatusCode >= 200 && response.StatusCode < 300 && meter.Finished() {
		// Only a compaction that completed replaced the thread's history.
		a.resolver.Compacted(r.Context(), resolved)
	}
	a.record(call, entry, response, meter, err)
}

// effort honors a candidate's single fixed effort, otherwise preferring a
// client's auto(effort) override to the thread's mapped reasoning level.
func (a *App) effort(model, level, chosen string) string {
	if candidate, ok := a.cfg.Auto.Candidate(model); ok && len(candidate.Efforts) == 1 {
		return candidate.Efforts[0]
	}
	return cmp.Or(chosen, a.cfg.Auto.Effort(model, level))
}

// virtual reports whether a requested model is the virtual one, and the
// effort the client chose for it with an `auto(effort)` suffix.
func (a *App) virtual(model string) (effort string, ok bool) {
	if model == a.cfg.Auto.ModelID {
		return "", true
	}
	base, suffix, cut := strings.Cut(model, "(")
	effort, closed := strings.CutSuffix(suffix, ")")
	if !cut || !closed || base != a.cfg.Auto.ModelID || !config.ValidEffort(effort) {
		return "", false
	}
	return effort, true
}

// modelNote tells the pinned model which model it is: Claude Code's own
// system prompt only knows that it asked for the virtual model. The text is
// the same on every request of a thread, so prompt caching still holds.
func modelNote(model string) string {
	return "Model note from ai-router: this conversation runs on the model " + model +
		", which ai-router chose automatically for Claude Code's auto model. " +
		"When asked which model you are, answer " + model + "."
}

// reachesAnthropic reports whether a model is Claude, which CLIProxy serves
// from Anthropic's API through the Claude subscription accounts. Those
// requests get no model note: Anthropic expects Claude Code's own request
// shape on them, and a system block naming the router is not part of it.
func reachesAnthropic(model string) bool {
	return strings.HasPrefix(model, "claude-")
}

// relay forwards a request unchanged.
func (a *App) relay(w http.ResponseWriter, call upstreamCall, model string, body []byte) {
	response, release, err := a.send(call, model, body)
	if err != nil {
		a.upstreamFailed(w, call, err)
		return
	}
	defer release()
	if reachesAnthropic(config.Bare(model)) {
		a.claudeLimit(call.r.Context(), config.Bare(model), response)
	}
	a.follow(call.r, config.Bare(model), response)
	if err := proxy.Copy(w, response, call.watch); err != nil && call.r.Context().Err() == nil {
		a.counters.Add("stream_errors")
	}
}

// claudeLimit adds a reset to CLIProxy's final cooldown response. A provider
// 429 normally carries Retry-After itself; the management fallback covers the
// response CLIProxy emits after every matching subscription account is cooling
// down and it has omitted that header.
func (a *App) claudeLimit(ctx context.Context, model string, response *http.Response) {
	now := time.Now()
	var fallback time.Time
	if response.StatusCode == http.StatusTooManyRequests &&
		response.Header.Get("Anthropic-Ratelimit-Unified-Status") == "" &&
		response.Header.Get("Retry-After") == "" && a.accounts != nil {
		fallback, _ = a.accounts.ClaudeReset(ctx, model, now)
	}
	proxy.ClaudeLimit(response, now, fallback)
}

// follow notes the subscription account a response came from, and counts
// and logs a conversation that moved to another account than its previous
// request's.
func (a *App) follow(r *http.Request, model string, response *http.Response) {
	if from, to, moved := a.hops.observe(r.Header, model, response, time.Now()); moved {
		a.counters.Add("account_hops." + model)
		a.log.Warn("conversation moved to another account", "model", model, "from", from, "to", to)
	}
}

// models serves CLIProxy's model list with the virtual model added.
func (a *App) models(w http.ResponseWriter, r *http.Request) {
	listing := r.Clone(r.Context())
	listing.Header.Del("Accept-Encoding")
	response, err := a.forward.Send(r.Context(), listing, nil)
	if err != nil {
		a.upstreamFailed(w, upstreamCall{r: r, ctx: r.Context(), kind: wire.Models}, err)
		return
	}
	if response.StatusCode != http.StatusOK {
		_ = proxy.Copy(w, response, nil)
		return
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if err == nil {
		body, err = proxy.AddModel(body, proxy.Virtual{
			ID: a.cfg.Auto.ModelID, Context: a.cfg.Auto.MinContext(), Output: a.cfg.Auto.MinOutput(),
		}, a.cfg.Auto.Retired)
	}
	if err != nil {
		proxy.Error(w, http.StatusBadGateway, "upstream_error", fmt.Sprintf("CLIProxy model list: %v", err))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	_, _ = w.Write(body)
}

// upstreamFailed answers a request CLIProxy did not answer: busy, silent
// for too long, or unreachable.
func (a *App) upstreamFailed(w http.ResponseWriter, call upstreamCall, err error) {
	switch busy, isBusy := errors.AsType[busyError](err); {
	case call.r.Context().Err() != nil:
	case isBusy:
		w.Header().Set("Retry-After", "30")
		proxy.Error(w, http.StatusTooManyRequests, "rate_limit_error", busy.Error())
	case call.watch.Fired():
		a.counters.Add("upstream.silent")
		a.log.Error("CLIProxy sent no response headers in time", "path", call.r.URL.Path, "timeout", call.watch.Timeout())
		proxy.Error(w, http.StatusGatewayTimeout, "timeout_error", "CLIProxy sent nothing for "+call.watch.Timeout().String())
	default:
		a.counters.Add("upstream.unreachable")
		a.log.Error("CLIProxy request failed", "path", call.r.URL.Path, "error", err)
		proxy.Error(w, http.StatusBadGateway, "upstream_error", "CLIProxy is unreachable")
	}
}
