// Package dashboard serves the operator dashboard under /dashboard: the
// routing overview, pinned threads with their decisions and requests, and a
// live request log. Pages render on the server with templ; Datastar keeps
// them live over server-sent events. Operators sign in with the configured
// dashboard username and password.
package dashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"github.com/a-h/templ"
	"github.com/starfederation/datastar-go/datastar"

	"github.com/MarioGK/AI-Router/internal/auth"
	"github.com/MarioGK/AI-Router/internal/catalog"
	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/costs"
	"github.com/MarioGK/AI-Router/internal/learn"
	"github.com/MarioGK/AI-Router/internal/outcome"
	"github.com/MarioGK/AI-Router/internal/pin"
)

// Source is the history the dashboard reads; *pin.Store is one.
type Source interface {
	Threads(ctx context.Context, filter pin.ThreadFilter) (pin.ThreadPage, error)
	Thread(ctx context.Context, key string) (*pin.ThreadDetail, error)
	Requests(ctx context.Context, filter pin.RequestFilter) ([]pin.RequestLog, error)
	Prune(ctx context.Context, before time.Time) (int64, error)
	Overview(ctx context.Context, since time.Time, bucket time.Duration) (pin.Overview, error)
	Decisions(ctx context.Context, filter pin.DecisionFilter) (pin.DecisionPage, error)
	DecisionMix(ctx context.Context, since time.Time) ([]pin.ModelDecisions, error)
	Reviews(ctx context.Context, since time.Time) (pin.ReviewSummary, error)
	Reviewed(ctx context.Context) ([]pin.Event, error)
	SaveReview(ctx context.Context, review pin.Review) error
	DeleteReview(ctx context.Context, event int64) error
}

// Costs estimates what a task costs on each model at each reasoning level;
// *costs.Table is one.
type Costs interface {
	Estimates(models, levels []string) []costs.Estimate
	Learned() (int, time.Time)
}

// Outcomes is how the settled threads went: what a task takes on each
// model at each level beyond its cost, and the unreviewed decisions most
// likely misrouted; *outcome.Table is one.
type Outcomes interface {
	Estimate(model, level string) outcome.Estimate
	Queue(limit int) []outcome.Outcome
	Counts() outcome.Counts
	Reviewed(decision int64)
}

// Learning makes the Decisions tab show what routing learned from its own
// history.
func (d *Dashboard) Learning(summary func() learn.Summary) *Dashboard {
	d.learning = summary
	return d
}

// Catalog reports which configured models CLIProxy serves;
// *catalog.Catalog is one.
type Catalog interface {
	Check(required []string) catalog.Status
}

// csp keeps every page on the router's own assets. Datastar evaluates its
// expressions and runs the scripts it is sent, hence the unsafe sources.
const csp = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
	"style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"

// Dashboard serves the dashboard's pages, live streams and assets.
type Dashboard struct {
	source     Source
	catalog    Catalog
	classifier Classifier
	accounts   Accounts
	limits     Limits
	costs      Costs
	outcomes   Outcomes
	// learning is what routing learned from its own history; nil when it
	// learns nothing.
	learning func() learn.Summary
	auto     config.Auto
	checker  *auth.Password
	sessions *sessions
	throttle *throttle
	// crossOrigin refuses state-changing requests from other origins: the
	// session cookie is SameSite=Strict, but a page on a sibling subdomain
	// is same-site and would still send it.
	crossOrigin *http.CrossOriginProtection
	// trials holds a slot for every trial running.
	trials chan struct{}
	log    *slog.Logger
	mux    *http.ServeMux
	// poll is how often a live stream looks for new requests.
	poll      time.Duration
	done      chan struct{}
	closeOnce sync.Once
	// logRevision invalidates live tables after an operator deletes logs.
	logRevision atomic.Uint64
}

// New returns the dashboard. auto names the configured models and their
// ladder; classifier decides trial prompts and persists routing settings;
// accounts, when not nil, shows the subscription accounts' quota; limits
// shows the concurrency gates; outcomes, when not nil, what settled
// threads taught and which decisions look misrouted.
func New(source Source, models Catalog, classifier Classifier, accounts Accounts, limits Limits, costs Costs,
	outcomes Outcomes, auto config.Auto, checker *auth.Password, log *slog.Logger) *Dashboard {
	d := &Dashboard{
		source: source, catalog: models, classifier: classifier, accounts: accounts, limits: limits, costs: costs,
		outcomes: outcomes,
		auto:     auto, checker: checker,
		sessions: &sessions{expires: map[string]time.Time{}}, throttle: &throttle{failed: map[string]failures{}},
		crossOrigin: http.NewCrossOriginProtection(), trials: make(chan struct{}, maxTrials),
		log: log, poll: 2 * time.Second, done: make(chan struct{}),
	}
	d.mux = http.NewServeMux()
	d.mux.HandleFunc("GET /dashboard/assets/{name}", serveAsset)
	d.mux.HandleFunc("GET /dashboard/login", d.loginPage)
	d.mux.HandleFunc("POST /dashboard/login", d.login)
	d.mux.HandleFunc("POST /dashboard/logout", d.logout)
	for pattern, handler := range map[string]http.HandlerFunc{
		"POST /dashboard/theme":                    d.theme,
		"GET /dashboard":                           d.overview,
		"GET /dashboard/{$}":                       d.overview,
		"GET /dashboard/sse/feed":                  d.feed,
		"GET /dashboard/sse/mix":                   d.mix,
		"GET /dashboard/threads":                   d.threads,
		"GET /dashboard/threads/{key}":             d.thread,
		"POST /dashboard/sse/threads/{key}/pin":    d.repin,
		"POST /dashboard/sse/threads/{key}/review": d.review,
		"GET /dashboard/decisions/cases.yaml":      d.cases,
		"GET /dashboard/sse/threads":               d.threadRows,
		"GET /dashboard/decisions":                 d.decisions,
		"GET /dashboard/sse/decisions":             d.decisionRows,
		"POST /dashboard/sse/try":                  d.try,
		"POST /dashboard/logs/delete":              d.deleteLogs,
		"GET /dashboard/logs":                      d.logs,
		"GET /dashboard/sse/logs":                  d.tailLogs,
		"GET /dashboard/routing":                   d.routing,
		"POST /dashboard/routing":                  d.saveRouting,
	} {
		d.mux.Handle(pattern, d.private(handler))
	}
	return d
}

// ServeHTTP serves one dashboard request. A POST that a browser marks as
// coming from another origin, by Sec-Fetch-Site or by an Origin whose host
// is not the request's, gets 403.
func (d *Dashboard) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	header := w.Header()
	header.Set("Content-Security-Policy", csp)
	header.Set("X-Content-Type-Options", "nosniff")
	header.Set("Referrer-Policy", "same-origin")
	if err := d.crossOrigin.Check(r); err != nil {
		http.Error(w, "cross-origin request refused", http.StatusForbidden)
		return
	}
	d.mux.ServeHTTP(w, r)
}

// Close ends the live streams, so a shutting-down server can drain.
func (d *Dashboard) Close() {
	d.closeOnce.Do(func() {
		d.sessions.close()
		close(d.done)
	})
}

// render writes a whole page.
func (d *Dashboard) render(w http.ResponseWriter, r *http.Request, status int, page templ.Component) {
	var body bytes.Buffer
	if err := page.Render(r.Context(), &body); err != nil {
		d.log.Error("dashboard page failed", "path", r.URL.Path, "error", err)
		http.Error(w, "the page failed to render", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_, _ = body.WriteTo(w)
}

// patch morphs components into the page, each onto the element with its id.
func (d *Dashboard) patch(sse *datastar.ServerSentEventGenerator, components ...templ.Component) {
	if err := sse.PatchElementTempl(templ.Join(components...)); err != nil && !sse.IsClosed() {
		d.log.Warn("dashboard patch failed", "error", err)
	}
}

// replaceURL points the browser's address at the page for what it shows.
func (d *Dashboard) replaceURL(sse *datastar.ServerSentEventGenerator, path string, query url.Values) {
	if err := sse.ReplaceURL(url.URL{Path: path, RawQuery: query.Encode()}); err != nil && !sse.IsClosed() {
		d.log.Warn("dashboard patch failed", "error", err)
	}
}

// follow calls step every poll interval until the client leaves or the
// dashboard closes.
func (d *Dashboard) follow(ctx context.Context, step func(tick int)) {
	ticker := time.NewTicker(d.poll)
	defer ticker.Stop()
	for tick := 1; ; tick++ {
		select {
		case <-ctx.Done():
			return
		case <-d.done:
			return
		case <-ticker.C:
			if ctx.Err() != nil {
				return
			}
			step(tick)
		}
	}
}

// jsString quotes text as a JavaScript string for a Datastar expression.
func jsString(text string) string {
	encoded, _ := json.Marshal(text)
	return string(encoded)
}

// signals renders Datastar signals for a data-signals attribute.
func signals(values map[string]any) string {
	encoded, _ := json.Marshal(values)
	return string(encoded)
}
