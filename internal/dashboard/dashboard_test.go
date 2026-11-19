package dashboard

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/crypto/bcrypt"

	"github.com/MarioGK/AI-Router/internal/accounts"
	"github.com/MarioGK/AI-Router/internal/auth"
	"github.com/MarioGK/AI-Router/internal/catalog"
	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/costs"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/limit"
	"github.com/MarioGK/AI-Router/internal/outcome"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/route"
)

const (
	dashboardUsername = "operator"
	dashboardPassword = "dashboard-test-password"
)

var dashboardTestHash = sync.OnceValues(func() (string, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(dashboardPassword), bcrypt.MinCost)
	return string(hash), err
})

// fakeSource serves canned history and records the filters it was asked for.
type fakeSource struct {
	mu        sync.Mutex
	threads   []pin.Thread
	detail    *pin.ThreadDetail
	requests  []pin.RequestLog
	overview  pin.Overview
	decisions []pin.Event
	mix       []pin.ModelDecisions
	err       error

	threadFilters   []pin.ThreadFilter
	requestFilters  []pin.RequestFilter
	decisionFilters []pin.DecisionFilter
	overviewCalls   []time.Duration
	reviews         map[int64]pin.Review
	pruneBefore     time.Time
	pruneCount      int64
	pruneErr        error
}

func (f *fakeSource) Reviews(context.Context, time.Time) (pin.ReviewSummary, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	summary := pin.ReviewSummary{Misses: map[string]int{}, Unreviewed: len(f.decisions) - len(f.reviews)}
	for _, e := range f.reviewedLocked() {
		summary.Reviewed++
		if model, _ := e.Review.Right(e); model {
			summary.RightModel++
		} else {
			summary.Misses[e.Decided()+"->"+e.Review.Model]++
		}
	}
	return summary, f.err
}

func (f *fakeSource) Reviewed(context.Context) ([]pin.Event, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.reviewedLocked(), f.err
}

// reviewedLocked is every decision with its review, of the thread detail
// and the decisions.
func (f *fakeSource) reviewedLocked() []pin.Event {
	var list []pin.Event
	var events []pin.Event
	if f.detail != nil && f.detail.Decision != nil {
		events = append(events, *f.detail.Decision)
	}
	events = append(events, f.decisions...)
	for _, e := range events {
		if review, ok := f.reviews[e.ID]; ok && !slices.ContainsFunc(list, func(l pin.Event) bool { return l.ID == e.ID }) {
			e.Review = &review
			list = append(list, e)
		}
	}
	return list
}

func (f *fakeSource) SaveReview(_ context.Context, review pin.Review) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.reviews == nil {
		f.reviews = map[int64]pin.Review{}
	}
	f.reviews[review.Event] = review
	return f.err
}

func (f *fakeSource) DeleteReview(_ context.Context, event int64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.reviews, event)
	return f.err
}

func (f *fakeSource) Decisions(_ context.Context, filter pin.DecisionFilter) (pin.DecisionPage, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.decisionFilters = append(f.decisionFilters, filter)
	page := pin.DecisionPage{}
	for _, e := range f.decisions {
		if (filter.Model == "" || e.Model == filter.Model) && (filter.Decider == "" || e.Decider == filter.Decider) {
			page.Total++
			if page.Total > filter.Offset && len(page.Decisions) < filter.Limit {
				page.Decisions = append(page.Decisions, e)
			}
		}
	}
	return page, f.err
}

func (f *fakeSource) DecisionMix(context.Context, time.Time) ([]pin.ModelDecisions, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.mix, f.err
}

// fakeAccounts is a fixed view of the subscription accounts: every ChatGPT
// account at its limit, one of two Claude accounts in rotation.
type fakeAccounts struct{ snapshot accounts.Snapshot }

func (f fakeAccounts) Snapshot() accounts.Snapshot { return f.snapshot }

func (f fakeAccounts) Available(model string) bool { return model != "gpt-6.1-sol" }

func testAccounts() fakeAccounts {
	now := time.Now()
	return fakeAccounts{accounts.Snapshot{CheckedAt: now.Add(-2 * time.Minute), Accounts: []accounts.Account{
		{Name: "codex-a.json", Provider: accounts.Codex, Email: "filippo@example.com", Plan: "pro", Priority: -10, Limited: true,
			Until: now.Add(100 * time.Hour), Models: []string{"gpt-6.1-sol"},
			Windows: []accounts.Window{{Name: "weekly", Used: 1, ResetsAt: now.Add(100 * time.Hour)}}},
		{Name: "codex-b.json", Provider: accounts.Codex, Email: "mario@example.com", Plan: "pro", Priority: -10, Limited: true,
			Until: now.Add(3 * time.Hour), Models: []string{"gpt-6.1-sol"},
			Windows: []accounts.Window{{Name: "weekly", Used: 0.93, ResetsAt: now.Add(125 * time.Hour)}, {Name: "5h", Used: 1, ResetsAt: now.Add(3 * time.Hour)}}},
		{Name: "claude-a.json", Provider: accounts.Claude, Email: "mbelchior@example.com", Priority: -10, Limited: true,
			Until: now.Add(55 * time.Hour), Models: []string{"claude-opus-5-5"},
			Windows: []accounts.Window{{Name: "seven_day", Used: 1, ResetsAt: now.Add(55 * time.Hour)}, {Name: "nimbus_quill"}}},
		{Name: "claude-b.json", Provider: accounts.Claude, Email: "mariogk01@example.com", Models: []string{"claude-opus-5-5"},
			Windows: []accounts.Window{{Name: "five_hour", Used: 0.04, ResetsAt: now.Add(4 * time.Hour)},
				{Name: "seven_day_opus", Used: 0.35}, {Name: "seven_day", Used: 0.16, ResetsAt: now.Add(150 * time.Hour)}}},
	}}}
}

// fakeLimits is GLM's gate, throttled down to three of four slots.
type fakeLimits struct{}

func (fakeLimits) Stats() []limit.Stats {
	return []limit.Stats{{Model: "glm-5.3", Limit: 3, Max: 4, InFlight: 3, Waiting: 2, Throttles: 7,
		Throttled: time.Now().Add(-5 * time.Minute), Spill: "gpt-6.1-sol"}}
}

// fakeCosts has learned nothing yet: a task costs its prior, $0.25, on
// every model.
type fakeCosts struct{}

func (fakeCosts) Estimates(models, _ []string) []costs.Estimate {
	estimates := make([]costs.Estimate, len(models))
	for i, model := range models {
		estimates[i] = costs.Estimate{Model: model, Cost: 0.25}
	}
	return estimates
}

func (fakeCosts) Learned() (int, time.Time) { return 0, time.Time{} }

// fakeOutcomes queues one thread an operator moved up and records which
// decisions were reviewed since.
type fakeOutcomes struct {
	reviewed *[]int64
}

func (fakeOutcomes) Estimate(model, level string) outcome.Estimate {
	return outcome.Estimate{Model: model, Level: level, Hours: 0.25, MissRate: 0.1, Factor: 1}
}

func (fakeOutcomes) Queue(int) []outcome.Outcome {
	return []outcome.Outcome{{Key: "thread-misrouted", Model: "glm-5.3", Level: "low", Decision: 7, Requests: 40,
		Active: 9 * time.Minute, Cost: 0.12, Signals: []outcome.Signal{
			{Kind: outcome.RepinUp, Strength: 1, Text: "An operator moved it up, glm-5.3 → claude-opus-5-5"},
			{Kind: outcome.QuotaFailover, Text: "It failed over when no account had quota left, glm-5.3 → gpt-6.1-sol"}}}}
}

func (fakeOutcomes) Counts() outcome.Counts { return outcome.Counts{Threads: 12, Queued: 1} }

func (f fakeOutcomes) Reviewed(decision int64) {
	if f.reviewed != nil {
		*f.reviewed = append(*f.reviewed, decision)
	}
}

// testPolicy is the policy the fake classifier decides under.
var testPolicy = decide.Policy{
	Instructions: decide.Instructions{Question: "Choose the cheapest model that can do the work well.", Focus: "Judge the whole task."},
	PairQuestion: "Can %[1]s do it, or does it need %[2]s?", Threshold: 0.1, Doubt: decide.DoubtPair,
	Deciders: []decide.DeciderInfo{{Name: "jev", Kind: decide.KindSystemOne, Model: "jev-latest"},
		{Name: "deepseek", Kind: decide.KindLLM, Model: "deepseek-flash"}},
	Candidates: []decide.Candidate{
		{ID: "glm-5.3", Description: "Everyday coding, small UI changes included.", NotFor: "Urgent work.",
			Examples: []string{"Hide the Reports menu entry."}},
		{ID: "claude-opus-5-5", Description: "Substantial UI design and the most complex tasks."}},
}

// fakeClassifier decides every trial prompt for glm-5.3 through jev.
type fakeClassifier struct {
	mu                         sync.Mutex
	tried                      []string
	images                     bool
	overrides                  []string
	routing                    route.Settings
	routingErr, saveRoutingErr error
	savedRouting               []route.Settings
}

func (c *fakeClassifier) Settings(context.Context) (route.Settings, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	settings := c.routing
	if settings.Mode == "" {
		settings = route.Settings{Mode: route.ModeAuto, JevEnabled: true}
	}
	settings.Models = slices.Clone(settings.Models)
	return settings, c.routingErr
}

func (c *fakeClassifier) SetSettings(_ context.Context, settings route.Settings) (route.Settings, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.saveRoutingErr != nil {
		return route.Settings{}, c.saveRoutingErr
	}
	settings.Models = slices.Clone(settings.Models)
	c.routing = settings
	c.savedRouting = append(c.savedRouting, settings)
	return settings, nil
}

func (c *fakeClassifier) Policy(bool) decide.Policy {
	return testPolicy
}

func (c *fakeClassifier) Override(_ context.Context, key, model, level string) (*pin.Pin, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if model == "gpt-9" {
		return nil, errors.New(`"gpt-9" is not a candidate`)
	}
	c.overrides = append(c.overrides, key+" "+model+"@"+level)
	return &pin.Pin{Key: key, Model: model, Level: level}, nil
}

func (c *fakeClassifier) Health() []decide.DeciderHealth {
	return []decide.DeciderHealth{{Name: "jev", Failures: 3, Resting: time.Now().Add(time.Minute)}, {Name: "deepseek"}}
}

func (c *fakeClassifier) Try(_ context.Context, prompt string, images bool) (pin.Event, decide.Policy) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tried, c.images = append(c.tried, prompt), images
	state := "images=no\n\n" + prompt
	if images {
		state = "images=yes\n\n" + prompt
	}
	return pin.Event{Kind: pin.Decision, Model: "glm-5.3", Decider: "jev", Choice: "glm-5.3",
		Probabilities: map[string]float64{"glm-5.3": 0.97, "claude-opus-5-5": 0.03}, Latency: 280 * time.Millisecond, Prompt: state,
		Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Doubted, Model: "jev-1.13.0", Choice: "glm-5.3",
			Probabilities: map[string]float64{"glm-5.3": 0.8, "claude-opus-5-5": 0.2}, Above: 0.2, LatencyMS: 280,
			Response: `{"model":"jev-1.13.0","answers":{"model":{"choice":"glm-5.3"}}}`},
			{Decider: "jev", Outcome: decide.Decided, Model: "jev-1.13.0", Pair: []string{"glm-5.3", "claude-opus-5-5"}, Choice: "glm-5.3",
				Probabilities: map[string]float64{"glm-5.3": 0.7, "claude-opus-5-5": 0.3}, Above: 0.3, LatencyMS: 260,
				Response: `{"answers":{"pair":{"choice":"glm-5.3"}}}`}},
		Basis: pin.Basis{Threshold: 0.1, Policy: testPolicy.Fingerprint()}}, testPolicy
}

func (f *fakeSource) Threads(_ context.Context, filter pin.ThreadFilter) (pin.ThreadPage, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.threadFilters = append(f.threadFilters, filter)
	page := pin.ThreadPage{}
	for _, t := range f.threads {
		if filter.Model == "" || t.Model == filter.Model {
			page.Total++
			if page.Total > filter.Offset && len(page.Threads) < filter.Limit {
				page.Threads = append(page.Threads, t)
			}
		}
	}
	return page, f.err
}

func (f *fakeSource) Thread(_ context.Context, key string) (*pin.ThreadDetail, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.detail == nil || f.detail.Key != key {
		return nil, f.err
	}
	return f.detail, f.err
}

// Requests keeps the store's contract: with AfterID the newer requests
// oldest first, else the newest first.
func (f *fakeSource) Prune(_ context.Context, before time.Time) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pruneBefore = before
	return f.pruneCount, f.pruneErr
}

func (f *fakeSource) Requests(_ context.Context, filter pin.RequestFilter) ([]pin.RequestLog, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requestFilters = append(f.requestFilters, filter)
	var found []pin.RequestLog
	matches := func(r pin.RequestLog) bool {
		return (filter.Model == "" || r.Model == filter.Model) && len(found) < filter.Limit &&
			(filter.Status != "errors" || failed(r)) && (filter.Status != "success" || !failed(r))
	}
	if filter.AfterID > 0 {
		for _, r := range f.requests {
			if r.ID > filter.AfterID && matches(r) {
				found = append(found, r)
			}
		}
		return found, f.err
	}
	for i := len(f.requests) - 1; i >= 0; i-- {
		if matches(f.requests[i]) {
			found = append(found, f.requests[i])
		}
	}
	return found, f.err
}

func (f *fakeSource) Overview(_ context.Context, since time.Time, bucket time.Duration) (pin.Overview, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.overviewCalls = append(f.overviewCalls, bucket)
	return f.overview, f.err
}

func (f *fakeSource) lastRequestFilter() pin.RequestFilter {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requestFilters[len(f.requestFilters)-1]
}

func (f *fakeSource) log(request pin.RequestLog) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, request)
}

type fakeCatalog struct{ status catalog.Status }

func (c fakeCatalog) Check([]string) catalog.Status { return c.status }

var testAuto = config.Auto{
	DefaultModel: "glm-5.3", AuxModel: "glm-5.3-flash", EscalationThreshold: 0.1, Cost: config.Cost{Miss: new(1.0)},
	Candidates: []config.Candidate{{ID: "deepseek-flash"}, {ID: "glm-5.3"}, {ID: "gpt-6.1-sol"}, {ID: "claude-opus-5-5"}},
	Fallback:   map[string]string{"claude-opus-5-5": "claude-sonnet-5"},
}

// history is a small day of routing: two threads, one escalated with a
// doubt, one failed over, and their requests.
func history() *fakeSource {
	now := time.Now()
	decision := &pin.Event{
		ID: 1, At: now.Add(-3 * time.Hour), Key: strings.Repeat("a", 64), Kind: pin.Decision,
		Model: "claude-opus-5-5", Decider: "judge", Choice: "gpt-6.1-sol", Escalated: true, Latency: 420 * time.Millisecond,
		Probabilities: map[string]float64{"gpt-6.1-sol": 0.83, "claude-opus-5-5": 0.17},
		Attempts: []decide.Attempt{
			{Decider: "jev", Outcome: decide.Doubted, Model: "jev-1.13.0", Choice: "gpt-6.1-sol",
				Probabilities: map[string]float64{"gpt-6.1-sol": 0.7, "claude-opus-5-5": 0.3}, Confidence: 0.8, Above: 0.3,
				InputTokens: 356, OutputTokens: 52, LatencyMS: 280},
			{Decider: "ollama", Outcome: decide.Failed, Error: "status 503", LatencyMS: 12},
			{Decider: "judge", Outcome: decide.Decided, Model: "gpt-6-luna", Choice: "gpt-6.1-sol",
				Probabilities: map[string]float64{"gpt-6.1-sol": 0.83, "claude-opus-5-5": 0.17}, Above: 0.17,
				Reason: "A subtle cross-service race.", LatencyMS: 130},
		},
		Basis: pin.Basis{Kind: "messages", UserAgent: "claude-cli/2.1.284 (external, cli)",
			Candidates: []string{"deepseek-flash", "glm-5.3", "gpt-6.1-sol", "claude-opus-5-5"}, Threshold: 0.1, Doubt: "consult",
			PromptChars: 18234, Truncated: true, Policy: "a1b2c3d4e5f6", Version: "9e9f7a1c0ffe"},
		Prompt: "images=no\n\nFind the race in the settlement service",
	}
	escalated := pin.Thread{
		Pin: pin.Pin{Key: decision.Key, Model: "claude-opus-5-5", Decided: "claude-opus-5-5",
			CreatedAt: decision.At},
		LastUsedAt: now.Add(-time.Minute), Requests: 2, Decision: decision,
	}
	failedOver := pin.Thread{
		Pin: pin.Pin{Key: strings.Repeat("b", 64), Model: "glm-5.3", Decided: "gpt-6.1-sol", FailedOver: true,
			CreatedAt: now.Add(-2 * time.Hour)},
		LastUsedAt: now.Add(-2 * time.Minute), Requests: 1,
	}
	requests := []pin.RequestLog{
		{ID: 1, At: now.Add(-3 * time.Hour), Key: escalated.Key, Kind: "messages", Requested: "auto",
			Model: "claude-opus-5-5", Reason: "decided", Status: 200, Latency: 3 * time.Second},
		{ID: 2, At: now.Add(-2 * time.Hour), Key: failedOver.Key, Kind: "messages", Requested: "auto",
			Model: "glm-5.3", Reason: "failover", Status: 200, Failover: true, Latency: 1500 * time.Millisecond, Queue: 1250 * time.Millisecond},
		{ID: 3, At: now.Add(-time.Minute), Key: escalated.Key, Kind: "messages", Requested: "auto",
			Model: "claude-opus-5-5", Reason: "pinned", Status: 529, Latency: 90 * time.Millisecond, Error: ""},
		{ID: 4, At: now.Add(-30 * time.Second), Kind: "messages", Requested: "auto",
			Model: "glm-5.3-flash", Reason: "auxiliary", Status: 0, Error: "canceled by the client"},
	}
	hour := now.Truncate(time.Hour)
	return &fakeSource{
		threads:  []pin.Thread{escalated, failedOver},
		requests: requests,
		// The second decision was recorded while the router served other
		// clients too, with their client line.
		decisions: []pin.Event{*decision, {ID: 3, At: now.Add(-2 * time.Hour), Key: failedOver.Key, Kind: pin.Decision,
			Model: "gpt-6.1-sol", Decider: "jev", Choice: "gpt-6.1-sol", Prompt: "client=codex images=no\n\nFix the flaky retry test"}},
		mix: []pin.ModelDecisions{{Model: "claude-opus-5-5", Count: 1, Examples: []pin.Event{*decision}},
			{Model: "gpt-6.1-sol", Count: 3}},
		detail: &pin.ThreadDetail{Thread: escalated, Policy: &testPolicy, Requests: []pin.RequestLog{requests[2], requests[0]}, Events: []pin.Event{
			*decision,
			{ID: 2, At: now.Add(-time.Hour), Key: escalated.Key, Kind: pin.CapabilityUpgrade, Previous: "glm-5.3",
				Model: "claude-opus-5-5", Detail: "images"},
		}},
		overview: pin.Overview{
			Threads: 2, Requests: 1284, Errors: 17, Cost: 3.4567,
			Queue: map[string]pin.QueueWait{"glm-5.3": {Requests: 80, Queued: 12, Mean: 3200 * time.Millisecond, P95: 18 * time.Second}},
			Models: []pin.ModelCost{
				{Model: "claude-opus-5-5", Threads: 2, Requests: 700, Cost: 3.2},
				{Model: "glm-5.3-flash", Requests: 84, Cost: 0.0042},
			},
			Summary: pin.Summary{
				Decisions: map[string]int{"claude-opus-5-5": 1}, Deciders: map[string]int{"judge": 1},
				Escalated: 1, Consulted: 1, Overrides: 2,
				Failovers: map[string]int{"gpt-6.1-sol->glm-5.3": 1}, Upgrades: map[string]int{},
				LatencyMS: map[string]int64{"p50": 420, "p95": 1300, "max": 1300},
			},
			Timeline: []pin.Bucket{
				{Start: hour.Add(-3 * time.Hour), Requests: map[string]int{"claude-opus-5-5": 700, "glm-5.3": 84}, Decisions: map[string]int{"claude-opus-5-5": 1}},
				{Start: hour, Requests: map[string]int{"claude-sonnet-5": 500}},
			},
		},
	}
}

// newDashboard serves source with a catalog missing one model.
func newDashboard(t *testing.T, source Source) *Dashboard {
	t.Helper()
	hash, err := dashboardTestHash()
	if err != nil {
		t.Fatal(err)
	}
	checker, err := auth.NewPassword(dashboardUsername, hash)
	if err != nil {
		t.Fatal(err)
	}
	d := New(source, fakeCatalog{catalog.Status{Loaded: true, Missing: []string{"claude-sonnet-5"}, CheckedAt: time.Now()}},
		&fakeClassifier{}, testAccounts(), fakeLimits{}, fakeCosts{}, fakeOutcomes{}, testAuto, checker, slog.New(slog.DiscardHandler))
	d.poll = 10 * time.Millisecond
	t.Cleanup(d.Close)
	return d
}

// signIn opens a session and returns its cookie.
func signIn(t *testing.T, d *Dashboard) *http.Cookie {
	t.Helper()
	response := serve(d, http.MethodPost, "/dashboard/login", url.Values{"username": {dashboardUsername}, "password": {dashboardPassword}}.Encode(), nil)
	for _, cookie := range response.Result().Cookies() {
		if cookie.Name == sessionCookie {
			return cookie
		}
	}
	t.Fatalf("sign-in set no session cookie: %d %s", response.Code, response.Body)
	return nil
}

func serve(d *Dashboard, method, target, form string, cookie *http.Cookie, header ...string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, target, strings.NewReader(form))
	if form != "" {
		request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	if cookie != nil {
		request.AddCookie(cookie)
	}
	for i := 0; i < len(header); i += 2 {
		request.Header.Set(header[i], header[i+1])
	}
	response := httptest.NewRecorder()
	d.ServeHTTP(response, request)
	return response
}

// page fetches a page as a signed-in operator and checks its status.
func page(t *testing.T, d *Dashboard, cookie *http.Cookie, target string, status int) string {
	t.Helper()
	response := serve(d, http.MethodGet, target, "", cookie)
	if response.Code != status {
		t.Fatalf("GET %s = %d, want %d: %s", target, response.Code, status, response.Body)
	}
	if got := response.Header().Get("Content-Type"); got != "text/html; charset=utf-8" {
		t.Fatalf("GET %s content type %q", target, got)
	}
	return response.Body.String()
}

func contains(t *testing.T, where, body string, wants ...string) {
	t.Helper()
	for _, want := range wants {
		if !strings.Contains(body, want) {
			t.Errorf("%s lacks %q", where, want)
		}
	}
}

func TestOverviewShowsTheDay(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	body := page(t, d, signIn(t, d), "/dashboard", http.StatusOK)
	contains(t, "overview", body,
		"Pinned threads", "1,284", "17 errors, 1%", "1 new in 24h", "2 re-pinned by hand", "420 ms",
		"median, p95 1.30 s", "Second opinions", "100% of decisions", "gpt-6.1-sol → glm-5.3", "Requests per hour",
		"Model mix", "1 missing", "claude-sonnet-5",
		"Cost per task", "$3.46 total", "$1.60", "$3.20", "$0.0042",
		`id="feed-rows"`, "canceled by the client", "no response", "data-init=\"@get(&#39;/dashboard/sse/feed&#39;",
		`aria-current="page"`)
	if got := source.overviewCalls; len(got) != 1 || got[0] != time.Hour {
		t.Fatalf("overview buckets = %v, want one hourly read", got)
	}
}

func TestOverviewReportsAnUnavailableHistory(t *testing.T) {
	source := history()
	source.err = errShown
	d := newDashboard(t, source)
	body := page(t, d, signIn(t, d), "/dashboard", http.StatusOK)
	contains(t, "overview", body, "The request history is unavailable", errShown.Error(), "—")
}

var errShown = errors.New("database is locked")

func TestThreadsFilterAndPage(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	cookie := signIn(t, d)
	body := page(t, d, cookie, "/dashboard/threads", http.StatusOK)
	contains(t, "threads", body, shortKey(source.threads[0].Key), shortKey(source.threads[1].Key), "escalated", "2nd opinion",
		"failover", "judge", "no decision", "1–2 of 2 threads", `data-bind="model"`)
	if strings.Contains(body, "Next page") {
		t.Error("a single page of threads has page buttons")
	}

	body = page(t, d, cookie, "/dashboard/threads?model=glm-5.3&page=3", http.StatusOK)
	if !strings.Contains(body, shortKey(source.threads[1].Key)) || strings.Contains(body, shortKey(source.threads[0].Key)) {
		t.Fatal("the model filter did not narrow the threads")
	}
	if got := source.threadFilters[len(source.threadFilters)-1]; got != (pin.ThreadFilter{Model: "glm-5.3", Limit: threadsPerPage}) {
		t.Fatalf("a page past the end did not fall back to the last: %+v", got)
	}
}

func TestThreadShowsDecisionHistoryAndRequests(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	cookie := signIn(t, d)
	body := page(t, d, cookie, "/dashboard/threads/"+source.detail.Key, http.StatusOK)
	contains(t, "thread", body, shortKey(source.detail.Key), source.detail.Key, "83%", "17%", "Why claude-opus-5-5",
		"jev chose gpt-6.1-sol (70%) with 80% confidence but put 30% on stronger models, at least the 10% threshold, so it asked ollama.",
		"ollama failed: status 503.",
		"judge chose gpt-6.1-sol (83%) but put 17% on stronger models, at least the 10% threshold, so the thread went one step up to claude-opus-5-5.",
		"A subtle cross-service race.", "jev-1.13.0", "gpt-6-luna", "356 in · 52 out", "doubted", "failed", "decided",
		"Decision input", "18,234 chars", "cut to fit", "a1b2c3d4e5f6", "9e9f7a1c0ffe", "claude-cli/2.1.284 (external, cli)",
		"Prompt as judged", "Find the race in the settlement service", "Task cost",
		"Upgraded", "the thread carried images", "Decided", "by judge",
		"took 420 ms, escalated, after a second opinion", "529", `id="thread-requests"`)
	odds := body[strings.Index(body, `id="odds"`):]
	if strings.Index(odds, "gpt-6.1-sol") > strings.Index(odds, "claude-opus-5-5") {
		t.Error("the probabilities are not in ladder order")
	}

	parent, agent := strings.Repeat("a", 64), strings.Repeat("b", 64)
	source.detail.Parent, source.detail.Agents = parent, []string{agent}
	source.detail.Events = append(source.detail.Events, pin.Event{
		Kind: pin.Redecision, Model: "glm-5.3", Previous: "claude-opus-5-5", Level: "medium", Decider: "jev",
		Latency: 300 * time.Millisecond, Detail: "idle for 12m0s; claude-opus-5-5 (xhigh) → glm-5.3 (medium)",
	})
	body = page(t, d, cookie, "/dashboard/threads/"+source.detail.Key, http.StatusOK)
	contains(t, "subagent thread", body, "Subagent of", "/dashboard/threads/"+parent, "Subagents",
		"/dashboard/threads/"+agent, "Decided again", "took 300 ms, idle for 12m0s; claude-opus-5-5 (xhigh) → glm-5.3 (medium)")

	body = page(t, d, cookie, "/dashboard/threads/"+strings.Repeat("c", 64), http.StatusNotFound)
	contains(t, "unknown thread", body, "No such thread")
	source.err = errShown
	body = page(t, d, cookie, "/dashboard/threads/"+source.detail.Key, http.StatusInternalServerError)
	contains(t, "failed thread", body, errShown.Error())
}

func TestLogsShowTheFilteredRequests(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	body := page(t, d, signIn(t, d), "/dashboard/logs?model=claude-opus-5-5&status=errors", http.StatusOK)
	contains(t, "logs", body, `id="log-rows"`, `id="request-3"`, "529", "@get(&#39;/dashboard/sse/logs&#39;", "$paused",
		`value="errors" selected`, `value="claude-opus-5-5" selected`)
	if strings.Contains(body, `id="request-1"`) {
		t.Error("the status filter kept a successful request")
	}
	if strings.Contains(body, `id="request-4"`) {
		t.Error("the model filter kept another model's request")
	}
	if got := source.lastRequestFilter(); got.Model != "claude-opus-5-5" || got.Limit != tailSize {
		t.Fatalf("log read %+v", got)
	}
}

// external matches a script, stylesheet or other resource loaded from
// another origin.
var external = regexp.MustCompile(`(?i)<(script|link|img|iframe)\b[^>]*\b(src|href)\s*=\s*["']?(https?:)?//`)

func TestPagesUseOnlyTheirOwnAssets(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	cookie := signIn(t, d)
	pages := map[string]string{
		"login":    page(t, d, nil, "/dashboard/login", http.StatusOK),
		"overview": page(t, d, cookie, "/dashboard", http.StatusOK),
		"threads":  page(t, d, cookie, "/dashboard/threads", http.StatusOK),
		"thread":   page(t, d, cookie, "/dashboard/threads/"+source.detail.Key, http.StatusOK),
		"logs":     page(t, d, cookie, "/dashboard/logs", http.StatusOK),
	}
	for name, body := range pages {
		if found := external.FindString(body); found != "" {
			t.Errorf("%s loads an external resource: %s", name, found)
		}
		contains(t, name, body, assetURL("dashboard.css"), assetURL("datastar.js"))
	}
	for _, name := range []string{"dashboard.css", "datastar.js"} {
		response := serve(d, http.MethodGet, assetURL(name), "", nil)
		if response.Code != http.StatusOK || !strings.Contains(response.Header().Get("Cache-Control"), "immutable") {
			t.Fatalf("%s: %d %q", name, response.Code, response.Header().Get("Cache-Control"))
		}
		if strings.Contains(response.Body.String(), "@import url(") {
			t.Errorf("%s imports another stylesheet", name)
		}
	}
	if response := serve(d, http.MethodGet, "/dashboard/assets/dashboard.css?v=old", "", nil); response.Header().Get("Cache-Control") != "no-cache" {
		t.Error("a stale asset version may be cached forever")
	}
	if csp := serve(d, http.MethodGet, "/dashboard/login", "", nil).Header().Get("Content-Security-Policy"); !strings.Contains(csp, "default-src 'self'") {
		t.Errorf("CSP = %q", csp)
	}
}

func TestThemeIsRemembered(t *testing.T) {
	d := newDashboard(t, history())
	cookie := signIn(t, d)
	response := serve(d, http.MethodPost, "/dashboard/theme", `{"theme":"light"}`, cookie, "Datastar-Request", "true")
	if response.Code != http.StatusNoContent {
		t.Fatalf("theme = %d", response.Code)
	}
	theme := response.Result().Cookies()[0]
	if theme.Name != themeCookie || theme.Value != "light" || !theme.HttpOnly || theme.SameSite != http.SameSiteStrictMode {
		t.Fatalf("theme cookie = %+v", theme)
	}
	request := httptest.NewRequest(http.MethodGet, "/dashboard", nil)
	request.AddCookie(cookie)
	request.AddCookie(theme)
	recorder := httptest.NewRecorder()
	d.ServeHTTP(recorder, request)
	body, _ := io.ReadAll(recorder.Body)
	contains(t, "light overview", string(body), `<html lang="en" class="light"`)
	if dark := page(t, d, cookie, "/dashboard", http.StatusOK); strings.Contains(dark, `class="light"`) {
		t.Error("dark is not the default theme")
	}
}

func TestDecisionsShowTheMixClassifierAndEveryDecision(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	cookie := signIn(t, d)
	body := page(t, d, cookie, "/dashboard/decisions", http.StatusOK)
	contains(t, "decisions", body, `aria-current="page"`, "How auto classifies prompts", `id="mix"`, "Latest prompts",
		"Find the race in the settlement service", "None recorded yet.", "Classifier", "jev-latest",
		testPolicy.Instructions.Question, "Everyday coding, small UI changes included.", "Not for:", "Urgent work.",
		"Hide the Reports menu entry.", "when in doubt", "Can ‹choice› do it, or does it need ‹next step›?", "deepseek-flash",
		testPolicy.Fingerprint(), "Expected cost per task", "$0.25", "Try a prompt",
		`@post('/dashboard/sse/try')`, `id="decisions-table"`, "Fix the flaky retry test", "1–2 of 2 decisions")
	if strings.Contains(body, "images=no") || strings.Contains(body, "client=") {
		t.Error("a prompt shows the router's images or client line")
	}

	body = page(t, d, cookie, "/dashboard/decisions?model=gpt-6.1-sol&decider=jev", http.StatusOK)
	if rows := body[strings.Index(body, `id="decisions-table"`):]; !strings.Contains(rows, "Fix the flaky retry test") ||
		strings.Contains(rows, "Find the race") {
		t.Fatal("the model filter did not narrow the decisions")
	}
	table := open(t, d, cookie, "/dashboard/sse/decisions", `{"model":"claude-opus-5-5","decider":""}`)
	if patch := table.until(`id="decisions-table"`); !strings.Contains(patch, "Find the race") || strings.Contains(patch, "Fix the flaky") {
		t.Fatalf("decision patch:\n%s", patch)
	}
	table.until(`replaceState({}, "", "/dashboard/decisions?model=claude-opus-5-5")`)
}

func TestTrialsShowWhatTheDecidersAreSent(t *testing.T) {
	d := newDashboard(t, history())
	cookie := signIn(t, d)
	response := serve(d, http.MethodPost, "/dashboard/sse/try", `{"prompt":"  Ocultar a opção do nav menu  ","images":true}`,
		cookie, "Datastar-Request", "true", "Content-Type", "application/json")
	body := response.Body.String()
	for _, want := range []string{`id="trial"`, "Would pin", "glm-5.3",
		"jev chose glm-5.3 (80%) but put 20% on stronger models, at least the 10% threshold, so it was asked to compare glm-5.3 with claude-opus-5-5",
		"jev compared glm-5.3 with claude-opus-5-5 and put 30% on claude-opus-5-5, so glm-5.3 stood", "glm-5.3 vs claude-opus-5-5",
		`&#34;pair&#34;: {`, "Sent to the deciders",
		`&#34;state&#34;: &#34;images=yes\n\nOcultar a opção do nav menu&#34;`, "jev-1.13.0", "Received"} {
		if !strings.Contains(body, want) {
			t.Errorf("trial lacks %q:\n%s", want, body)
		}
	}
	classifier := d.classifier.(*fakeClassifier)
	if len(classifier.tried) != 1 || classifier.tried[0] != "Ocultar a opção do nav menu" || !classifier.images {
		t.Fatalf("tried %q, images %t", classifier.tried, classifier.images)
	}
	empty := serve(d, http.MethodPost, "/dashboard/sse/try", `{"prompt":"   "}`, cookie, "Datastar-Request", "true", "Content-Type", "application/json")
	if strings.Contains(empty.Body.String(), "Would pin") || len(classifier.tried) != 1 {
		t.Fatal("an empty prompt was classified")
	}
}

func TestOperatorsRepinThreads(t *testing.T) {
	d := newDashboard(t, history())
	cookie := signIn(t, d)
	key := strings.Repeat("a", 64)
	response := serve(d, http.MethodPost, "/dashboard/sse/threads/"+key+"/pin", `{"repinModel":"gpt-6.1-sol","repinLevel":"high"}`,
		cookie, "Datastar-Request", "true", "Content-Type", "application/json")
	classifier := d.classifier.(*fakeClassifier)
	if body := response.Body.String(); len(classifier.overrides) != 1 || classifier.overrides[0] != key+" gpt-6.1-sol@high" ||
		!strings.Contains(body, "/dashboard/threads/"+key) {
		t.Fatalf("re-pin = %v:\n%s", classifier.overrides, body)
	}
	response = serve(d, http.MethodPost, "/dashboard/sse/threads/"+key+"/pin", `{"repinModel":"gpt-9"}`,
		cookie, "Datastar-Request", "true", "Content-Type", "application/json")
	if body := response.Body.String(); !strings.Contains(body, `id="repin-problem"`) || !strings.Contains(body, "is not a candidate") {
		t.Fatalf("a failed re-pin shows no problem:\n%s", body)
	}
	unsigned := serve(d, http.MethodPost, "/dashboard/sse/threads/"+key+"/pin", `{"repinModel":"gpt-6.1-sol"}`, nil,
		"Datastar-Request", "true", "Content-Type", "application/json")
	if unsigned.Code == http.StatusOK || len(classifier.overrides) != 1 {
		t.Fatalf("a signed-out re-pin answered %d", unsigned.Code)
	}
}

func TestOverviewShowsTheSubscriptionAccounts(t *testing.T) {
	d := newDashboard(t, history())
	body := page(t, d, signIn(t, d), "/dashboard", http.StatusOK)
	accountsCard := body[strings.Index(body, `id="accounts"`):]
	contains(t, "accounts", accountsCard, "Subscription accounts", "ChatGPT", "0 of 2 accounts in rotation", "Claude",
		"1 of 2 accounts in rotation", "No account can serve these models, so new threads skip", "gpt-6.1-sol",
		"filippo@example.com", "at limit in 4d", "at limit in 3h", "in rotation", "Weekly", "5-hour", "Weekly Opus",
		"93%", "100%", "35%", "CLIProxy priority -10")
	if strings.Contains(accountsCard, "nimbus") {
		t.Error("a window that does not limit routing is shown")
	}
	mario := accountsCard[strings.Index(accountsCard, "mario@example.com"):]
	if i, j := strings.Index(mario, "5-hour"), strings.Index(mario, "Weekly"); i > j {
		t.Error("the windows are not in order")
	}
}

func TestOverviewShowsTheConcurrencyGates(t *testing.T) {
	d := newDashboard(t, history())
	body := page(t, d, signIn(t, d), "/dashboard", http.StatusOK)
	gates := body[strings.Index(body, `id="limits"`):]
	contains(t, "concurrency", gates, "Concurrency", "glm-5.3", "3 / 3 running", "cap 3 of 4", "2 requests waiting",
		"7 throttles, last 5m ago", "new threads spill to", "gpt-6.1-sol", "bg-critical/30",
		"queue wait avg 3.20 s, p95 18.00 s", "12 of 80 requests")
}
