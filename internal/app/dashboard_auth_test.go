package app

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/config"
)

const (
	dashboardUsername     = "mariogk"
	dashboardPassword     = "dashboard-password-for-tests"
	dashboardPasswordHash = "$2a$04$vnPwv08lA97RHYt6oNFNA.RnsyMisEcyqpMR3uIhlpwM8sVCXd7sy"
)

func appDashboardRequest(t *testing.T, a *App, method, path string, form url.Values, cookies ...*http.Cookie) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequestWithContext(t.Context(), method, "https://router.test"+path, strings.NewReader(form.Encode()))
	if form != nil {
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	for _, cookie := range cookies {
		r.AddCookie(cookie)
	}
	w := httptest.NewRecorder()
	a.ServeHTTP(w, r)
	return w
}

func TestDashboardPasswordLoginIsSeparateFromUpstreamAuthentication(t *testing.T) {
	t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", dashboardPasswordHash)
	h := newPolicyHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "high"})

	r := httptest.NewRequestWithContext(t.Context(), http.MethodGet, "https://router.test/dashboard", nil)
	r.Header.Set("Authorization", "Bearer "+proxyKey)
	w := httptest.NewRecorder()
	h.app.ServeHTTP(w, r)
	if w.Code != http.StatusSeeOther || len(w.Result().Cookies()) != 0 {
		t.Fatalf("API key opened the dashboard: %d %s", w.Code, w.Body.String())
	}
	for _, form := range []url.Values{
		{"key": {proxyKey}},
		{"username": {dashboardUsername}, "password": {proxyKey}},
		{"username": {"wrong-operator"}, "password": {dashboardPassword}},
	} {
		w := appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/login", form)
		if w.Code != http.StatusUnauthorized || len(w.Result().Cookies()) != 0 {
			t.Fatalf("invalid dashboard credentials opened a session: %d %s", w.Code, w.Body.String())
		}
	}
	w = appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/login", url.Values{
		"username": {dashboardUsername}, "password": {dashboardPassword}, "next": {"/dashboard/threads"},
	})
	cookies := w.Result().Cookies()
	if w.Code != http.StatusSeeOther || w.Header().Get("Location") != "/dashboard/threads" || len(cookies) != 1 {
		t.Fatalf("password sign-in = %d, location %q, cookies %+v", w.Code, w.Header().Get("Location"), cookies)
	}
	if cookie := cookies[0]; !cookie.HttpOnly || !cookie.Secure || cookie.SameSite != http.SameSiteStrictMode || cookie.Path != "/dashboard" {
		t.Errorf("dashboard session cookie = %+v", cookie)
	}
	if w := appDashboardRequest(t, h.app, http.MethodGet, "/dashboard", nil, cookies[0]); w.Code != http.StatusOK {
		t.Fatalf("signed-in dashboard = %d %s", w.Code, w.Body.String())
	}
	if w := appDashboardRequest(t, h.app, http.MethodPost, "/v1/messages", nil, cookies[0]); w.Code != http.StatusUnauthorized {
		t.Fatalf("dashboard session authorized the Messages API: %d %s", w.Code, w.Body.String())
	}
	if len(h.transport.classified) != 0 || len(h.transport.routed) != 0 {
		t.Fatal("dashboard authentication made outbound model requests")
	}
	h.checkResponse(h.messages("password-api-still-works", "auto(low)", "Implement the urgent handler fix."), "gpt-6.1-sol", "high")
}

func TestDashboardPasswordLoginIsDisabledWithoutHash(t *testing.T) {
	t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", "")
	h := newPolicyHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "high"})
	w := appDashboardRequest(t, h.app, http.MethodGet, "/dashboard/login", nil)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "Dashboard sign-in is unavailable.") || strings.Contains(w.Body.String(), "<form") {
		t.Fatalf("disabled sign-in page = %d %s", w.Code, w.Body.String())
	}
	for _, form := range []url.Values{
		{"key": {proxyKey}},
		{"username": {dashboardUsername}, "password": {proxyKey}},
		{"username": {dashboardUsername}, "password": {dashboardPassword}},
	} {
		w := appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/login", form)
		if w.Code != http.StatusUnauthorized || len(w.Result().Cookies()) != 0 {
			t.Fatalf("disabled password login opened a session: %d %s", w.Code, w.Body.String())
		}
	}
	h.checkResponse(h.messages("disabled-dashboard-api", "auto", "Implement the urgent handler fix."), "gpt-6.1-sol", "high")
}

func TestDashboardPasswordLoginUsesConfiguredUsernameAndSecret(t *testing.T) {
	t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", "")
	t.Setenv("APP_DASHBOARD_PASSWORD_HASH", dashboardPasswordHash)
	h := newPolicyHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "high"}, func(cfg *config.Config) {
		cfg.Dashboard.Username = "routing-operator"
		cfg.Dashboard.PasswordHashEnv = "APP_DASHBOARD_PASSWORD_HASH"
	})
	w := appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/login", url.Values{
		"username": {dashboardUsername}, "password": {dashboardPassword},
	})
	if w.Code != http.StatusUnauthorized || len(w.Result().Cookies()) != 0 {
		t.Fatalf("default username bypassed configured operator: %d %s", w.Code, w.Body.String())
	}
	w = appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/login", url.Values{
		"username": {"routing-operator"}, "password": {dashboardPassword},
	})
	if w.Code != http.StatusSeeOther || len(w.Result().Cookies()) != 1 {
		t.Fatalf("configured dashboard credentials = %d %s", w.Code, w.Body.String())
	}
}

type dashboardStreamRecorder struct {
	*httptest.ResponseRecorder
	started chan struct{}
	once    sync.Once
}

func (w *dashboardStreamRecorder) Flush() {
	w.ResponseRecorder.Flush()
	w.once.Do(func() { close(w.started) })
}

func TestAppCloseRevokesDashboardSessionAndEndsItsStream(t *testing.T) {
	t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", dashboardPasswordHash)
	h := newPolicyHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "high"})
	login := appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/login", url.Values{
		"username": {dashboardUsername}, "password": {dashboardPassword},
	})
	cookies := login.Result().Cookies()
	if login.Code != http.StatusSeeOther || len(cookies) != 1 {
		t.Fatalf("password sign-in = %d, cookies %+v", login.Code, cookies)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	r := httptest.NewRequestWithContext(ctx, http.MethodGet, "https://router.test/dashboard/sse/feed", nil)
	r.AddCookie(cookies[0])
	w := &dashboardStreamRecorder{ResponseRecorder: httptest.NewRecorder(), started: make(chan struct{})}
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		h.app.ServeHTTP(w, r)
	}()
	select {
	case <-w.started:
	case <-time.After(5 * time.Second):
		t.Fatal("dashboard stream never started")
	}
	select {
	case <-finished:
		t.Fatal("dashboard stream ended before App.Close")
	default:
	}
	if err := h.app.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("App.Close left the dashboard stream running")
	}
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "text/event-stream" {
		t.Errorf("dashboard stream = %d, content type %q", w.Code, w.Header().Get("Content-Type"))
	}
	if w := appDashboardRequest(t, h.app, http.MethodGet, "/dashboard", nil, cookies[0]); w.Code != http.StatusSeeOther {
		t.Fatalf("App.Close retained the dashboard session: %d %s", w.Code, w.Body.String())
	}
	if w := appDashboardRequest(t, h.app, http.MethodGet, "/dashboard/sse/feed", nil, cookies[0]); w.Code != http.StatusUnauthorized {
		t.Fatalf("App.Close retained stream authorization: %d %s", w.Code, w.Body.String())
	}
}
