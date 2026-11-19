package app

import (
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
)

func TestDashboardShowsRoutedThreadsBehindItsOwnSignIn(t *testing.T) {
	t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", dashboardPasswordHash)
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	readAll(t, h.claude("ses_dashboard", "design the storage layer"))

	browser := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	response, err := browser.Get(h.server.URL + "/dashboard")
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusSeeOther || response.Header.Get("Location") != "/dashboard/login?next=%2Fdashboard" {
		t.Fatalf("GET /dashboard without a session = %d to %q", response.StatusCode, response.Header.Get("Location"))
	}
	response, err = browser.PostForm(h.server.URL+"/dashboard/login", url.Values{
		"username": {dashboardUsername}, "password": {dashboardPassword},
	})
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusSeeOther || len(response.Cookies()) != 1 {
		t.Fatalf("sign-in = %d", response.StatusCode)
	}
	session := response.Cookies()[0]
	page := func(path string) string {
		t.Helper()
		request, _ := http.NewRequest(http.MethodGet, h.server.URL+path, nil)
		request.AddCookie(session)
		response, err := browser.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		body, _ := io.ReadAll(response.Body)
		if response.StatusCode != http.StatusOK {
			t.Fatalf("GET %s = %d: %s", path, response.StatusCode, body)
		}
		return string(body)
	}

	threads, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || len(threads.Threads) != 1 {
		t.Fatalf("threads = %+v, %v", threads, err)
	}
	key := threads.Threads[0].Key
	// The request log is written off the request path.
	for deadline := time.Now().Add(5 * time.Second); !strings.Contains(page("/dashboard/logs"), key[:12]); time.Sleep(10 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the logs never showed the routed request")
		}
	}
	for path, wants := range map[string][]string{
		"/dashboard":                 {"Pinned threads", "gpt-6.1-sol", "Requests per hour"},
		"/dashboard/threads":         {key[:12], "jev"},
		"/dashboard/threads/" + key:  {key, "by jev", "100%", "decided"},
		"/dashboard/threads?page=99": {key[:12]},
	} {
		body := page(path)
		for _, want := range wants {
			if !strings.Contains(body, want) {
				t.Errorf("%s lacks %q", path, want)
			}
		}
	}
}
