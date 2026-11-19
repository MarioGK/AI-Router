package dashboard

import (
	"bytes"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

func TestEveryRouteButSignInNeedsASession(t *testing.T) {
	d := newDashboard(t, history())
	for _, target := range []string{"/dashboard", "/dashboard/", "/dashboard/threads?model=glm-5.3", "/dashboard/threads/abc", "/dashboard/decisions", "/dashboard/logs"} {
		response := serve(d, http.MethodGet, target, "", nil)
		if want := "/dashboard/login?next=" + url.QueryEscape(target); response.Code != http.StatusSeeOther || response.Header().Get("Location") != want {
			t.Errorf("GET %s = %d to %q, want a redirect to %s", target, response.Code, response.Header().Get("Location"), want)
		}
	}
	for _, target := range []string{"/dashboard/sse/feed", "/dashboard/sse/mix", "/dashboard/sse/threads", "/dashboard/sse/decisions", "/dashboard/sse/logs"} {
		if response := serve(d, http.MethodGet, target, "", nil); response.Code != http.StatusUnauthorized {
			t.Errorf("GET %s = %d, want 401", target, response.Code)
		}
	}
	for _, target := range []string{"/dashboard/theme", "/dashboard/sse/try"} {
		if response := serve(d, http.MethodPost, target, `{"prompt":"x"}`, nil, "Datastar-Request", "true"); response.Code != http.StatusUnauthorized {
			t.Errorf("POST %s without a session = %d, want 401", target, response.Code)
		}
	}
	expired := &http.Cookie{Name: sessionCookie, Value: "forged"}
	if response := serve(d, http.MethodGet, "/dashboard", "", expired); response.Code != http.StatusSeeOther {
		t.Errorf("an unknown session = %d, want a redirect", response.Code)
	}
}

func TestSignInChecksTheCredentials(t *testing.T) {
	d := newDashboard(t, history())
	response := serve(d, http.MethodPost, "/dashboard/login", url.Values{"username": {dashboardUsername}, "password": {"wrong"}, "next": {"/dashboard/logs"}}.Encode(), nil)
	if response.Code != http.StatusUnauthorized || len(response.Result().Cookies()) != 0 {
		t.Fatalf("a wrong password = %d with cookies %v", response.Code, response.Result().Cookies())
	}
	contains(t, "rejected sign-in", response.Body.String(), "Invalid username or password.", `value="/dashboard/logs"`)

	response = serve(d, http.MethodPost, "/dashboard/login", url.Values{"username": {dashboardUsername}, "password": {dashboardPassword}, "next": {"/dashboard/logs?model=glm-5.3"}}.Encode(), nil,
		"X-Forwarded-Proto", "https")
	if response.Code != http.StatusSeeOther || response.Header().Get("Location") != "/dashboard/logs?model=glm-5.3" {
		t.Fatalf("sign-in = %d to %q", response.Code, response.Header().Get("Location"))
	}
	cookie := response.Result().Cookies()[0]
	if cookie.Name != sessionCookie || len(cookie.Value) < 26 || !cookie.HttpOnly || !cookie.Secure ||
		cookie.SameSite != http.SameSiteStrictMode || cookie.Path != "/dashboard" || cookie.MaxAge != 12*60*60 {
		t.Fatalf("session cookie = %+v", cookie)
	}
	page(t, d, cookie, "/dashboard", http.StatusOK)
	if location := serve(d, http.MethodGet, "/dashboard/login", "", cookie).Header().Get("Location"); location != "/dashboard" {
		t.Errorf("the sign-in page for a signed-in operator leads to %q", location)
	}

	plain := serve(d, http.MethodPost, "/dashboard/login", url.Values{"username": {dashboardUsername}, "password": {dashboardPassword}}.Encode(), nil).Result().Cookies()[0]
	if plain.Secure {
		t.Error("the cookie of a plain HTTP sign-in is Secure")
	}
}

func TestInvalidCredentialsAndKeysCannotOpenASession(t *testing.T) {
	for name, form := range map[string]url.Values{
		"wrong username":          {"username": {"wrong-operator"}, "password": {dashboardPassword}},
		"wrong password":          {"username": {dashboardUsername}, "password": {"incorrect-secret"}},
		"missing username":        {"password": {dashboardPassword}},
		"legacy key":              {"key": {dashboardPassword}},
		"key instead of password": {"username": {dashboardUsername}, "key": {dashboardPassword}},
		"oversized password":      {"username": {dashboardUsername}, "password": {strings.Repeat("p", 73)}},
	} {
		t.Run(name, func(t *testing.T) {
			d := newDashboard(t, history())
			var logs bytes.Buffer
			d.log = slog.New(slog.NewTextHandler(&logs, nil))
			response := serve(d, http.MethodPost, "/dashboard/login", form.Encode(), nil,
				"X-Api-Key", dashboardPassword, "Authorization", "Bearer "+dashboardPassword)
			if response.Code != http.StatusUnauthorized || len(response.Result().Cookies()) != 0 {
				t.Fatalf("invalid credentials = %d with cookies %v", response.Code, response.Result().Cookies())
			}
			contains(t, "invalid credentials", response.Body.String(), "Invalid username or password.")
			for _, value := range []string{dashboardUsername, dashboardPassword, "wrong-operator", "incorrect-secret"} {
				if strings.Contains(logs.String(), value) || strings.Contains(response.Body.String(), value) {
					t.Fatal("credentials were reflected in the rejection or its log")
				}
			}
		})
	}
}

func TestLoginFormUsesCredentialsAndFailsClosedWithoutAChecker(t *testing.T) {
	d := newDashboard(t, history())
	body := page(t, d, nil, "/dashboard/login", http.StatusOK)
	contains(t, "credential form", body, `name="username"`, `autocomplete="username"`, `name="password"`, `autocomplete="current-password"`)
	if strings.Contains(body, `name="key"`) || strings.Contains(body, "CLIProxy client key") {
		t.Fatal("the old key login is still present")
	}
	d.checker = nil
	body = page(t, d, nil, "/dashboard/login", http.StatusOK)
	contains(t, "disabled sign-in", body, "Dashboard sign-in is unavailable.")
	if strings.Contains(body, "<form") {
		t.Fatal("disabled sign-in still offers a form")
	}
	response := serve(d, http.MethodPost, "/dashboard/login", url.Values{"username": {dashboardUsername}, "password": {dashboardPassword}}.Encode(), nil)
	if response.Code != http.StatusUnauthorized || len(response.Result().Cookies()) != 0 {
		t.Fatalf("disabled sign-in = %d with cookies %v", response.Code, response.Result().Cookies())
	}
}

func TestLoginRejectsMalformedOversizedAndQueryCredentials(t *testing.T) {
	credentials := url.Values{"username": {dashboardUsername}, "password": {dashboardPassword}}.Encode()
	for _, test := range []struct {
		name   string
		target string
		body   string
		status int
	}{
		{"malformed form", "/dashboard/login", credentials + "&broken=%ZZ", http.StatusBadRequest},
		{"oversized form", "/dashboard/login", credentials + "&padding=" + strings.Repeat("x", maxLoginBody), http.StatusRequestEntityTooLarge},
		{"query credentials", "/dashboard/login?" + credentials, "", http.StatusUnauthorized},
	} {
		t.Run(test.name, func(t *testing.T) {
			d := newDashboard(t, history())
			response := serve(d, http.MethodPost, test.target, test.body, nil)
			if response.Code != test.status || len(response.Result().Cookies()) != 0 {
				t.Fatalf("rejected form = %d with cookies %v, want %d", response.Code, response.Result().Cookies(), test.status)
			}
			if d.throttle.active != 0 || len(d.throttle.pending) != 0 {
				t.Fatal("a rejected form retained a sign-in reservation")
			}
		})
	}
}

func TestLoginRefusesBodyFormatsThatParseFormDoesNotRead(t *testing.T) {
	d := newDashboard(t, history())
	body := url.Values{"username": {dashboardUsername}, "password": {dashboardPassword}}.Encode()
	request := httptest.NewRequest(http.MethodPost, "/dashboard/login", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	d.ServeHTTP(response, request)
	if response.Code != http.StatusUnsupportedMediaType || response.Header().Get("Connection") != "close" || len(response.Result().Cookies()) != 0 {
		t.Fatalf("unparsed body = %d, Connection %q, cookies %v", response.Code, response.Header().Get("Connection"), response.Result().Cookies())
	}
	if d.throttle.active != 0 || len(d.throttle.pending) != 0 {
		t.Fatal("an unparsed body reserved a verification slot")
	}
}

type stalledLoginBody struct {
	deadline time.Time
	reserved func() bool
}

func (b *stalledLoginBody) Read([]byte) (int, error) {
	if b.deadline.IsZero() || b.reserved() {
		return 0, os.ErrInvalid
	}
	time.Sleep(time.Until(b.deadline))
	return 0, os.ErrDeadlineExceeded
}

func (*stalledLoginBody) Close() error { return nil }

type loginDeadlineRecorder struct {
	*httptest.ResponseRecorder
	body *stalledLoginBody
}

func (w *loginDeadlineRecorder) SetReadDeadline(deadline time.Time) error {
	w.body.deadline = deadline
	return nil
}

func TestSlowLoginBodyTimesOutWithoutReservingChecks(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		d := newDashboard(t, history())
		defer d.Close()
		body := &stalledLoginBody{reserved: func() bool { return d.throttle.active != 0 }}
		request := httptest.NewRequest(http.MethodPost, "/dashboard/login", nil).WithContext(t.Context())
		request.Body = body
		request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		response := &loginDeadlineRecorder{ResponseRecorder: httptest.NewRecorder(), body: body}
		start := time.Now()
		d.ServeHTTP(response, request)
		if response.Code != http.StatusRequestTimeout || time.Since(start) != loginBodyTimeout || !body.deadline.Equal(start.Add(loginBodyTimeout)) || response.Header().Get("Connection") != "close" {
			t.Fatalf("slow body = %d after %v; deadline = %v; Connection = %q", response.Code, time.Since(start), body.deadline, response.Header().Get("Connection"))
		}
		if d.throttle.active != 0 || len(d.throttle.pending) != 0 || len(response.Result().Cookies()) != 0 {
			t.Fatal("slow body retained a verification slot or opened a session")
		}
		if good := serve(d, http.MethodPost, "/dashboard/login", url.Values{"username": {dashboardUsername}, "password": {dashboardPassword}}.Encode(), nil); good.Code != http.StatusSeeOther {
			t.Fatalf("sign-in after a stalled body = %d", good.Code)
		}
	})
}

func TestLoginReservationsBoundConcurrentAttempts(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		th := &throttle{failed: map[string]failures{}}
		release := make(chan struct{})
		finish := sync.OnceFunc(func() { close(release) })
		defer finish()
		var wg sync.WaitGroup
		for range 30 {
			wg.Go(func() {
				if th.acquireCheck("same-address", time.Now()) != 0 {
					return
				}
				<-release
				th.releaseCheck("same-address")
			})
		}
		synctest.Wait()
		if th.active != maxFailures || th.pending["same-address"] != maxFailures {
			t.Fatalf("concurrent address reservations = %d, %d", th.active, th.pending["same-address"])
		}
		for i := range maxLoginChecks - maxFailures {
			addr := strconv.Itoa(i)
			if wait := th.acquireCheck(addr, time.Now()); wait != 0 {
				t.Fatalf("available global slot refused: %v", wait)
			}
			defer th.releaseCheck(addr)
		}
		if wait := th.acquireCheck("other-address", time.Now()); wait <= 0 {
			t.Fatal("global verification limit was bypassed")
		}
		finish()
		wg.Wait()
	})
}

type sessionStreamRecorder struct {
	*httptest.ResponseRecorder
	mu      sync.Mutex
	flushed chan struct{}
}

func (r *sessionStreamRecorder) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.ResponseRecorder.Write(p)
}

func (r *sessionStreamRecorder) Flush() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ResponseRecorder.Flush()
	select {
	case r.flushed <- struct{}{}:
	default:
	}
}

func (r *sessionStreamRecorder) snapshot() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.Body.String()
}

func TestAuthenticatedFeedEndsWhenTheSessionEnds(t *testing.T) {
	for _, ending := range []string{"logout", "expiry", "close"} {
		t.Run(ending, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				d := newDashboard(t, history())
				defer d.Close()
				d.poll = time.Hour
				cookie := signIn(t, d)
				request := httptest.NewRequest(http.MethodGet, "/dashboard/sse/feed", nil).WithContext(t.Context())
				request.AddCookie(cookie)
				response := &sessionStreamRecorder{ResponseRecorder: httptest.NewRecorder(), flushed: make(chan struct{}, 1)}
				ended := make(chan struct{})
				go func() {
					d.ServeHTTP(response, request)
					close(ended)
				}()
				synctest.Wait()
				select {
				case <-response.flushed:
				default:
					t.Fatal("authenticated feed did not stream")
				}
				switch ending {
				case "logout":
					serve(d, http.MethodPost, "/dashboard/logout", "", cookie)
				case "expiry":
					time.Sleep(sessionTTL)
				case "close":
					d.Close()
				}
				synctest.Wait()
				select {
				case <-ended:
				default:
					t.Fatal("feed retained access after its session ended")
				}
				before := response.snapshot()
				time.Sleep(2 * d.poll)
				synctest.Wait()
				if response.snapshot() != before {
					t.Fatal("ended session received more stream data")
				}
				if got := serve(d, http.MethodGet, "/dashboard/sse/feed", "", cookie); got.Code != http.StatusUnauthorized {
					t.Fatalf("ended session reconnect = %d, want 401", got.Code)
				}
				if len(d.sessions.expires) != 0 || len(d.sessions.lifetimes) != 0 {
					t.Fatal("ended session retained its timer or token")
				}
			})
		})
	}
}

func TestSignInOnlyLeadsIntoTheDashboard(t *testing.T) {
	for next, want := range map[string]string{
		"":                                    "/dashboard",
		"/dashboard/threads/abc":              "/dashboard/threads/abc",
		"/dashboard/logs?model=glm-5.3":       "/dashboard/logs?model=glm-5.3",
		"/dashboard/./threads//abc/":          "/dashboard/threads/abc",
		"https://evil.example":                "/dashboard",
		"//evil.example":                      "/dashboard",
		"/dashboardx":                         "/dashboard",
		"/stats":                              "/dashboard",
		"dashboard/threads":                   "/dashboard",
		`/dashboard/../\evil.example`:         "/dashboard",
		`/dashboard/\evil.example`:            "/dashboard",
		"/dashboard/../stats":                 "/dashboard",
		"/dashboard/..%2F..%2Fevil.example":   "/dashboard",
		"/dashboard/%2e%2e/%5Cevil.example":   "/dashboard",
		"/dashboard/threads/a%2Fb?page=2#top": "/dashboard/threads/a%2Fb?page=2",
	} {
		if got := landing(next); got != want {
			t.Errorf("landing(%q) = %q, want %q", next, got, want)
		}
	}
}

func TestSignInNeverLeadsOffTheOrigin(t *testing.T) {
	d := newDashboard(t, history())
	cookie := signIn(t, d)
	for _, next := range []string{`/dashboard/../\evil.example`, `/dashboard/../../\\evil.example/x`, "/dashboard/..%2F%5Cevil.example"} {
		posted := serve(d, http.MethodPost, "/dashboard/login", url.Values{"username": {dashboardUsername}, "password": {dashboardPassword}, "next": {next}}.Encode(), nil)
		visited := serve(d, http.MethodGet, "/dashboard/login?next="+url.QueryEscape(next), "", cookie)
		for _, response := range []*httptest.ResponseRecorder{posted, visited} {
			if location := response.Header().Get("Location"); response.Code != http.StatusSeeOther || location != "/dashboard" {
				t.Errorf("sign-in with next %q = %d to %q, want the overview", next, response.Code, location)
			}
		}
	}
}

func TestCrossOriginPostsAreRefused(t *testing.T) {
	d := newDashboard(t, history())
	cookie := signIn(t, d)
	for _, target := range []string{"/dashboard/login", "/dashboard/logout", "/dashboard/theme", "/dashboard/sse/try"} {
		for _, header := range [][]string{
			{"Sec-Fetch-Site", "cross-site"},
			{"Sec-Fetch-Site", "same-site"},
			{"Origin", "https://pages.ai-router.example"},
			{"Origin", "null"},
		} {
			response := serve(d, http.MethodPost, target, `{"prompt":"x"}`, cookie, header...)
			if response.Code != http.StatusForbidden {
				t.Errorf("POST %s with %s: %s = %d, want 403", target, header[0], header[1], response.Code)
			}
		}
	}
	if len(d.classifier.(*fakeClassifier).tried) != 0 {
		t.Error("a refused trial reached the classifier")
	}

	// The dashboard's own pages, and clients that send neither header, pass.
	for _, header := range [][]string{{"Sec-Fetch-Site", "same-origin"}, {"Origin", "http://example.com"}, nil} {
		form := url.Values{"username": {dashboardUsername}, "password": {dashboardPassword}}.Encode()
		if response := serve(d, http.MethodPost, "/dashboard/login", form, nil, header...); response.Code != http.StatusSeeOther {
			t.Errorf("same-origin sign-in with %v = %d, want 303", header, response.Code)
		}
	}
	// Links from elsewhere still open pages.
	page(t, d, cookie, "/dashboard/logs", http.StatusOK)
	if response := serve(d, http.MethodGet, "/dashboard/logs", "", cookie, "Sec-Fetch-Site", "cross-site"); response.Code != http.StatusOK {
		t.Errorf("a cross-site link to a page = %d, want 200", response.Code)
	}
}

func TestFailedSignInsAreThrottledPerAddress(t *testing.T) {
	d := newDashboard(t, history())
	attempt := func(password, remote string, header ...string) *httptest.ResponseRecorder {
		request := httptest.NewRequest(http.MethodPost, "/dashboard/login", strings.NewReader(url.Values{"username": {dashboardUsername}, "password": {password}}.Encode()))
		request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		request.RemoteAddr = remote
		for i := 0; i < len(header); i += 2 {
			request.Header.Set(header[i], header[i+1])
		}
		response := httptest.NewRecorder()
		d.ServeHTTP(response, request)
		return response
	}
	for i := range maxFailures {
		if response := attempt("wrong", "203.0.113.7:4000"); response.Code != http.StatusUnauthorized {
			t.Fatalf("failed sign-in %d = %d, want 401", i+1, response.Code)
		}
	}
	for _, password := range []string{"wrong", dashboardPassword} {
		response := attempt(password, "203.0.113.7:4001", "X-Forwarded-For", "198.51.100.1")
		if seconds, _ := strconv.Atoi(response.Header().Get("Retry-After")); response.Code != http.StatusTooManyRequests ||
			seconds < 1 || seconds > int(failureWindow/time.Second) {
			t.Errorf("sign-in past the limit = %d, Retry-After %q", response.Code, response.Header().Get("Retry-After"))
		}
	}
	if response := attempt(dashboardPassword, "198.51.100.1:4000"); response.Code != http.StatusSeeOther {
		t.Errorf("another address signing in = %d, want 303", response.Code)
	}

	// IPv6 addresses count by their /64.
	for i := range maxFailures {
		attempt("wrong", "[2001:db8:1:2::"+strconv.Itoa(i+1)+"]:4000")
	}
	if response := attempt(dashboardPassword, "[2001:db8:1:2::ff]:4000"); response.Code != http.StatusTooManyRequests {
		t.Errorf("a throttled /64 signing in from a new address = %d, want 429", response.Code)
	}
}

func TestThrottleWindowsPassAndStayBounded(t *testing.T) {
	th := &throttle{failed: map[string]failures{}}
	start := time.Now()
	for range maxFailures {
		th.fail("a", start)
	}
	if wait := th.acquireCheck("a", start.Add(time.Minute)); wait != failureWindow-time.Minute {
		t.Fatalf("wait = %v, want the rest of the window", wait)
	}
	if wait := th.acquireCheck("a", start.Add(failureWindow)); wait != 0 {
		t.Fatalf("wait after the window = %v", wait)
	}
	th.releaseCheck("a")
	th.fail("a", start.Add(failureWindow))
	if th.failed["a"].count != 1 {
		t.Fatalf("a failure after the window counts %d", th.failed["a"].count)
	}
	th.forget("a")
	if _, kept := th.failed["a"]; kept {
		t.Fatal("a sign-in kept the address's failures")
	}

	for i := range maxThrottled + 10 {
		th.fail(strconv.Itoa(i), start.Add(time.Duration(i)*time.Millisecond))
	}
	if len(th.failed) != maxThrottled {
		t.Fatalf("the throttle keeps %d addresses, want at most %d", len(th.failed), maxThrottled)
	}
	if _, kept := th.failed["0"]; kept {
		t.Fatal("a full throttle dropped a newer address before the oldest")
	}
	th.fail("late", start.Add(failureWindow+time.Hour))
	if len(th.failed) != 1 {
		t.Fatalf("expired addresses were kept: %d", len(th.failed))
	}
}

func TestSignOutEndsTheSession(t *testing.T) {
	d := newDashboard(t, history())
	cookie := signIn(t, d)
	response := serve(d, http.MethodPost, "/dashboard/logout", "", cookie)
	if response.Code != http.StatusSeeOther || response.Header().Get("Location") != "/dashboard/login" {
		t.Fatalf("sign-out = %d to %q", response.Code, response.Header().Get("Location"))
	}
	if cleared := response.Result().Cookies()[0]; cleared.MaxAge >= 0 || cleared.Value != "" {
		t.Errorf("sign-out cookie = %+v", cleared)
	}
	if response := serve(d, http.MethodGet, "/dashboard", "", cookie); response.Code != http.StatusSeeOther {
		t.Errorf("an ended session still opens the dashboard: %d", response.Code)
	}
}

func TestSessionsExpire(t *testing.T) {
	s := &sessions{expires: map[string]time.Time{}}
	defer s.close()
	start := time.Now()
	old := s.start(start)
	if !s.valid(old, start.Add(sessionTTL-time.Second)) || s.valid(old, start.Add(sessionTTL)) {
		t.Fatal("a session does not last exactly its TTL")
	}
	fresh := s.start(start.Add(sessionTTL))
	if _, kept := s.expires[old]; kept || fresh == old || strings.TrimSpace(fresh) != fresh {
		t.Fatalf("starting a session kept the expired one: %v", s.expires)
	}
}

func TestThemeRejectsOversizedBodiesWithoutACookie(t *testing.T) {
	d := newDashboard(t, history())
	response := serve(d, http.MethodPost, "/dashboard/theme", `{"theme":"`+strings.Repeat("x", maxThemeBody)+`"}`, signIn(t, d))
	if response.Code != http.StatusRequestEntityTooLarge || len(response.Result().Cookies()) != 0 {
		t.Fatalf("oversized theme = %d with cookies %v", response.Code, response.Result().Cookies())
	}
}
