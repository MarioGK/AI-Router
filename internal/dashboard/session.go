package dashboard

import (
	"context"
	"crypto/rand"
	"errors"
	"mime"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/starfederation/datastar-go/datastar"
)

const (
	sessionCookie = "tsr_session"
	sessionTTL    = 12 * time.Hour
	themeCookie   = "tsr_theme"
	// A client address may fail to sign in maxFailures times within
	// failureWindow; then it waits for the window to pass.
	maxFailures   = 5
	failureWindow = 10 * time.Minute
	// maxThrottled bounds the client addresses the sign-in throttle keeps.
	maxThrottled     = 4096
	maxLoginBody     = 64 << 10
	maxThemeBody     = 4 << 10
	loginBodyTimeout = 10 * time.Second
	// maxLoginChecks bounds expensive password verification across clients.
	maxLoginChecks = 8
)

// sessions are the signed-in operators, kept in memory: a restart signs
// everyone out.
type sessions struct {
	mu        sync.Mutex
	expires   map[string]time.Time
	lifetimes map[string]sessionLifetime
	closed    bool
}

type sessionLifetime struct {
	ctx    context.Context
	cancel context.CancelFunc
	stop   func() bool
}

// start opens a session and returns its token, dropping expired ones.
func (s *sessions) start(now time.Time) string {
	token := rand.Text()
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ""
	}
	for other, expires := range s.expires {
		if !now.Before(expires) {
			s.endLocked(other)
		}
	}
	if s.expires == nil {
		s.expires = map[string]time.Time{}
	}
	if s.lifetimes == nil {
		s.lifetimes = map[string]sessionLifetime{}
	}
	ctx, cancel := context.WithDeadline(context.Background(), now.Add(sessionTTL))
	stop := context.AfterFunc(ctx, func() { s.end(token) })
	s.expires[token] = now.Add(sessionTTL)
	s.lifetimes[token] = sessionLifetime{ctx: ctx, cancel: cancel, stop: stop}
	return token
}

func (s *sessions) valid(token string, now time.Time) bool {
	_, ok := s.lifetime(token, now)
	return ok
}

func (s *sessions) lifetime(token string, now time.Time) (context.Context, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	expires, ok := s.expires[token]
	lifetime, held := s.lifetimes[token]
	if !ok || !held {
		return nil, false
	}
	if !now.Before(expires) || lifetime.ctx.Err() != nil {
		s.endLocked(token)
		return nil, false
	}
	return lifetime.ctx, true
}

func (s *sessions) end(token string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.endLocked(token)
}

func (s *sessions) endLocked(token string) {
	if lifetime, ok := s.lifetimes[token]; ok {
		lifetime.stop()
		lifetime.cancel()
		delete(s.lifetimes, token)
	}
	delete(s.expires, token)
}

func (s *sessions) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	for token := range s.lifetimes {
		s.endLocked(token)
	}
	clear(s.expires)
}

// throttle counts failed sign-ins per client address, in windows that open
// with an address's first failure.
type throttle struct {
	mu      sync.Mutex
	failed  map[string]failures
	active  int
	pending map[string]int
}

// acquireCheck reserves a credential check. Pending checks count against
// the address's failure limit.
func (t *throttle) acquireCheck(addr string, now time.Time) time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	f := t.failed[addr]
	if !now.Before(f.opened.Add(failureWindow)) {
		f = failures{}
	}
	if f.count+t.pending[addr] >= maxFailures {
		return max(time.Second, f.opened.Add(failureWindow).Sub(now))
	}
	if t.active >= maxLoginChecks {
		return time.Second
	}
	if t.pending == nil {
		t.pending = map[string]int{}
	}
	t.active++
	t.pending[addr]++
	return 0
}

func (t *throttle) releaseCheck(addr string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.active--
	t.pending[addr]--
	if t.pending[addr] == 0 {
		delete(t.pending, addr)
	}
}

// failures are an address's failed sign-ins since its window opened.
type failures struct {
	count  int
	opened time.Time
}

// fail records a failed sign-in of the address. A throttle full of
// addresses makes room first.
func (t *throttle) fail(addr string, now time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	f, ok := t.failed[addr]
	if !ok && len(t.failed) >= maxThrottled {
		t.prune(now)
	}
	if !now.Before(f.opened.Add(failureWindow)) {
		f = failures{opened: now}
	}
	f.count++
	t.failed[addr] = f
}

// forget clears the failures of an address that signed in.
func (t *throttle) forget(addr string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.failed, addr)
}

// prune drops the addresses whose window has passed, and the one whose
// window opened first when that leaves the throttle full.
func (t *throttle) prune(now time.Time) {
	oldest := ""
	for addr, f := range t.failed {
		switch {
		case !now.Before(f.opened.Add(failureWindow)):
			delete(t.failed, addr)
		case oldest == "" || f.opened.Before(t.failed[oldest].opened):
			oldest = addr
		}
	}
	if len(t.failed) >= maxThrottled {
		delete(t.failed, oldest)
	}
}

// clientAddress is who a sign-in counts against: the peer's IP, or its /64
// for IPv6, where one host holds a whole prefix. Forwarding headers are
// ignored, since any client can set them.
func clientAddress(r *http.Request) string {
	peer, err := netip.ParseAddrPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	ip := peer.Addr().Unmap().WithZone("")
	if ip.Is4() {
		return ip.String()
	}
	prefix, _ := ip.Prefix(64)
	return prefix.String()
}

func (d *Dashboard) signedIn(r *http.Request) bool {
	cookie, err := r.Cookie(sessionCookie)
	return err == nil && d.sessions.valid(cookie.Value, time.Now())
}

// private admits signed-in operators. Everyone else is sent to the sign-in
// page, except Datastar requests and streams, which get 401.
func (d *Dashboard) private(next http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if cookie, err := r.Cookie(sessionCookie); err == nil {
			if lifetime, ok := d.sessions.lifetime(cookie.Value, time.Now()); ok {
				ctx, cancel := context.WithCancel(r.Context())
				stop := context.AfterFunc(lifetime, cancel)
				defer stop()
				defer cancel()
				next(w, r.WithContext(ctx))
				return
			}
		}
		switch {
		case r.Header.Get("Datastar-Request") != "" || strings.HasPrefix(r.URL.Path, "/dashboard/sse/"):
			http.Error(w, "sign in to the dashboard", http.StatusUnauthorized)
		default:
			http.Redirect(w, r, "/dashboard/login?next="+url.QueryEscape(r.URL.RequestURI()), http.StatusSeeOther)
		}
	})
}

func (d *Dashboard) loginPage(w http.ResponseWriter, r *http.Request) {
	next := landing(r.URL.Query().Get("next"))
	if d.signedIn(r) {
		http.Redirect(w, r, next, http.StatusSeeOther)
		return
	}
	enabled := d.checker != nil && d.checker.Enabled()
	d.render(w, r, http.StatusOK, loginPage(newFrame(r, "Sign in", ""), next, enabled, false))
}

// login checks the dashboard username and password and opens a session.
// A client address that failed maxFailures times gets 429 until its window
// has passed, without its credentials being checked.
func (d *Dashboard) login(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	now, addr := time.Now(), clientAddress(r)
	controller := http.NewResponseController(w)
	deadlineErr := controller.SetReadDeadline(now.Add(loginBodyTimeout))
	if deadlineErr != nil && !errors.Is(deadlineErr, http.ErrNotSupported) {
		http.Error(w, "sign-in is unavailable", http.StatusServiceUnavailable)
		return
	}
	if r.Body != nil && r.Body != http.NoBody && r.ContentLength != 0 {
		mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil || mediaType != "application/x-www-form-urlencoded" {
			d.throttle.fail(addr, now)
			if r.ProtoMajor == 1 {
				w.Header().Set("Connection", "close")
			}
			http.Error(w, "invalid sign-in form", http.StatusUnsupportedMediaType)
			return
		}
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxLoginBody)
	if err := r.ParseForm(); err != nil {
		d.throttle.fail(addr, now)
		if r.ProtoMajor == 1 {
			w.Header().Set("Connection", "close")
		}
		status := http.StatusBadRequest
		if _, oversized := errors.AsType[*http.MaxBytesError](err); oversized {
			status = http.StatusRequestEntityTooLarge
		} else if errors.Is(err, os.ErrDeadlineExceeded) {
			status = http.StatusRequestTimeout
		}
		http.Error(w, "invalid sign-in form", status)
		return
	}
	if deadlineErr == nil {
		defer controller.SetReadDeadline(time.Time{})
	}
	if wait := d.throttle.acquireCheck(addr, time.Now()); wait > 0 {
		w.Header().Set("Retry-After", strconv.Itoa(int((wait+time.Second-1)/time.Second)))
		http.Error(w, "too many sign-ins; try again later", http.StatusTooManyRequests)
		return
	}
	defer d.throttle.releaseCheck(addr)
	next := landing(r.PostForm.Get("next"))
	if d.checker == nil || !d.checker.Valid(r.PostForm.Get("username"), r.PostForm.Get("password")) {
		d.throttle.fail(addr, now)
		d.log.Warn("dashboard sign-in rejected", "remote", r.RemoteAddr)
		enabled := d.checker != nil && d.checker.Enabled()
		d.render(w, r, http.StatusUnauthorized, loginPage(newFrame(r, "Sign in", ""), next, enabled, true))
		return
	}
	d.throttle.forget(addr)
	token := d.sessions.start(now)
	if token == "" {
		http.Error(w, "dashboard sign-in is unavailable", http.StatusServiceUnavailable)
		return
	}
	setCookie(w, r, sessionCookie, token, int(sessionTTL/time.Second))
	http.Redirect(w, r, next, http.StatusSeeOther)
}

func (d *Dashboard) logout(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if cookie, err := r.Cookie(sessionCookie); err == nil {
		d.sessions.end(cookie.Value)
	}
	setCookie(w, r, sessionCookie, "", -1)
	http.Redirect(w, r, "/dashboard/login", http.StatusSeeOther)
}

// theme remembers the operator's color theme.
func (d *Dashboard) theme(w http.ResponseWriter, r *http.Request) {
	var chosen struct {
		Theme string `json:"theme"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxThemeBody)
	if err := datastar.ReadSignals(r, &chosen); err != nil {
		status := http.StatusBadRequest
		if _, oversized := errors.AsType[*http.MaxBytesError](err); oversized {
			status = http.StatusRequestEntityTooLarge
		}
		http.Error(w, "invalid theme request", status)
		return
	}
	setCookie(w, r, themeCookie, themeName(chosen.Theme), 365*24*60*60)
	w.WriteHeader(http.StatusNoContent)
}

// setCookie sets a dashboard cookie for maxAge seconds; a negative age
// deletes it.
func setCookie(w http.ResponseWriter, r *http.Request, name, value string, maxAge int) {
	http.SetCookie(w, &http.Cookie{
		Name: name, Value: value, Path: "/dashboard", MaxAge: maxAge,
		HttpOnly: true, SameSite: http.SameSiteStrictMode,
		Secure: r.TLS != nil || strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https"),
	})
}

// landing is where a sign-in leads: the page asked for, when its cleaned
// path is one of the dashboard's, else the overview. A backslash is refused
// outright: browsers read it as a slash, so "/\host" leaves the origin.
func landing(next string) string {
	u, err := url.Parse(next)
	if err != nil || u.Scheme != "" || u.Host != "" || u.User != nil || strings.Contains(next, `\`) ||
		!strings.HasPrefix(u.Path, "/") {
		return "/dashboard"
	}
	u.Path = path.Clean(u.Path)
	if u.Path != "/dashboard" && !strings.HasPrefix(u.Path, "/dashboard/") {
		return "/dashboard"
	}
	return u.RequestURI()
}

func themeName(value string) string {
	if value == "light" {
		return "light"
	}
	return "dark"
}
