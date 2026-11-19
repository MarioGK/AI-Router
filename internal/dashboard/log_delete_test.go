package dashboard

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
)

func TestDeleteLogsRequiresSessionOriginAndConfirmation(t *testing.T) {
	for _, c := range []struct {
		name, form, origin string
		session            bool
		want               int
	}{
		{"signed out", "age=all&confirm=delete", "", false, http.StatusSeeOther},
		{"cross origin", "age=all&confirm=delete", "https://other.example", true, http.StatusForbidden},
		{"unconfirmed", "age=all", "", true, http.StatusBadRequest},
		{"invalid age", "age=bad&confirm=delete", "", true, http.StatusBadRequest},
		{"duplicate age", "age=all&age=24h&confirm=delete", "", true, http.StatusBadRequest},
		{"unexpected field", "age=all&confirm=delete&extra=x", "", true, http.StatusBadRequest},
	} {
		t.Run(c.name, func(t *testing.T) {
			source := history()
			d := newDashboard(t, source)
			r := httptest.NewRequest(http.MethodPost, "/dashboard/logs/delete", strings.NewReader(c.form))
			r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			if c.session {
				r.AddCookie(signIn(t, d))
			}
			if c.origin != "" {
				r.Header.Set("Origin", c.origin)
			}
			w := httptest.NewRecorder()
			d.ServeHTTP(w, r)
			if w.Code != c.want || !source.pruneBefore.IsZero() || d.logRevision.Load() != 0 {
				t.Fatalf("delete = %d, cutoff %v, revision %d", w.Code, source.pruneBefore, d.logRevision.Load())
			}
		})
	}
}

func TestDeleteLogsAgeFiltersAndPartialFailure(t *testing.T) {
	for _, c := range []struct {
		age    string
		window time.Duration
	}{
		{"all", 0}, {"24h", 24 * time.Hour}, {"7d", 7 * 24 * time.Hour}, {"30d", 30 * 24 * time.Hour},
	} {
		t.Run(c.age, func(t *testing.T) {
			source := history()
			source.pruneCount = 12
			d := newDashboard(t, source)
			before := time.Now().Add(-c.window)
			form := url.Values{"age": {c.age}, "confirm": {"delete"}, "model": {"glm-5.3"}, "status": {"errors"}, "search": {"invoice parser"}}
			w := serve(d, http.MethodPost, "/dashboard/logs/delete", form.Encode(), signIn(t, d))
			u, err := url.Parse(w.Header().Get("Location"))
			if w.Code != http.StatusSeeOther || err != nil || u.Query().Get("deleted") != "12" ||
				u.Query().Get("search") != "invoice parser" || u.Query().Get("status") != "errors" ||
				source.pruneBefore.Before(before) || source.pruneBefore.After(time.Now().Add(-c.window)) || d.logRevision.Load() != 1 {
				t.Fatalf("delete = %d %s, cutoff %v, revision %d", w.Code, w.Header().Get("Location"), source.pruneBefore, d.logRevision.Load())
			}
		})
	}
	source := history()
	source.pruneCount, source.pruneErr = 3, errors.New("disk unavailable")
	d := newDashboard(t, source)
	w := serve(d, http.MethodPost, "/dashboard/logs/delete", "age=all&confirm=delete", signIn(t, d))
	if w.Code != http.StatusInternalServerError || !strings.Contains(w.Body.String(), "Deleted 3 request logs before") || d.logRevision.Load() != 1 {
		t.Fatalf("partial deletion = %d: %s", w.Code, w.Body)
	}
}

func TestTailReloadsAfterDeletionAndIDReuse(t *testing.T) {
	source := &fakeSource{requests: []pin.RequestLog{{ID: 50}, {ID: 51}}}
	d := newDashboard(t, source)
	tail := &tail{source: source, revision: d.logRevision.Load}
	tail.load(t.Context())
	source.mu.Lock()
	source.requests = nil
	source.mu.Unlock()
	d.logRevision.Add(1)
	if !tail.poll(t.Context()) || len(tail.rows) != 0 || tail.after != 0 {
		t.Fatalf("deleted rows survived: %+v", tail)
	}
	source.log(pin.RequestLog{ID: 1})
	if !tail.poll(t.Context()) || len(tail.rows) != 1 || tail.rows[0].ID != 1 {
		t.Fatalf("reused ID missed: %+v", tail)
	}
}

func TestSearchFiltersReachSourcesAndURLs(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	cookie := signIn(t, d)
	for _, path := range []string{"/dashboard/decisions?search=invoice+parser&model=glm-5.3", "/dashboard/logs?search=invoice+parser&model=glm-5.3"} {
		body := page(t, d, cookie, path, http.StatusOK)
		contains(t, "search input", body, "Search prompt text", `value="invoice parser"`)
	}
	if source.decisionFilters[0].Search != "invoice parser" || source.lastRequestFilter().Search != "invoice parser" {
		t.Fatal("search was dropped from page filters")
	}
	sse := open(t, d, cookie, "/dashboard/sse/decisions?page=2", `{"search":"invoice parser","model":"glm-5.3"}`)
	sse.until(`replaceState({}, "", "/dashboard/decisions?model=glm-5.3&search=invoice+parser")`)
	sse.ends()
	logs := open(t, d, cookie, "/dashboard/sse/logs", `{"search":"invoice parser","status":"errors","model":"glm-5.3"}`)
	logs.until(`replaceState({}, "", "/dashboard/logs?model=glm-5.3&search=invoice+parser&status=errors")`)
	if source.lastRequestFilter().Search != "invoice parser" {
		t.Fatal("live search filter was dropped")
	}
}
