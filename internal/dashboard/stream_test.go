package dashboard

import (
	"bufio"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
)

// stream reads the Datastar events of one SSE response.
type stream struct {
	t        *testing.T
	response *http.Response
	events   chan string
}

// open starts a Datastar request as the signed-in operator; signals are
// sent the way Datastar sends them with a GET.
func open(t *testing.T, d *Dashboard, cookie *http.Cookie, target, signals string) *stream {
	t.Helper()
	server := httptest.NewServer(d)
	t.Cleanup(server.Close)
	if signals != "" {
		target += map[bool]string{true: "&", false: "?"}[strings.Contains(target, "?")] + "datastar=" + url.QueryEscape(signals)
	}
	request, err := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL+target, nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Datastar-Request", "true")
	request.AddCookie(cookie)
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { response.Body.Close() })
	s := &stream{t: t, response: response, events: make(chan string, 64)}
	go func() {
		defer close(s.events)
		lines := bufio.NewScanner(response.Body)
		lines.Buffer(nil, 1<<20)
		var event strings.Builder
		for lines.Scan() {
			if lines.Text() == "" {
				s.events <- event.String()
				event.Reset()
				continue
			}
			event.WriteString(lines.Text() + "\n")
		}
	}()
	return s
}

// until returns the first event that contains want.
func (s *stream) until(want string) string {
	s.t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event, ok := <-s.events:
			if !ok {
				s.t.Fatalf("the stream ended before an event with %q", want)
			}
			if strings.Contains(event, want) {
				return event
			}
		case <-deadline:
			s.t.Fatalf("no event with %q", want)
		}
	}
}

// ends waits for the server to close the stream.
func (s *stream) ends() {
	s.t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		select {
		case _, ok := <-s.events:
			if !ok {
				return
			}
		case <-deadline:
			s.t.Fatal("the stream did not end")
		}
	}
}

func TestFeedStreamsNewRequestsAndFigures(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	feed := open(t, d, signIn(t, d), "/dashboard/sse/feed", "")
	if got := feed.response.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("content type %q", got)
	}
	first := feed.until("event: datastar-patch-elements")
	if !strings.Contains(first, `id="feed-rows"`) || !strings.Contains(first, `id="request-4"`) || strings.Contains(first, "arrive") {
		t.Fatalf("first patch:\n%s", first)
	}

	source.log(pin.RequestLog{ID: 5, At: time.Now(), Kind: "messages", Requested: "auto", Model: "gpt-6.1-sol",
		Reason: "decided", Status: 200})
	arrived := feed.until(`id="request-5"`)
	if !regexp.MustCompile(`id="request-5" class="[^"]* arrive"`).MatchString(arrived) || !strings.Contains(arrived, `id="request-4"`) {
		t.Fatalf("the new request did not arrive on top of the others:\n%s", arrived)
	}
	figures := feed.until(`id="kpis"`)
	for _, section := range []string{`id="timeline"`, `id="catalog"`, `id="deciders"`} {
		if !strings.Contains(figures, section) {
			t.Errorf("the figures refresh lacks %s", section)
		}
	}

	d.Close()
	feed.ends()
}

func TestLogStreamFollowsItsFilters(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	cookie := signIn(t, d)
	log := open(t, d, cookie, "/dashboard/sse/logs", `{"model":"claude-opus-5-5","status":"errors","paused":false}`)
	first := log.until(`id="log-rows"`)
	if !strings.Contains(first, `id="request-3"`) || strings.Contains(first, `id="request-1"`) || strings.Contains(first, `id="request-4"`) {
		t.Fatalf("first patch ignores the filters:\n%s", first)
	}
	log.until(`replaceState({}, "", "/dashboard/logs?model=claude-opus-5-5&status=errors")`)

	source.log(pin.RequestLog{ID: 5, At: time.Now(), Model: "claude-opus-5-5", Status: 200})
	source.log(pin.RequestLog{ID: 6, At: time.Now(), Model: "claude-opus-5-5", Status: 502})
	source.log(pin.RequestLog{ID: 7, At: time.Now(), Model: "glm-5.3", Status: 500})
	next := log.until(`id="request-6"`)
	if strings.Contains(next, `id="request-5"`) || strings.Contains(next, `id="request-7"`) {
		t.Fatalf("the tail kept requests outside the filters:\n%s", next)
	}
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(time.Millisecond) {
		if got := source.lastRequestFilter(); got.AfterID == 6 && got.Model == "claude-opus-5-5" {
			break
		} else if time.Now().After(deadline) {
			t.Fatalf("the tail's cursor did not move to the last request it saw: %+v", got)
		}
	}

	paused := open(t, d, cookie, "/dashboard/sse/logs", `{"paused":true}`)
	if paused.response.StatusCode != http.StatusNoContent {
		t.Fatalf("a paused log = %d, want 204", paused.response.StatusCode)
	}
}

func TestThreadTableFollowsFiltersAndPages(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	table := open(t, d, signIn(t, d), "/dashboard/sse/threads?page=2", `{"model":"glm-5.3"}`)
	patch := table.until(`id="threads-table"`)
	if !strings.Contains(patch, shortKey(source.threads[1].Key)) || strings.Contains(patch, shortKey(source.threads[0].Key)) {
		t.Fatalf("thread patch:\n%s", patch)
	}
	table.until(`replaceState({}, "", "/dashboard/threads?model=glm-5.3")`)
	table.ends()
	source.mu.Lock()
	defer source.mu.Unlock()
	if got := source.threadFilters[0]; got.Model != "glm-5.3" || got.Offset != threadsPerPage {
		t.Fatalf("first read %+v, want the second page of glm-5.3 threads", got)
	}
}

func TestModelMixSwitchesToTheWeek(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	mix := open(t, d, signIn(t, d), "/dashboard/sse/mix?range=7d", "")
	patch := mix.until(`id="mix"`)
	if !strings.Contains(patch, `aria-pressed="true"`) || !strings.Contains(patch, "7 days</button>") {
		t.Fatalf("mix patch:\n%s", patch)
	}
	mix.ends()
	source.mu.Lock()
	defer source.mu.Unlock()
	if got := source.overviewCalls; len(got) != 1 || got[0] != 7*24*time.Hour {
		t.Fatalf("overview buckets = %v, want one week-long bucket", got)
	}
}
