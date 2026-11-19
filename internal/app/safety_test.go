package app

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
)

// notServed checks that a request got the router's own 404, in the
// Anthropic shape.
func notServed(t *testing.T, h *harness, method, path, body string, header ...string) {
	t.Helper()
	response := h.do(method, path, body, header...)
	kind, message := anthropicError(t, response)
	if response.StatusCode != http.StatusNotFound || kind != "not_found_error" ||
		message != "ai-router serves only Claude Code's Messages API" {
		t.Errorf("%s %s = %d %s: %s", method, path, response.StatusCode, kind, message)
	}
}

func TestOnlyTheMessagesAPIReachesCLIProxy(t *testing.T) {
	h := newHarness(t, "")
	notServed(t, h, http.MethodGet, "/v0/management/auth-files", "", "X-Management-Key", "guessed")
	readAll(t, h.do(http.MethodPost, "/v1/messages", `{"model":"gpt-6.1-sol","max_tokens":1024,"messages":[]}`,
		"X-Management-Key", "guessed", "X-Forwarded-For", "10.0.0.9"))
	routed := h.cpa.routed()
	if len(routed) != 1 {
		t.Fatalf("CLIProxy saw %d requests, want the message", len(routed))
	}
	if r := routed[0]; r.Path != "/v1/messages" || r.Header.Get("X-Management-Key") != "" || r.Header.Get("X-Forwarded-For") != "" {
		t.Fatalf("CLIProxy saw %s with management key %q, forwarded for %q", r.Path,
			r.Header.Get("X-Management-Key"), r.Header.Get("X-Forwarded-For"))
	}
}

func TestOtherHarnessesAPIsAreNotServed(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	chat := `{"model":"auto","messages":[{"role":"user","content":"design the storage layer"}]}`
	for _, c := range []struct {
		method, path, body string
		header             []string
	}{
		{http.MethodPost, "/v1/chat/completions", chat, []string{"User-Agent", "opencode/1.18.33", "X-Session-Id", "ses_chat"}},
		{http.MethodPost, "/v1/chat/completions", strings.Replace(chat, `"auto"`, `"gpt-6.1-sol"`, 1), nil},
		{http.MethodPost, "/v1/responses", `{"model":"auto","input":"design the storage layer"}`,
			[]string{"Originator", "codex_cli_rs", "Session-Id", "codex-1"}},
		{http.MethodPost, "/v1/responses/compact", `{"model":"auto","input":[]}`, []string{"Originator", "codex_cli_rs"}},
		{http.MethodGet, "/v1/responses", "", []string{"Connection", "Upgrade", "Upgrade", "websocket"}},
		{http.MethodPost, "/v1beta/models/gemini-3-pro:generateContent", `{"contents":[]}`, nil},
		{http.MethodGet, "/v1/messages", "", nil},
	} {
		notServed(t, h, c.method, c.path, c.body, c.header...)
	}
	if routed := h.cpa.routed(); len(routed) != 0 || h.jev.calls.Load() != 0 {
		t.Fatalf("CLIProxy saw %d requests and the decider %d", len(routed), h.jev.calls.Load())
	}
	if logged := h.logged(); len(logged) != 0 {
		t.Fatalf("logged %+v", logged)
	}
}

func TestDotSegmentsNeverReachCLIProxy(t *testing.T) {
	h := newHarness(t, "")
	for _, path := range []string{
		"/v1/messages/../../v0/management/auth-files",
		"/v1/%2e%2e/v0/management/x",
		"/v1/messages/%2E%2E/%2e%2e/v0/management/config",
		"/v1/messages/..%2f..%2fv0/management/auth-files",
		"/v1/messages/count_tokens/../../../v0/management/api-keys",
		"/v1/models/../../v0/management/auth-files",
		"/v1/./messages",
	} {
		for _, method := range []string{http.MethodGet, http.MethodPost} {
			notServed(t, h, method, path, `{"model":"gpt-6.1-sol","max_tokens":1024,"messages":[]}`, "X-Management-Key", "guessed")
		}
	}
	if routed := h.cpa.routed(); len(routed) != 0 {
		t.Fatalf("CLIProxy saw %s", routed[0].Path)
	}
}

func TestAnAutoEffortSuffixSetsTheEffort(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	response := h.do(http.MethodPost, "/v1/messages", `{"model":"auto(xhigh)","max_tokens":1024,"messages":[{"role":"user","content":"hard"}],`+tools+`}`,
		"User-Agent", claudeCode, "X-Claude-Code-Session-Id", "ses_suffix")
	if body := readAll(t, response); !strings.Contains(body, `"gpt-6.1-sol(xhigh)"`) || response.Header.Get("X-Ts-Router-Effort") != "xhigh" {
		t.Fatalf("auto(xhigh) = %s, effort %q", body, response.Header.Get("X-Ts-Router-Effort"))
	}
	readAll(t, h.do(http.MethodPost, "/v1/messages", `{"model":"auto(very high)","max_tokens":1024,"messages":[]}`))
	if routed := h.cpa.routed(); routed[len(routed)-1].Model != "auto(very high)" {
		t.Fatalf("an invalid effort was routed as auto: %q", routed[len(routed)-1].Model)
	}
	logged := h.logged()
	if len(logged) != 1 || logged[0].Requested != "auto(xhigh)" || logged[0].Effort != "xhigh" {
		t.Fatalf("logged = %+v", logged)
	}
}

func TestASilentUpstreamIsCutOff(t *testing.T) {
	h := newHarness(t, "")
	h.yaml = strings.Replace(h.yaml, "upstream: {base_url:", "upstream: {idle_timeout: 1s, base_url:", 1)
	h.restart()
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if strings.Contains(string(r.Body), "headers") {
			time.Sleep(1500 * time.Millisecond)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"1\"}\n\n")
		w.(http.Flusher).Flush()
		time.Sleep(1500 * time.Millisecond)
	}
	started := time.Now()
	silent := h.claude("ses_silent_headers", "headers")
	if body := readAll(t, silent); silent.StatusCode != http.StatusGatewayTimeout || !strings.Contains(body, "sent nothing for 1s") {
		t.Fatalf("silent upstream = %d %s", silent.StatusCode, body)
	}
	stalled := h.claude("ses_silent_stream", "stream")
	if body := readAll(t, stalled); stalled.StatusCode != http.StatusOK || !strings.Contains(body, `"id":"1"`) {
		t.Fatalf("stalled stream = %d %s", stalled.StatusCode, body)
	}
	if elapsed := time.Since(started); elapsed > 2500*time.Millisecond {
		t.Fatalf("silent upstreams held their requests for %v", elapsed)
	}
	for _, entry := range h.logged() {
		if entry.Error != "CLIProxy sent nothing for 1s" {
			t.Fatalf("logged error = %q", entry.Error)
		}
	}
}

func TestAnImageThreadRecoversFromAFailover(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		switch r.Model {
		case "gpt-6.1-sol":
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"usage limit reached"}}`))
		case "glm-5.3":
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"model does not support images"}}`))
		default:
			_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
		}
	}
	for turn := range 3 {
		response := h.claudeImage("ses_image_failover")
		if body := readAll(t, response); response.StatusCode != http.StatusOK || !strings.Contains(body, `"deepseek-flash"`) {
			t.Fatalf("turn %d = %d %s", turn, response.StatusCode, body)
		}
	}
}

func TestCountingTokensNeverWaitsForAGenerationSlot(t *testing.T) {
	h := newHarness(t, "")
	h.yaml += glmLimits(1)
	h.restart()
	hold := make(chan struct{})
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Path == "/v1/messages" {
			<-hold
		}
		_, _ = fmt.Fprintf(w, `{"model":%q,"input_tokens":12}`, r.Model)
	}
	claude := []string{"User-Agent", claudeCode, "X-Claude-Code-Session-Id", "cc-count"}
	first := make(chan struct{})
	go func() {
		readAll(t, h.do(http.MethodPost, "/v1/messages", `{"model":"auto","messages":[{"role":"user","content":"add a flag"}],`+tools+`}`, claude...))
		close(first)
	}()
	for h.app.limits.Gate("glm-5.3").Stats().InFlight != 1 {
		time.Sleep(time.Millisecond)
	}
	started := time.Now()
	count := h.do(http.MethodPost, "/v1/messages/count_tokens", `{"model":"auto","messages":[{"role":"user","content":"x"}]}`, claude...)
	if body := readAll(t, count); count.StatusCode != http.StatusOK || time.Since(started) > 500*time.Millisecond {
		t.Fatalf("count_tokens = %d %s after %v", count.StatusCode, body, time.Since(started))
	}
	close(hold)
	<-first
	logged := h.logged()
	if len(logged) != 2 || logged[1].Kind != "count_tokens" || logged[1].Key == "" || logged[1].Key != logged[0].Key {
		t.Fatalf("a thread's side request is not charged to it: %+v", logged)
	}
}

func TestEveryEffortOfAModelSharesItsGate(t *testing.T) {
	h := newHarness(t, "")
	h.yaml += glmLimits(1)
	h.restart()
	hold := make(chan struct{})
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		<-hold
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}
	done := make(chan struct{})
	go func() {
		readAll(t, h.do(http.MethodPost, "/v1/messages", `{"model":"glm-5.3(high)","max_tokens":1024,"messages":[]}`))
		close(done)
	}()
	deadline := time.Now().Add(2 * time.Second)
	for h.app.limits.Gate("glm-5.3").Stats().InFlight != 1 {
		if time.Now().After(deadline) {
			close(hold)
			t.Fatal("an explicit glm-5.3(high) request passed the glm-5.3 gate")
		}
		time.Sleep(time.Millisecond)
	}
	close(hold)
	<-done
}

func TestHealthHidesRefreshErrorsAndReportsDraining(t *testing.T) {
	h := newHarness(t, "")
	h.cpa.server.Close()
	_ = h.app.catalog.Refresh(t.Context())
	response, err := http.Get(h.server.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	var status struct {
		Loaded   bool   `json:"loaded"`
		Error    string `json:"error"`
		Draining bool   `json:"draining"`
	}
	if err := json.NewDecoder(response.Body).Decode(&status); err != nil || response.StatusCode != http.StatusOK ||
		strings.Contains(status.Error, "127.0.0.1") || status.Error == "" {
		t.Fatalf("health = %d %+v, %v", response.StatusCode, status, err)
	}
	response.Body.Close()
	h.app.draining.Store(true)
	response, err = http.Get(h.server.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	_ = json.NewDecoder(response.Body).Decode(&status)
	response.Body.Close()
	if response.StatusCode != http.StatusServiceUnavailable || !status.Draining {
		t.Fatalf("draining health = %d %+v", response.StatusCode, status)
	}
}

func TestPinsLeaveOnlyARetiredModel(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "deepseek-flash"
	readAll(t, h.claude("ses_retired", "rename x"))
	// A model that is merely no longer a candidate keeps its threads.
	h.yaml = strings.NewReplacer(
		`    - {id: deepseek-flash, context: 1000000, output: 64000, vision: true, description: "quick questions and small edits"}`+"\n", "",
		"{gpt-6.1-sol: glm-5.3, glm-5.3: deepseek-flash, deepseek-flash: glm-5.3}", "{gpt-6.1-sol: glm-5.3}",
	).Replace(h.yaml)
	h.restart()
	if body := readAll(t, h.claude("ses_retired", "continue")); !strings.Contains(body, `"deepseek-flash"`) {
		t.Fatalf("a pin on a model that is no longer a candidate was moved: %s", body)
	}
	// Retiring it moves them: side requests at once, the pin at the
	// thread's next turn.
	h.yaml = strings.Replace(h.yaml, "{gpt-6.1-sol: glm-5.3}", "{gpt-6.1-sol: glm-5.3}\n  retired: [deepseek-flash]", 1)
	h.restart()
	count := h.do(http.MethodPost, "/v1/messages/count_tokens", `{"model":"auto","messages":[{"role":"user","content":"x"}]}`,
		"User-Agent", claudeCode, "X-Claude-Code-Session-Id", "ses_retired")
	if body := readAll(t, count); !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("count_tokens went to a retired model: %s", body)
	}
	if body := readAll(t, h.claude("ses_retired", "continue")); !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("a pin on a retired model was not moved: %s", body)
	}
	page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || len(page.Threads) != 1 || page.Threads[0].Model != "glm-5.3" || page.Threads[0].FailedOver {
		t.Fatalf("threads = %+v, %v", page.Threads, err)
	}
	key := page.Threads[0].Key
	if _, err := h.app.resolver.Override(t.Context(), key, "gpt-6.1-sol", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := h.app.resolver.Override(t.Context(), key, "deepseek-flash", ""); err == nil {
		t.Fatal("an operator re-pinned a thread to a model that is not a candidate")
	}
	if body := readAll(t, h.claude("ses_retired", "and again")); !strings.Contains(body, `"gpt-6.1-sol"`) {
		t.Fatalf("the override did not hold: %s", body)
	}
	detail, err := h.app.store.Thread(t.Context(), key)
	if err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, e := range detail.Events {
		kinds = append(kinds, e.Kind+":"+e.Previous+"->"+e.Model)
	}
	if got := strings.Join(kinds, ","); got != "decision:->deepseek-flash,migration:deepseek-flash->glm-5.3,override:glm-5.3->gpt-6.1-sol" {
		t.Fatalf("history = %s", got)
	}
}

func TestBackgroundCallsWithoutToolsNeverDecide(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	claude := []string{"User-Agent", claudeCode, "X-Claude-Code-Session-Id", "cc-title"}
	title := `{"model":"auto","max_tokens":64,"messages":[{"role":"user","content":"Write a 5-word title for: design the storage layer"}]}`
	// Without the hint headers a title request carries no class: it must
	// not decide the thread, and goes to the aux model.
	if body := readAll(t, h.do(http.MethodPost, "/v1/messages", title, claude...)); !strings.Contains(body, `"glm-5.3-flash"`) {
		t.Fatalf("a title request before the thread's first turn = %s", body)
	}
	if calls := h.jev.calls.Load(); calls != 0 {
		t.Fatalf("a background call made the thread's decision (%d calls)", calls)
	}
	readAll(t, h.claude("cc-title", "design the storage layer"))
	// Once the thread is pinned, its background calls follow the pin.
	if body := readAll(t, h.do(http.MethodPost, "/v1/messages", title, claude...)); !strings.Contains(body, `"gpt-6.1-sol"`) {
		t.Fatalf("a background call of a pinned thread = %s", body)
	}
	if calls := h.jev.calls.Load(); calls != 1 {
		t.Fatalf("decider called %d times", calls)
	}
}

func TestTheFirstPromptIsKeptWhole(t *testing.T) {
	h := newHarness(t, "  state_max_chars: 200")
	long := "Refactor the billing module. " + strings.Repeat("Keep every invoice total exact. ", 40) + "Then add tests."
	prompt := "<system-reminder>CLAUDE.md says hi</system-reminder>" + long
	readAll(t, h.claude("cc-first", prompt))
	page, err := h.app.store.Decisions(t.Context(), pin.DecisionFilter{})
	if err != nil || len(page.Decisions) != 1 {
		t.Fatalf("decisions = %+v, %v", page, err)
	}
	decision := page.Decisions[0]
	if decision.FirstPrompt != long || !strings.Contains(decision.Prompt, "[…]") || decision.Basis.PromptChars != len(long) {
		t.Fatalf("first prompt %q, judged %q", decision.FirstPrompt, decision.Prompt)
	}

	quiet := newHarness(t, "  record_prompts: false")
	readAll(t, quiet.claude("cc-quiet", "rename a variable"))
	page, err = quiet.app.store.Decisions(t.Context(), pin.DecisionFilter{})
	if err != nil || page.Decisions[0].FirstPrompt != "" || page.Decisions[0].Prompt != "" {
		t.Fatalf("prompts kept without record_prompts: %+v, %v", page.Decisions, err)
	}
}
