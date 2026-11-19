package app

import (
	"database/sql"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/route"
	"github.com/MarioGK/AI-Router/internal/thread"
	"github.com/MarioGK/AI-Router/internal/wire"
)

// turn sends a Claude Code request of a session, from a subagent when agent
// is set and with a request class when class is, and returns the model it
// was routed to.
func (h *policyHarness) turn(session, agent, class, body string) string {
	h.t.Helper()
	r := httptest.NewRequestWithContext(h.t.Context(), http.MethodPost, "http://router.test/v1/messages", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer "+proxyKey)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("User-Agent", claudeCode)
	r.Header.Set("X-Claude-Code-Session-Id", session)
	if agent != "" {
		r.Header.Set("X-Claude-Code-Agent-Id", agent)
	}
	if class != "" {
		r.Header.Set("X-Claude-Code-Request-Class", class)
	}
	w := httptest.NewRecorder()
	h.app.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		h.t.Fatalf("request = %d: %s", w.Code, w.Body.String())
	}
	return w.Header().Get("X-Ts-Router-Model")
}

// prompt is a request whose newest message is a new prompt from the user.
func prompt(text string) string {
	return fmt.Sprintf(`{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":%q}],%s}`, text, tools)
}

// toolResult is the next turn of a tool loop: the newest user message
// carries a tool result, beside text the user sent along with it.
func toolResult(text string) string {
	return fmt.Sprintf(`{"model":"auto","max_tokens":1024,"messages":[`+
		`{"role":"user","content":%q},`+
		`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]},`+
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"README.md"},`+
		`{"type":"text","text":"Also check the storage layer."}]}],%s}`, text, tools)
}

// key is the pin key of a session, or of one of its subagents.
func (h *policyHarness) key(session, agent string) string {
	h.t.Helper()
	target := h.app.resolver.Resolve(h.t.Context(), wire.CountTokens,
		thread.Identity{Session: session, Agent: agent}, wire.Conversation{HasTools: true})
	if target.Thread == "" {
		h.t.Fatalf("no pinned thread for session %q agent %q", session, agent)
	}
	return target.Thread
}

// idle makes a thread's last request look older than it was, as if the
// thread had been idle for d since.
func (h *policyHarness) idle(key string, d time.Duration) {
	h.t.Helper()
	db, err := sql.Open("sqlite", "file:"+h.app.cfg.Database+"?_pragma=busy_timeout(5000)")
	if err != nil {
		h.t.Fatal(err)
	}
	defer db.Close()
	result, err := db.ExecContext(h.t.Context(), `UPDATE pins SET last_request_at = last_request_at - ? WHERE key = ?`,
		d.Milliseconds(), key)
	if err != nil {
		h.t.Fatal(err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		h.t.Fatalf("aged %d pins, want 1", n)
	}
}

// history is a thread's events of the given kind.
func (h *policyHarness) history(key, kind string) []pin.Event {
	h.t.Helper()
	detail, err := h.app.store.Thread(h.t.Context(), key)
	if err != nil || detail == nil {
		h.t.Fatalf("thread %s = %+v, %v", key, detail, err)
	}
	return slices.DeleteFunc(detail.Events, func(e pin.Event) bool { return e.Kind != kind })
}

func (h *policyHarness) checkPin(key, model, level string) *pin.Pin {
	h.t.Helper()
	p, err := h.app.store.Get(h.t.Context(), key)
	if err != nil || p == nil || p.Model != model || p.Level != level {
		h.t.Fatalf("pin = %+v, %v; want %s at %s", p, err, model, level)
	}
	return p
}

func (h *policyHarness) checkDecisions(want int) {
	h.t.Helper()
	if got := len(h.transport.classified); got != want {
		h.t.Fatalf("classifier called %d times, want %d", got, want)
	}
}

func TestSubagentsGetTheirOwnDecisions(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "deepseek-flash", level: "low"})
	const session = "orchestration"
	if got := h.turn(session, "", "", prompt("Spin up agents to audit the storage layer.")); got != "deepseek-flash" {
		t.Fatalf("session routed to %q", got)
	}
	h.transport.verdict = policyVerdict{model: "claude-opus-5-5", level: "xhigh"}
	if got := h.turn(session, "agent-hard", "", prompt("Find why writes across regions corrupt tenant data.")); got != "claude-opus-5-5" {
		t.Fatalf("hard subagent routed to %q", got)
	}
	h.transport.verdict = policyVerdict{model: "glm-5.3", level: "medium"}
	if got := h.turn(session, "agent-routine", "", prompt("Add pagination and endpoint tests; no rush.")); got != "glm-5.3" {
		t.Fatalf("routine subagent routed to %q", got)
	}
	h.checkDecisions(3)
	// Each thread keeps its own pin afterwards.
	for _, tc := range []struct{ agent, model string }{
		{"", "deepseek-flash"}, {"agent-hard", "claude-opus-5-5"}, {"agent-routine", "glm-5.3"},
	} {
		if got := h.turn(session, tc.agent, "", toolResult("Continue.")); got != tc.model {
			t.Errorf("agent %q continued on %q, want %q", tc.agent, got, tc.model)
		}
	}
	h.checkDecisions(3)

	parent, hard, routine := h.key(session, ""), h.key(session, "agent-hard"), h.key(session, "agent-routine")
	if parent == hard || parent == routine || hard == routine {
		t.Fatalf("keys collide: %s %s %s", parent, hard, routine)
	}
	if p := h.checkPin(hard, "claude-opus-5-5", "xhigh"); p.Parent != parent {
		t.Errorf("subagent pin parent = %q, want %q", p.Parent, parent)
	}
	if p := h.checkPin(parent, "deepseek-flash", "low"); p.Parent != "" {
		t.Errorf("session pin parent = %q", p.Parent)
	}
	decisions := h.history(hard, pin.Decision)
	if len(decisions) != 1 || decisions[0].Basis.Parent != parent {
		t.Fatalf("subagent decision = %+v", decisions)
	}
	detail, err := h.app.store.Thread(t.Context(), parent)
	if err != nil || detail == nil || !slices.Equal(detail.Agents, []string{hard, routine}) {
		t.Fatalf("session's subagents = %+v, %v", detail, err)
	}
}

func TestSubagentSideRequestsFollowItsPinThenTheSession(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "deepseek-flash", level: "low"})
	const session = "side-orchestration"
	h.turn(session, "", "", prompt("Plan the migration."))
	parent := h.key(session, "")
	side := func(kind wire.Kind, class string) route.Target {
		t.Helper()
		return h.app.resolver.Resolve(t.Context(), kind,
			thread.Identity{Session: session, Agent: "agent-1", Class: class}, wire.Conversation{HasTools: true})
	}
	// Before the subagent has a pin its side requests follow the session's,
	// and create none.
	for _, target := range []route.Target{side(wire.CountTokens, ""), side(wire.Messages, thread.Compaction)} {
		if target.Model != "deepseek-flash" || target.Thread != parent || target.Pinned() || target.Reason != route.ReasonSide {
			t.Errorf("side request before the subagent's pin = %+v", target)
		}
	}
	page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || page.Total != 1 {
		t.Fatalf("side requests created pins: %+v, %v", page, err)
	}
	if p := h.checkPin(parent, "deepseek-flash", "low"); p.Compacted {
		t.Error("a subagent's compaction marked its session's cache cold")
	}
	h.transport.verdict = policyVerdict{model: "gpt-6.1-sol", level: "high"}
	h.turn(session, "agent-1", "", prompt("Fix the cursor race before today's release."))
	agent := h.key(session, "agent-1")
	for _, target := range []route.Target{side(wire.CountTokens, ""), side(wire.Messages, thread.Compaction)} {
		if target.Model != "gpt-6.1-sol" || target.Level != "high" || target.Thread != agent || target.Pinned() {
			t.Errorf("side request after the subagent's pin = %+v", target)
		}
	}
	if got := h.turn(session, "agent-1", thread.Compaction, prompt("Summarize the conversation.")); got != "gpt-6.1-sol" {
		t.Errorf("subagent compaction routed to %q", got)
	}
	// Auxiliary calls are unchanged: they go to aux_model and decide nothing.
	aux := h.app.resolver.Resolve(t.Context(), wire.Messages,
		thread.Identity{Session: session, Agent: "agent-2", Class: thread.Auxiliary}, wire.Conversation{HasTools: true})
	if aux.Model != h.app.cfg.Auto.AuxModel || aux.Reason != route.ReasonAuxiliary || aux.Thread != "" {
		t.Errorf("auxiliary subagent request = %+v", aux)
	}
	h.checkDecisions(2)
}

func TestRedecisionAfterIdleMovesUpAndDownTheLadder(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "deepseek-flash", level: "low"})
	const session = "idle-thread"
	h.turn(session, "", "", prompt("Explain HTTP 204 in one sentence."))
	key := h.key(session, "")

	h.idle(key, 10*time.Minute)
	h.transport.verdict = policyVerdict{model: "claude-opus-5-5", level: "xhigh"}
	if got := h.turn(session, "", "", prompt("Now find why writes across regions corrupt tenant data.")); got != "claude-opus-5-5" {
		t.Fatalf("idle new prompt routed to %q", got)
	}
	h.checkDecisions(2)
	if state := h.transport.classified[1].State; !strings.HasSuffix(state, "Now find why writes across regions corrupt tenant data.") {
		t.Errorf("redecision judged %q, not the new prompt", state)
	}
	p := h.checkPin(key, "claude-opus-5-5", "xhigh")
	if p.Decided != "deepseek-flash" || p.Turns != 1 {
		t.Errorf("redecided pin = %+v", p)
	}

	h.idle(key, 30*time.Minute)
	h.transport.verdict = policyVerdict{model: "glm-5.3-flash", level: "medium"}
	if got := h.turn(session, "", "", prompt("Fix the form's disabled save button.")); got != "glm-5.3-flash" {
		t.Fatalf("second idle new prompt routed to %q", got)
	}
	h.checkPin(key, "glm-5.3-flash", "medium")
	h.checkDecisions(3)

	redecisions := h.history(key, pin.Redecision)
	if len(redecisions) != 2 {
		t.Fatalf("redecisions = %+v", redecisions)
	}
	up, down := redecisions[0], redecisions[1]
	if up.Previous != "deepseek-flash" || up.Model != "claude-opus-5-5" || up.Level != "xhigh" || up.Decider != "jev" ||
		len(up.Attempts) == 0 || up.Prompt == "" || !strings.HasPrefix(up.Detail, "idle for 10m") ||
		!strings.Contains(up.Detail, "deepseek-flash (low) → claude-opus-5-5 (xhigh)") {
		t.Errorf("upward redecision = %+v", up)
	}
	if down.Previous != "claude-opus-5-5" || down.Model != "glm-5.3-flash" ||
		!strings.Contains(down.Detail, "claude-opus-5-5 (xhigh) → glm-5.3-flash (medium)") {
		t.Errorf("downward redecision = %+v", down)
	}
	entries := h.logged()
	if len(entries) != 3 || entries[1].Reason != route.ReasonRedecided || entries[2].Reason != route.ReasonRedecided {
		t.Fatalf("logged requests = %+v", entries)
	}
	// A redecided thread no longer teaches the cost of its first decision.
	usages, err := h.app.store.TaskUsages(t.Context(), time.Unix(0, 0), time.Now().Add(time.Hour), 1<<40)
	if err != nil || len(usages) != 0 {
		t.Fatalf("task usages = %+v, %v", usages, err)
	}
}

func TestRedecisionNeverWhileTheCacheIsWarm(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "deepseek-flash", level: "low"})
	const session = "warm-thread"
	h.turn(session, "", "", prompt("Explain HTTP 204 in one sentence."))
	key := h.key(session, "")
	h.idle(key, 4*time.Minute)
	h.transport.verdict = policyVerdict{model: "claude-opus-5-5", level: "xhigh"}
	for _, text := range []string{"Now redesign the whole storage layer.", "And the replication too."} {
		if got := h.turn(session, "", "", prompt(text)); got != "deepseek-flash" {
			t.Fatalf("warm new prompt routed to %q", got)
		}
	}
	h.checkDecisions(1)
	if events := h.history(key, pin.Redecision); len(events) != 0 {
		t.Fatalf("warm cache redecided: %+v", events)
	}
}

func TestRedecisionNeverOnToolResults(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "deepseek-flash", level: "low"})
	const session = "tool-loop"
	h.turn(session, "", "", prompt("Explain HTTP 204 in one sentence."))
	key := h.key(session, "")
	h.transport.verdict = policyVerdict{model: "claude-opus-5-5", level: "xhigh"}

	// A tool loop resumed after a long tool call, or after a compaction,
	// stays where it is; the request warms the cache again.
	h.idle(key, time.Hour)
	if got := h.turn(session, "", "", toolResult("Explain HTTP 204 in one sentence.")); got != "deepseek-flash" {
		t.Fatalf("idle tool result routed to %q", got)
	}
	if got := h.turn(session, "", "", prompt("Next question, right away.")); got != "deepseek-flash" {
		t.Fatalf("prompt after a tool result routed to %q", got)
	}
	h.turn(session, "", thread.Compaction, prompt("Summarize the conversation."))
	if p := h.checkPin(key, "deepseek-flash", "low"); !p.Compacted {
		t.Fatal("compaction was not recorded")
	}
	if got := h.turn(session, "", "", toolResult("Summary of the conversation.")); got != "deepseek-flash" {
		t.Fatalf("tool result after a compaction routed to %q", got)
	}
	h.checkDecisions(1)

	// A new prompt right after a compaction finds the cache cold.
	h.turn(session, "", thread.Compaction, prompt("Summarize the conversation."))
	if got := h.turn(session, "", "", prompt("Now redesign the storage layer.")); got != "claude-opus-5-5" {
		t.Fatalf("new prompt after a compaction routed to %q", got)
	}
	h.checkDecisions(2)
	if events := h.history(key, pin.Redecision); len(events) != 1 || !strings.HasPrefix(events[0].Detail, "after a compaction") {
		t.Fatalf("redecisions = %+v", events)
	}
}

func TestOnlyACompletedCompactionMakesTheCacheCold(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "deepseek-flash", level: "low"})
	const session = "compacting-thread"
	h.turn(session, "", "", prompt("Explain HTTP 204 in one sentence."))
	key := h.key(session, "")
	h.transport.verdict = policyVerdict{model: "claude-opus-5-5", level: "xhigh"}
	compact := func() *httptest.ResponseRecorder {
		t.Helper()
		r := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "http://router.test/v1/messages",
			strings.NewReader(prompt("Summarize the conversation.")))
		r.Header.Set("Authorization", "Bearer "+proxyKey)
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("X-Claude-Code-Session-Id", session)
		r.Header.Set("X-Claude-Code-Request-Class", thread.Compaction)
		w := httptest.NewRecorder()
		h.app.ServeHTTP(w, r)
		return w
	}

	// A compaction the upstream refused, or whose stream was cut, left the
	// client with its old history and warm cache.
	h.transport.fail = http.StatusBadRequest
	if w := compact(); w.Code != http.StatusBadRequest {
		t.Fatalf("failed compaction = %d", w.Code)
	}
	h.transport.fail, h.transport.cut = 0, true
	if w := compact(); w.Code != http.StatusOK {
		t.Fatalf("cut compaction = %d", w.Code)
	}
	h.transport.cut = false
	if p := h.checkPin(key, "deepseek-flash", "low"); p.Compacted {
		t.Fatal("an unfinished compaction marked the cache cold")
	}
	if got := h.turn(session, "", "", prompt("Now redesign the storage layer.")); got != "deepseek-flash" {
		t.Fatalf("prompt after an unfinished compaction routed to %q", got)
	}
	h.checkDecisions(1)

	// A completed one did replace it.
	if w := compact(); w.Code != http.StatusOK {
		t.Fatalf("compaction = %d", w.Code)
	}
	if got := h.turn(session, "", "", prompt("Now redesign the storage layer.")); got != "claude-opus-5-5" {
		t.Fatalf("prompt after a completed compaction routed to %q", got)
	}
	h.checkDecisions(2)
}

func TestRedecisionOfAThreadThatOutgrewItsStep(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "deepseek-flash", level: "low"}, func(cfg *config.Config) {
		idle, grow := time.Duration(0), 3
		cfg.Auto.Redecide = config.Redecide{Idle: &idle, GrowRequests: &grow}
	})
	const session = "growing-thread"
	h.turn(session, "", "", prompt("Explain HTTP 204 in one sentence."))
	key := h.key(session, "")
	h.transport.verdict = policyVerdict{model: "gpt-6.1-sol", level: "high"}

	// With idle redecision off, neither a long idle time nor a compaction
	// decides a thread again before it outgrows its step.
	h.idle(key, time.Hour)
	h.turn(session, "", "", prompt("Another small question."))
	h.turn(session, "", thread.Compaction, prompt("Summarize the conversation."))
	if got := h.turn(session, "", "", prompt("And another one.")); got != "deepseek-flash" {
		t.Fatalf("thread below the threshold routed to %q", got)
	}
	h.checkDecisions(1)

	// Three requests in, the next compaction is a cold point it re-decides at.
	h.turn(session, "", thread.Compaction, prompt("Summarize the conversation."))
	if got := h.turn(session, "", "", prompt("Now take on the release blocker.")); got != "gpt-6.1-sol" {
		t.Fatalf("grown thread routed to %q", got)
	}
	h.checkDecisions(2)
	events := h.history(key, pin.Redecision)
	if len(events) != 1 || !strings.Contains(events[0].Detail, "3 requests below the top of the ladder") {
		t.Fatalf("redecisions = %+v", events)
	}

	// A thread on the top step has nothing to outgrow.
	h.transport.verdict = policyVerdict{model: "claude-opus-5-5", level: "xhigh"}
	const top = "top-thread"
	h.turn(top, "", "", prompt("Find why writes across regions corrupt tenant data."))
	for range 4 {
		h.turn(top, "", "", toolResult("Continue."))
	}
	h.transport.verdict = policyVerdict{model: "deepseek-flash", level: "low"}
	h.turn(top, "", thread.Compaction, prompt("Summarize the conversation."))
	if got := h.turn(top, "", "", prompt("Explain HTTP 204.")); got != "claude-opus-5-5" {
		t.Fatalf("top thread routed to %q", got)
	}
	h.checkDecisions(3)
}

func TestRedecisionNeverInFixedMode(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "deepseek-flash", level: "low"})
	const session = "fixed-thread"
	h.turn(session, "", "", prompt("Explain HTTP 204 in one sentence."))
	key := h.key(session, "")
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeFixed, FixedModel: "gpt-6.1-sol", JevEnabled: true})
	h.idle(key, time.Hour)
	h.transport.verdict = policyVerdict{model: "claude-opus-5-5", level: "xhigh"}
	if got := h.turn(session, "", "", prompt("Now redesign the storage layer.")); got != "gpt-6.1-sol" {
		t.Fatalf("fixed mode routed to %q", got)
	}
	h.checkDecisions(1)
	h.checkPin(key, "deepseek-flash", "low")
	if events := h.history(key, pin.Redecision); len(events) != 0 {
		t.Fatalf("fixed mode redecided: %+v", events)
	}

	// A compaction in fixed mode still replaced the automatic pin's history.
	h.turn(session, "", thread.Compaction, prompt("Summarize the conversation."))
	if p := h.checkPin(key, "deepseek-flash", "low"); !p.Compacted {
		t.Fatal("a fixed-mode compaction left the automatic pin's cache warm")
	}
}

func TestRedecisionKeepsAnOperatorsPin(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "deepseek-flash", level: "low"})
	const session = "overridden-thread"
	h.turn(session, "", "", prompt("Explain HTTP 204 in one sentence."))
	key := h.key(session, "")
	if _, err := h.app.resolver.Override(t.Context(), key, "glm-5.3", "medium"); err != nil {
		t.Fatal(err)
	}
	h.idle(key, time.Hour)
	h.transport.verdict = policyVerdict{model: "claude-opus-5-5", level: "xhigh"}
	if got := h.turn(session, "", "", prompt("Now redesign the storage layer.")); got != "glm-5.3" {
		t.Fatalf("overridden thread routed to %q", got)
	}
	h.checkDecisions(1)
}
