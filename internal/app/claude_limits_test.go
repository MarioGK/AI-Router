package app

import (
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/accounts"
	"github.com/MarioGK/AI-Router/internal/pin"
)

func TestClaudeCooldownReachesTheClient(t *testing.T) {
	for _, model := range []string{"claude-opus-5-5", "claude-opus-5-5(high)", "auto"} {
		t.Run(model, func(t *testing.T) {
			h := newHarness(t, "")
			// Auto is restricted to Claude with no fallback, so the final 429
			// reaches the client rather than a different model's response.
			if model == "auto" {
				h.yaml = strings.Replace(h.yaml, "  fallback:",
					"    - {id: claude-opus-5-5, context: 1000000, output: 128000, vision: true, description: \"the hardest work\"}\n  fallback:", 1)
				h.restart()
				h.jev.answer = "claude-opus-5-5"
				if err := h.app.store.SaveRoutingSettings(t.Context(), pin.RoutingSettings{Mode: "auto", Models: []string{"claude-opus-5-5"}, JevEnabled: true}); err != nil {
					t.Fatal(err)
				}
			}
			body := `{"type":"error","error":{"type":"rate_limit_error","message":"All credentials for model claude-opus-5-5 are cooling down"}}`
			h.cpa.respond = func(w http.ResponseWriter, _ recorded) {
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", "3600")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(body))
			}
			before := time.Now().Unix()
			response := h.do(http.MethodPost, "/v1/messages", fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"shell"}]}`, model),
				"X-Claude-Code-Session-Id", "claude-limit")
			reset, _ := strconv.ParseInt(response.Header.Get("Anthropic-Ratelimit-Unified-Reset"), 10, 64)
			if response.StatusCode != 429 || response.Header.Get("Anthropic-Ratelimit-Unified-Status") != "rejected" ||
				reset < before+3600 || reset > time.Now().Unix()+3600 || response.Header.Get("Retry-After") != "3600" || readAll(t, response) != body {
				t.Fatalf("Claude cooldown: status %d, headers %v", response.StatusCode, response.Header)
			}
		})
	}
}

func TestClaudeCooldownUsesManagementResetWhenProxyOmitsRetryAfter(t *testing.T) {
	h := newHarness(t, "")
	observed := time.Now().Add(-time.Second).UTC().Format(time.RFC3339Nano)
	reset := time.Now().Add(2 * time.Hour).Unix()
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Path == "/v0/management/auth-files" {
			_, _ = fmt.Fprintf(w, `{"files":[{"name":"claude-a.json","provider":"claude","quota":{"observed_at":%q,"signals":{"Anthropic-Ratelimit-Unified-5h-Utilization":"1","Anthropic-Ratelimit-Unified-5h-Reset":%q,"Anthropic-Ratelimit-Unified-5h-Status":"rejected","Anthropic-Ratelimit-Unified-Status":"rejected"}}}]}`, observed, strconv.FormatInt(reset, 10))
			return
		}
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"all Claude credentials are cooling down"}}`))
	}
	h.app.accounts = accounts.New(h.cpa.server.URL, "management-key", []string{accounts.Claude}, -10, h.cpa.server.Client(), slog.New(slog.DiscardHandler))
	response := h.do(http.MethodPost, "/v1/messages", `{"model":"claude-opus-5-5","messages":[]}`)
	if response.StatusCode != http.StatusTooManyRequests || response.Header.Get("Anthropic-Ratelimit-Unified-Status") != "rejected" ||
		response.Header.Get("Anthropic-Ratelimit-Unified-Reset") != strconv.FormatInt(reset, 10) || response.Header.Get("Retry-After") == "" {
		t.Fatalf("Claude cooldown headers = %v", response.Header)
	}
}

func TestClaudeSignalsPassThroughOnSuccessAndErrors(t *testing.T) {
	for _, status := range []int{200, 429} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			h := newHarness(t, "")
			body := "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
			h.cpa.respond = func(w http.ResponseWriter, _ recorded) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Anthropic-Ratelimit-Unified-Status", "allowed_warning")
				w.Header().Set("Anthropic-Ratelimit-Unified-Reset", "1791630000")
				w.Header().Set("Anthropic-Ratelimit-Unified-Representative-Claim", "seven_day")
				w.Header().Set("Anthropic-Ratelimit-Unified-Overage-Status", "allowed")
				w.Header().Set("Retry-After", "30")
				w.WriteHeader(status)
				_, _ = w.Write([]byte(body))
			}
			response := h.do(http.MethodPost, "/v1/messages", `{"model":"claude-opus-5-5","messages":[]}`)
			if response.StatusCode != status || readAll(t, response) != body {
				t.Fatalf("response changed: %d", response.StatusCode)
			}
			for key, value := range map[string]string{
				"Status": "allowed_warning", "Reset": "1791630000", "Representative-Claim": "seven_day", "Overage-Status": "allowed",
			} {
				if response.Header.Get("Anthropic-Ratelimit-Unified-"+key) != value {
					t.Errorf("lost %s: %v", key, response.Header)
				}
			}
		})
	}
}

func TestNonClaudeCooldownDoesNotBecomeClaudeQuota(t *testing.T) {
	h := newHarness(t, "")
	h.cpa.respond = func(w http.ResponseWriter, _ recorded) {
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"error":"busy"}`))
	}
	response := h.do(http.MethodPost, "/v1/messages", `{"model":"gpt-6.1-sol","messages":[]}`)
	if response.StatusCode != 429 || !strings.Contains(readAll(t, response), "busy") || response.Header.Get("Anthropic-Ratelimit-Unified-Status") != "" {
		t.Fatalf("non-Claude cooldown: %d, %v", response.StatusCode, response.Header)
	}
}
