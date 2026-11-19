package proxy

import (
	"net/http"
	"reflect"
	"strconv"
	"testing"
	"time"
)

func TestClaudeCooldownReset(t *testing.T) {
	now := time.Unix(1791216000, 0)
	for _, hint := range []string{"3600", now.Add(time.Hour).UTC().Format(http.TimeFormat)} {
		t.Run(hint, func(t *testing.T) {
			response := &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": {hint}}}
			ClaudeLimit(response, now)
			if response.Header.Get(claudeLimitStatus) != "rejected" ||
				response.Header.Get(claudeLimitReset) != strconv.FormatInt(now.Add(time.Hour).Unix(), 10) ||
				response.Header.Get("Retry-After") != hint {
				t.Fatalf("cooldown headers = %v", response.Header)
			}
		})
	}
}

func TestClaudeLimitPreservesProviderSignals(t *testing.T) {
	for _, status := range []string{"allowed", "allowed_warning", "rejected"} {
		t.Run(status, func(t *testing.T) {
			headers := http.Header{
				"Retry-After": {"30"}, claudeLimitStatus: {status}, claudeLimitReset: {"1791630000"},
				"Anthropic-Ratelimit-Unified-Overage-Status": {"allowed"},
			}
			response := &http.Response{StatusCode: 429, Header: headers.Clone()}
			ClaudeLimit(response, time.Now())
			if !reflect.DeepEqual(response.Header, headers) {
				t.Fatalf("changed provider headers: %v", response.Header)
			}
		})
	}
	response := &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": {"30"}, claudeLimitReset: {"1791630000"}}}
	ClaudeLimit(response, time.Now())
	if response.Header.Get(claudeLimitReset) != "1791630000" {
		t.Fatal("replaced the provider's reset timestamp")
	}
}

func TestClaudeLimitDoesNotInventAReset(t *testing.T) {
	now := time.Unix(1791216000, 0)
	for _, hint := range []string{"", "unknown", "-1", "0", "1.5", "9223372036854775807", now.Add(-time.Hour).UTC().Format(http.TimeFormat)} {
		response := &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": {hint}}}
		ClaudeLimit(response, now)
		if response.Header.Get(claudeLimitStatus) != "" || response.Header.Get(claudeLimitReset) != "" {
			t.Fatalf("invented a reset for %q: %v", hint, response.Header)
		}
	}
	for _, status := range []int{200, 403, 503, 529} {
		response := &http.Response{StatusCode: status, Header: http.Header{"Retry-After": {"30"}}}
		ClaudeLimit(response, now)
		if response.Header.Get(claudeLimitStatus) != "" {
			t.Fatalf("reported a usage limit for status %d", status)
		}
	}
}

func TestClaudeLimitUsesManagementFallback(t *testing.T) {
	now := time.Unix(1791216000, 0)
	reset := now.Add(90 * time.Second)
	response := &http.Response{StatusCode: http.StatusTooManyRequests, Header: make(http.Header)}
	ClaudeLimit(response, now, reset)
	if response.Header.Get(claudeLimitStatus) != "rejected" ||
		response.Header.Get(claudeLimitReset) != strconv.FormatInt(reset.Unix(), 10) ||
		response.Header.Get("Retry-After") != "90" {
		t.Fatalf("fallback cooldown headers = %v", response.Header)
	}
}
