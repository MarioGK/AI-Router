package app

import (
	"net/http"
	"testing"
	"time"
)

func TestHopsNoticeAConversationOnAnotherAccount(t *testing.T) {
	var h hops
	start := time.Date(2026, 9, 29, 14, 0, 0, 0, time.UTC)
	request := func(session, agent string) http.Header {
		header := http.Header{}
		header.Set("X-Claude-Code-Session-Id", session)
		if agent != "" {
			header.Set("X-Claude-Code-Agent-Id", agent)
		}
		return header
	}
	response := func(status int, account string) *http.Response {
		header := http.Header{}
		if account != "" {
			header.Set("X-CPA-TRACE-ID", "20260929140000-"+account+"-req-1")
		}
		return &http.Response{StatusCode: status, Header: header}
	}
	for i, step := range []struct {
		header   http.Header
		model    string
		response *http.Response
		after    time.Duration
		from, to string
	}{
		{request("s1", ""), "claude-opus-5-5", response(200, "aaaa"), 0, "", ""},
		{request("s1", ""), "claude-opus-5-5", response(200, "aaaa"), time.Minute, "", ""},
		{request("s1", ""), "claude-opus-5-5", response(429, "bbbb"), 2 * time.Minute, "", ""}, // a failure serves nothing
		{request("s1", ""), "claude-opus-5-5", response(200, "bbbb"), 3 * time.Minute, "aaaa", "bbbb"},
		{request("s1", "agent-1"), "claude-opus-5-5", response(200, "cccc"), 4 * time.Minute, "", ""},      // a subagent has its own
		{request("s1", ""), "claude-haiku-4-5", response(200, "cccc"), 5 * time.Minute, "", ""},            // and so has each model
		{request("", ""), "claude-opus-5-5", response(200, "dddd"), 6 * time.Minute, "", ""},               // no session, nothing to follow
		{request("s1", ""), "claude-opus-5-5", response(200, ""), 7 * time.Minute, "", ""},                 // no trace either
		{request("s1", ""), "claude-opus-5-5", response(200, "dddd"), 3*time.Minute + 2*time.Hour, "", ""}, // idle past the affinity
	} {
		from, to, moved := h.observe(step.header, step.model, step.response, start.Add(step.after))
		if from != step.from || to != step.to || moved != (step.from != "") {
			t.Errorf("step %d: observe = %q, %q, %v; want %q, %q", i, from, to, moved, step.from, step.to)
		}
	}
}

func TestTraceAccount(t *testing.T) {
	for trace, want := range map[string]string{
		"20260929140000-cb5f7df4d1c6a1d9-7f3a-11ef": "cb5f7df4d1c6a1d9",
		"20260929140000-cb5f7df4d1c6a1d9-req":       "cb5f7df4d1c6a1d9",
		"":                                          "",
		"garbage":                                   "",
	} {
		if got := traceAccount(trace); got != want {
			t.Errorf("traceAccount(%q) = %q, want %q", trace, got, want)
		}
	}
}
