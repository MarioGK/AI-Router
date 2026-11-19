package app

import (
	"net/http"
	"strings"
	"sync"
	"time"
)

// hopMemory is how long a conversation's account is remembered: CLIProxy's
// session affinity forgets an idle session after an hour.
const hopMemory = time.Hour

// hopPrune is how many remembered conversations make an observation prune
// the ones idle for longer than hopMemory.
const hopPrune = 4096

// hops notices a Claude Code conversation that CLIProxy served from another
// subscription account than its previous request. CLIProxy binds each
// session to one account; a session that moves loses its prompt cache, and
// Anthropic sees one Claude Code session on two accounts.
type hops struct {
	mu   sync.Mutex
	last map[string]served
}

// served is the account that last served a conversation, and when.
type served struct {
	account string
	at      time.Time
}

// observe records the account that served a successful response of model
// and returns the account the conversation's previous request was served
// from when that was another one. Requests without a Claude Code session,
// and responses without CLIProxy's trace, are not followed.
func (h *hops) observe(header http.Header, model string, response *http.Response, now time.Time) (from, to string, moved bool) {
	session := header.Get("X-Claude-Code-Session-Id")
	to = traceAccount(response.Header.Get("X-CPA-TRACE-ID"))
	if session == "" || to == "" || response.StatusCode >= 300 {
		return "", "", false
	}
	// CLIProxy keeps a subagent on an account of its own, per model.
	key := session + "\x00" + header.Get("X-Claude-Code-Agent-Id") + "\x00" + model
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.last == nil {
		h.last = map[string]served{}
	}
	previous, known := h.last[key]
	h.last[key] = served{account: to, at: now}
	if len(h.last) > hopPrune {
		for k, s := range h.last {
			if now.Sub(s.at) > hopMemory {
				delete(h.last, k)
			}
		}
	}
	if !known || previous.account == to || now.Sub(previous.at) > hopMemory {
		return "", "", false
	}
	return previous.account, to, true
}

// traceAccount is the account index in CLIProxy's X-CPA-TRACE-ID response
// header, <selection time>-<auth index>-<request id>, or "".
func traceAccount(trace string) string {
	parts := strings.SplitN(trace, "-", 3)
	if len(parts) != 3 {
		return ""
	}
	return parts[1]
}
