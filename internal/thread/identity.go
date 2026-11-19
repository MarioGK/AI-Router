// Package thread identifies the Claude Code thread a request belongs to,
// from what Claude Code sends with every request.
package thread

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/MarioGK/AI-Router/internal/wire"
)

// Request classes Claude Code sends in x-claude-code-request-class when
// CLAUDE_CODE_GATEWAY_HINT_HEADERS=1.
const (
	Auxiliary  = "auxiliary"
	Compaction = "compaction"
)

// Identity is the thread a request belongs to.
type Identity struct {
	// Session identifies the Claude Code session, shared by its subagents;
	// empty when the request names none.
	Session string
	// Agent identifies the subagent within the session that sent the
	// request; empty for the session's main thread.
	Agent string
	// Class is the Claude Code request class, when sent.
	Class string
	// UserAgent names the Claude Code version.
	UserAgent string
}

// Identify derives the identity of a request. Claude Code sends
// X-Claude-Code-Session-Id, shared by its subagents, and the session inside
// metadata.user_id; a subagent's requests also carry
// X-Claude-Code-Agent-Id.
func Identify(header http.Header, conversation wire.Conversation) Identity {
	return Identity{
		Session:   first(header.Get("X-Claude-Code-Session-Id"), claudeSession(conversation.MetadataUserID)),
		Agent:     strings.TrimSpace(header.Get("X-Claude-Code-Agent-Id")),
		Class:     strings.ToLower(header.Get("X-Claude-Code-Request-Class")),
		UserAgent: header.Get("User-Agent"),
	}
}

// claudeSession extracts the session from an Anthropic metadata.user_id,
// which Claude Code sends either as a JSON object or as
// user_<hash>_account_<uuid>_session_<uuid>.
func claudeSession(userID string) string {
	var structured struct {
		SessionID string `json:"session_id"`
	}
	if json.Unmarshal([]byte(userID), &structured) == nil {
		return structured.SessionID
	}
	if _, session, ok := strings.Cut(userID, "_session_"); ok {
		return session
	}
	return ""
}

func first(values ...string) string {
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return ""
}
