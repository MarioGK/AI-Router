// Package wire understands just enough of the Anthropic Messages API to
// route a Claude Code request: the endpoint, the top-level model and the
// conversation text.
package wire

import (
	"net/http"
	"strings"
)

// Kind is the kind of endpoint a request targets.
type Kind int

const (
	Other Kind = iota
	Messages
	CountTokens
	Models
)

var kindNames = map[Kind]string{
	Other: "other", Messages: "messages", CountTokens: "count_tokens", Models: "models",
}

func (k Kind) String() string {
	return kindNames[k]
}

// KindOf classifies a request by method and path.
func KindOf(method, path string) Kind {
	path = strings.TrimSuffix(path, "/")
	switch {
	case method == http.MethodGet && path == "/v1/models":
		return Models
	case method != http.MethodPost:
		return Other
	case path == "/v1/messages":
		return Messages
	case path == "/v1/messages/count_tokens":
		return CountTokens
	}
	return Other
}
