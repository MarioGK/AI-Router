package thread

import (
	"net/http"
	"testing"

	"github.com/MarioGK/AI-Router/internal/wire"
)

func headers(pairs ...string) http.Header {
	h := http.Header{}
	for i := 0; i < len(pairs); i += 2 {
		h.Set(pairs[i], pairs[i+1])
	}
	return h
}

func TestIdentify(t *testing.T) {
	cases := []struct {
		name         string
		header       http.Header
		conversation wire.Conversation
		want         Identity
	}{
		{
			name:   "session header and class",
			header: headers("User-Agent", "claude-cli/2.1.283 (external, cli)", "X-Claude-Code-Session-Id", "cc-1", "X-Claude-Code-Request-Class", "Auxiliary"),
			want:   Identity{Session: "cc-1", Class: Auxiliary},
		},
		{
			name:         "the header wins over the metadata",
			header:       headers("X-Claude-Code-Session-Id", "cc-header"),
			conversation: wire.Conversation{MetadataUserID: `{"session_id":"cc-metadata"}`},
			want:         Identity{Session: "cc-header"},
		},
		{
			name:         "JSON metadata",
			header:       headers("User-Agent", "claude-cli/2.1.283"),
			conversation: wire.Conversation{MetadataUserID: `{"device_id":"d","account_uuid":"a","session_id":"cc-2"}`},
			want:         Identity{Session: "cc-2"},
		},
		{
			name:         "legacy metadata",
			header:       headers("User-Agent", "claude-cli/1.0"),
			conversation: wire.Conversation{MetadataUserID: "user_abc_account_def_session_cc-3"},
			want:         Identity{Session: "cc-3"},
		},
		{
			name:   "no session",
			header: headers("User-Agent", "curl/8"),
			want:   Identity{},
		},
	}
	for _, c := range cases {
		got := Identify(c.header, c.conversation)
		if got.UserAgent != c.header.Get("User-Agent") {
			t.Errorf("%s: user agent = %q", c.name, got.UserAgent)
		}
		if got.UserAgent = ""; got != c.want {
			t.Errorf("%s: Identify = %+v, want %+v", c.name, got, c.want)
		}
	}
}
