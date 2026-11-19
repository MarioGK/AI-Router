package wire

import "strings"

// Conversation is what routing needs from a request body.
type Conversation struct {
	// LatestUser is the text of the newest user turn that carries text.
	// Tool results are not user text.
	LatestUser string
	// UserTexts keeps text-bearing user turns, newest first, so routing can
	// skip client-wrapper-only turns after applying its configured cleaner.
	// Text nested inside tool results is excluded.
	UserTexts []string
	// HasImage is set when any message carries an image or document, so
	// the thread needs a multimodal model.
	HasImage bool
	// HasTools is set when the request offers the model any tool, as every
	// turn of Claude Code's agent loop does and its background calls
	// (titles, summaries) do not.
	HasTools bool
	// Prompted is set when the newest message is the user's and carries
	// text but no tool result: a new prompt rather than the next turn of a
	// tool loop. LatestUser is then its text.
	Prompted       bool
	MetadataUserID string
}

// Parse extracts the conversation from a Messages or CountTokens body;
// other kinds carry none. Malformed bodies yield an empty conversation;
// CLIProxy reports the error. The body is read in place and only the
// values kept are decoded, the way encoding/json decodes them into a
// struct: names match without regard to case and a later member overrides
// an earlier one.
func Parse(kind Kind, body []byte) Conversation {
	if kind != Messages && kind != CountTokens {
		return Conversation{}
	}
	var turns turns
	var metadata struct {
		UserID string `json:"user_id"`
	}
	tools := false
	s := scanner{data: body}
	if !s.object(func(key []byte) bool {
		switch {
		case named(key, "messages"):
			return turns.read(&s)
		case named(key, "metadata"):
			return s.decode(&metadata)
		case named(key, "tools"):
			tools = false
			if s.at('[') {
				return s.array(func() bool {
					tools = true
					return s.value()
				})
			}
		}
		return s.value()
	}) || !s.end() {
		return Conversation{}
	}
	users := turns.texts()
	latest := ""
	if len(users) > 0 {
		latest = users[0]
	}
	return Conversation{LatestUser: latest, UserTexts: users, HasImage: turns.image, HasTools: tools,
		Prompted: turns.prompted(), MetadataUserID: metadata.UserID}
}

// turns is what routing needs from the messages of a request: whether any
// carries an image, the content of each user message, oldest first, and
// whether the newest message is the user's.
type turns struct {
	image bool
	users [][]byte
	// user is set when the newest message is the user's.
	user bool
}

// read reads the messages at the current position: a list replaces any
// read before, null clears them and any other value, which encoding/json
// would not decode into a slice, leaves them be.
func (t *turns) read(s *scanner) bool {
	switch {
	case s.at('['):
		*t = turns{}
		return s.array(func() bool { return t.message(s) })
	case s.at('n'):
		*t = turns{}
	}
	return s.value()
}

// message reads the message at the current position, a role and a content
// that is a string or a list of content blocks; a value that is not an
// object adds nothing.
func (t *turns) message(s *scanner) bool {
	if !s.at('{') {
		return s.value()
	}
	var role, content []byte
	image := false
	if !s.object(func(key []byte) bool {
		switch {
		case named(key, "role"):
			return s.keep(&role)
		case named(key, "content"):
			start := s.pos
			var ok bool
			image, ok = s.image()
			content = s.data[start:s.pos]
			return ok
		}
		return s.value()
	}) {
		return false
	}
	t.image = t.image || image
	t.user = string(unquote(role)) == "user"
	if t.user {
		t.users = append(t.users, content)
	}
	return true
}

// prompted reports whether the newest message is the user's, with text and
// without a tool result.
func (t *turns) prompted() bool {
	if !t.user || len(t.users) == 0 {
		return false
	}
	content := t.users[len(t.users)-1]
	results := false
	s := scanner{data: content}
	s.blocks(func(kind, _ []byte, _ bool) {
		results = results || string(unquote(kind)) == "tool_result"
	})
	return !results && text(content) != ""
}

// texts returns text-bearing user messages, newest first.
func (t *turns) texts() []string {
	var users []string
	for i := len(t.users) - 1; i >= 0; i-- {
		if text := text(t.users[i]); text != "" {
			users = append(users, text)
		}
	}
	return users
}

// image reads the message content at the current position and reports
// whether it carries an image or document, directly or in a tool result.
func (s *scanner) image() (image, ok bool) {
	valid, ok := s.blocks(func(kind, _ []byte, nested bool) {
		switch string(unquote(kind)) {
		case "image", "document":
			image = true
		case "tool_result":
			image = image || nested
		}
	})
	return image && valid, ok
}

// text returns the user-visible text of a message content: a string, or
// the text blocks of a list, one per line.
func text(content []byte) string {
	s := scanner{data: content}
	if s.stringy() {
		return strings.TrimSpace(str(content))
	}
	var texts []string
	if valid, _ := s.blocks(func(kind, raw []byte, _ bool) {
		if string(unquote(kind)) == "text" {
			if text := strings.TrimSpace(str(raw)); text != "" {
				texts = append(texts, text)
			}
		}
	}); !valid {
		return ""
	}
	return strings.Join(texts, "\n")
}

// blocks reads the message content at the current position and, when it
// is a list, calls block with the raw type and text of each content block
// and whether the block's own content carries an image. valid reports
// whether encoding/json decodes the content into blocks with a string type
// and text; a content that it does not carries neither text nor images.
func (s *scanner) blocks(block func(kind, text []byte, image bool)) (valid, ok bool) {
	if !s.at('[') {
		return false, s.value()
	}
	valid = true
	ok = s.array(func() bool {
		if !s.at('{') {
			valid = valid && s.at('n') // null decodes into an empty block
			return s.value()
		}
		var kind, text []byte
		image := false
		if !s.object(func(key []byte) bool {
			switch {
			case named(key, "type"):
				valid = valid && s.stringy()
				return s.keep(&kind)
			case named(key, "text"):
				valid = valid && s.stringy()
				return s.keep(&text)
			case named(key, "content"):
				var ok bool
				image, ok = s.image()
				return ok
			}
			return s.value()
		}) {
			return false
		}
		block(kind, text, image)
		return true
	})
	return valid, ok
}
