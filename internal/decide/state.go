package decide

import (
	"strings"
	"unicode/utf8"
)

// StateBuilder builds the text a decider judges, stripping client wrapper
// blocks and keeping representative head, middle and tail excerpts.
type StateBuilder struct {
	stripTags map[string]struct{}
	maxChars  int
}

// NewStateBuilder records the tags whose wrapper blocks are stripped.
func NewStateBuilder(stripTags []string, maxChars int) *StateBuilder {
	builder := &StateBuilder{stripTags: make(map[string]struct{}, len(stripTags)), maxChars: maxChars}
	for _, tag := range stripTags {
		builder.stripTags[tag] = struct{}{}
	}
	return builder
}

// State is the text a decider judges and what became of the user text.
type State struct {
	// Text is the decider state; empty when no user text remains.
	Text string
	// Prompt is the user text without the wrapper blocks, cut only past
	// maxPrompt runes: the prompt as the user wrote it.
	Prompt string
	// Chars is the length of the user text after stripping, in runes;
	// Truncated is set when it was cut to fit.
	Chars     int
	Truncated bool
}

// maxPrompt bounds the prompt a state keeps whole.
const maxPrompt = 100_000

// BuildLatest returns the newest user turn with text after wrapper stripping.
// users is newest first; tool-result text must not be included.
func (b *StateBuilder) BuildLatest(users []string, hasImage bool) State {
	for _, user := range users {
		if state := b.Build(user, hasImage); state.Text != "" {
			return state
		}
	}
	return State{}
}

// Build returns the decider state of the user text: whether the thread
// carries images, then the text.
func (b *StateBuilder) Build(user string, hasImage bool) State {
	text := strings.TrimSpace(b.strip(user))
	if text == "" {
		return State{}
	}
	images := "no"
	if hasImage {
		images = "yes"
	}
	chars := utf8.RuneCountInString(text)
	return State{
		Text:      "images=" + images + "\n\n" + cutCount(text, b.maxChars, chars),
		Prompt:    cutCount(text, maxPrompt, chars),
		Chars:     chars,
		Truncated: chars > b.maxChars,
	}
}

// strip removes complete wrappers at a line's first nonspace position, also
// allowing adjacent prefix wrappers. Tags in prose, quotes and Markdown code
// are task data. An incomplete wrapper is kept rather than guessing its end.
func (b *StateBuilder) strip(text string) string {
	if len(b.stripTags) == 0 || !strings.Contains(text, "<") {
		return text
	}
	var result strings.Builder
	codes := countCodeRuns(text)
	linePrefix, indent := true, 0
	var fence byte
	fenceSize := 0
	for pos := 0; pos < len(text); {
		if text[pos] == '\n' {
			result.WriteByte(text[pos])
			pos++
			linePrefix, indent = true, 0
			continue
		}
		if linePrefix && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\r') {
			if text[pos] == ' ' {
				indent++
			} else if text[pos] == '\t' {
				indent += 4
			}
			result.WriteByte(text[pos])
			pos++
			continue
		}
		if linePrefix && indent < 4 {
			if size := fenceRun(text, pos); size >= 3 {
				if fence == 0 {
					fence, fenceSize = text[pos], size
				} else if text[pos] == fence && size >= fenceSize && closesFence(text, pos, size) {
					fence, fenceSize = 0, 0
				}
				result.WriteString(text[pos : pos+size])
				pos += size
				linePrefix = false
				continue
			}
			if fence == 0 && text[pos] == '<' {
				if tag, ok := b.tag(text, pos); ok && !tag.closing {
					if end, complete := b.wrapperEnd(text, tag, &codes); complete {
						pos = end
						indent = 0
						continue
					}
					result.WriteString(text[pos:])
					return result.String()
				}
			}
		}
		if fence == 0 {
			if end := literalEnd(text, pos, &codes); end > pos {
				result.WriteString(text[pos:end])
				pos = end
				linePrefix = false
				continue
			}
		}
		linePrefix = false
		result.WriteByte(text[pos])
		pos++
	}
	return result.String()
}

type wrapperTag struct {
	name                 string
	end                  int
	closing, selfClosing bool
}

// tag recognizes configured tags, respecting quoted attribute values.
func (b *StateBuilder) tag(text string, pos int) (wrapperTag, bool) {
	if pos >= len(text) || text[pos] != '<' {
		return wrapperTag{}, false
	}
	pos++
	closing := pos < len(text) && text[pos] == '/'
	if closing {
		pos++
	}
	start := pos
	for pos < len(text) && !tagSpace(text[pos]) && text[pos] != '>' && text[pos] != '/' && text[pos] != '<' {
		pos++
	}
	name := text[start:pos]
	if _, ok := b.stripTags[name]; !ok {
		return wrapperTag{}, false
	}
	if pos < len(text) && text[pos] == '/' && (pos+1 == len(text) || text[pos+1] != '>') {
		return wrapperTag{}, false
	}
	var quote byte
	for ; pos < len(text); pos++ {
		c := text[pos]
		if quote != 0 {
			if c == quote {
				quote = 0
			}
			continue
		}
		switch {
		case c == '>':
			return wrapperTag{name: name, end: pos + 1, closing: closing, selfClosing: text[pos-1] == '/'}, true
		case closing && !tagSpace(c):
			return wrapperTag{}, false
		case c == '\'' || c == '"':
			quote = c
		case c == '<':
			return wrapperTag{}, false
		}
	}
	return wrapperTag{}, false
}

func tagSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f'
}

// wrapperEnd balances nested configured tags; literals inside wrappers do
// not introduce or close wrappers.
func (b *StateBuilder) wrapperEnd(text string, first wrapperTag, codes *codeRuns) (int, bool) {
	if first.selfClosing {
		return first.end, true
	}
	stack := []string{first.name}
	for pos := first.end; pos < len(text); {
		if wrapperLinePrefix(text, first.end, pos) {
			if size := fenceRun(text, pos); size >= 3 {
				end, complete := fencedEnd(text, pos, size)
				if !complete {
					return 0, false
				}
				pos = end
				continue
			}
		}
		if end := literalEnd(text, pos, codes); end > pos {
			pos = end
			continue
		}
		if text[pos] != '<' {
			pos++
			continue
		}
		tag, ok := b.tag(text, pos)
		if !ok {
			pos++
			continue
		}
		pos = tag.end
		if tag.closing {
			if tag.name != stack[len(stack)-1] {
				return 0, false
			}
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				return pos, true
			}
		} else if !tag.selfClosing {
			stack = append(stack, tag.name)
		}
	}
	return 0, false
}

func fenceRun(text string, pos int) int {
	if text[pos] != '`' && text[pos] != '~' {
		return 0
	}
	end := pos + 1
	for end < len(text) && text[end] == text[pos] {
		end++
	}
	return end - pos
}

func closesFence(text string, pos, size int) bool {
	line, _, _ := strings.Cut(text[pos+size:], "\n")
	return strings.Trim(line, " \t\r") == ""
}

func wrapperLinePrefix(text string, start, pos int) bool {
	for indent := 0; ; indent++ {
		if pos == start || text[pos-1] == '\n' {
			return true
		}
		if indent == 3 || text[pos-1] != ' ' {
			return false
		}
		pos--
	}
}

// fencedEnd requires a closing fence on its own line, for both Markdown
// delimiter kinds. Wrapper-looking content inside the fence stays literal.
func fencedEnd(text string, pos, size int) (int, bool) {
	for start := pos + size; start < len(text); {
		offset := strings.IndexByte(text[start:], '\n')
		if offset < 0 {
			return 0, false
		}
		start += offset + 1
		closing := start
		for closing < len(text) && text[closing] == ' ' && closing-start < 4 {
			closing++
		}
		if closing-start >= 4 || closing == len(text) || text[closing] != text[pos] {
			continue
		}
		if run := fenceRun(text, closing); run >= size && closesFence(text, closing, run) {
			return closing + run, true
		}
	}
	return 0, false
}

// codeRuns counts remaining backtick runs by length. Its cursor accounts
// for skipped tags, quotes and fences before each inline-code lookup. This
// avoids searching the suffix repeatedly for unmatched delimiter lengths.
type codeRuns struct {
	remaining map[int]int
	consumed  int
}

func countCodeRuns(text string) codeRuns {
	codes := codeRuns{remaining: map[int]int{}}
	for start := 0; start < len(text); {
		offset := strings.IndexByte(text[start:], '`')
		if offset < 0 {
			break
		}
		pos := start + offset
		size := fenceRun(text, pos)
		codes.remaining[size]++
		start = pos + size
	}
	return codes
}

func (c *codeRuns) consumeBefore(text string, pos int) {
	for c.consumed < pos {
		offset := strings.IndexByte(text[c.consumed:pos], '`')
		if offset < 0 {
			c.consumed = pos
			break
		}
		start := c.consumed + offset
		size := fenceRun(text, start)
		c.remaining[size]--
		c.consumed = start + size
	}
}

// literalEnd skips a quoted or code literal when its closing delimiter is
// present, or consumes an entire unmatched backtick run. Successful code
// searches cover spans subsequently skipped by the parser, so scanning is
// linear. Apostrophes within words are ordinary prose.
func literalEnd(text string, pos int, codes *codeRuns) int {
	c := text[pos]
	if c == '`' {
		size := fenceRun(text, pos)
		codes.consumeBefore(text, pos)
		if codes.remaining[size] <= 1 {
			return pos + size
		}
		for start := pos + size; start < len(text); {
			offset := strings.IndexByte(text[start:], '`')
			if offset < 0 {
				break
			}
			closing := start + offset
			run := fenceRun(text, closing)
			if run == size {
				return closing + run
			}
			start = closing + run
		}
		return pos + size
	}
	if c != '\'' && c != '"' || escapedQuote(text, pos) || pos > 0 && ((text[pos-1] >= 'a' && text[pos-1] <= 'z') || (text[pos-1] >= 'A' && text[pos-1] <= 'Z')) {
		return pos
	}
	for end := pos + 1; end < len(text) && text[end] != '\n'; end++ {
		if text[end] == '\\' {
			end++
			continue
		}
		if text[end] == c {
			return end + 1
		}
	}
	return pos
}

func escapedQuote(text string, pos int) bool {
	start := pos
	for start > 0 && text[start-1] == '\\' {
		start--
	}
	return (pos-start)%2 != 0
}

const omission = "\n[…]\n"

// cutCount reuses the count computed for state metadata and copies only
// retained head, center and tail excerpts, including omission markers in the
// budget. It avoids a whole-prompt rune allocation on large input.
func cutCount(text string, maxChars, chars int) string {
	if chars <= maxChars {
		return text
	}
	if maxChars <= 0 {
		return ""
	}
	markerChars := utf8.RuneCountInString(omission)
	if maxChars <= 2*markerChars {
		if len(text) == chars {
			return text[:maxChars]
		}
		return string([]rune(text)[:maxChars])
	}
	payload := maxChars - 2*markerChars
	head := payload / 3
	middle := payload / 3
	tail := payload - head - middle
	center := (chars - middle) / 2
	if len(text) == chars {
		return text[:head] + omission + text[center:center+middle] + omission + text[chars-tail:]
	}
	if !utf8.ValidString(text) {
		// Match rune conversion's replacement behavior for malformed UTF-8.
		runes := []rune(text)
		return string(runes[:head]) + omission + string(runes[center:center+middle]) + omission + string(runes[chars-tail:])
	}
	targets := [4]int{head, center, center + middle, chars - tail}
	var offsets [4]int
	runeIndex, next := 0, 0
	for byteIndex := range text {
		for next < len(targets) && runeIndex == targets[next] {
			offsets[next] = byteIndex
			next++
		}
		if next == len(targets) {
			break
		}
		runeIndex++
	}
	return text[:offsets[0]] + omission + text[offsets[1]:offsets[2]] + omission + text[offsets[3]:]
}
