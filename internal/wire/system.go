package wire

import "bytes"

// AppendSystem returns a copy of an Anthropic Messages body whose system
// prompt ends with note, every other byte unchanged: a string prompt gains
// the note as a new paragraph, a block list gains a text block after the
// existing ones (so their cache breakpoints still match), and a body
// without a system prompt gets one. ok is false for a body that is not a
// JSON object or whose system prompt has another shape.
func AppendSystem(body []byte, note string) ([]byte, bool) {
	start, end, found, ok := member(body, "system")
	switch {
	case !ok:
		return body, false
	case !found:
		brace := bytes.IndexByte(body, '{')
		entry := append([]byte(`"system":`), quote(note)...)
		if rest := bytes.TrimLeft(body[brace+1:], " \t\r\n"); rest[0] != '}' {
			entry = append(entry, ',')
		}
		return splice(body, brace+1, brace+1, entry), true
	}
	raw := body[start:end]
	switch raw[0] {
	case '"', 'n': // encoding/json decodes null into an empty string
		return splice(body, start, end, quote(str(raw)+"\n\n"+note)), true
	case '[':
		block := append(append([]byte(`{"type":"text","text":`), quote(note)...), '}')
		if len(bytes.TrimSpace(raw[1:len(raw)-1])) > 0 {
			block = append([]byte{','}, block...)
		}
		return splice(body, end-1, end-1, block), true
	}
	return body, false
}
