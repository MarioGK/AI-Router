package wire

import (
	"bytes"
	"encoding/json"
)

// maxDepth is how deeply encoding/json lets arrays and objects nest.
const maxDepth = 10000

// scanner reads JSON in place: it checks the grammar as encoding/json does
// and finds where values start and end, without decoding or copying them.
// Its methods report false when the input is malformed.
type scanner struct {
	data  []byte
	pos   int
	depth int
}

// member finds the first top-level member of a JSON object body named name
// and returns the span of its value. The body is read only as far as that
// value: ok reports whether it is a well-formed JSON object that far, which
// is to its end when there is no such member.
func member(body []byte, name string) (start, end int, found, ok bool) {
	s := scanner{data: body}
	ok = s.object(func(key []byte) bool {
		start = s.pos
		if !s.value() {
			return false
		}
		found = string(unquote(key)) == name
		return !found
	}) && s.end()
	if !found {
		return 0, 0, false, ok
	}
	return start, s.pos, true, true
}

// object reads the object at the current position, calling member with
// the raw key of each member and the position at its value, which member
// must read. A walk that member stops reports false.
func (s *scanner) object(member func(key []byte) bool) bool {
	return s.list('{', '}', func() bool {
		start := s.pos
		if !s.at('"') || !s.quoted() {
			return false
		}
		key := s.data[start:s.pos]
		if !s.next(':') {
			return false
		}
		s.space()
		return member(key)
	})
}

// array reads the array at the current position, calling element with the
// position at each element, which element must read. A walk that element
// stops reports false.
func (s *scanner) array(element func() bool) bool {
	return s.list('[', ']', element)
}

// list reads the object or array that opening and closing delimit, calling
// item with the position at each of its items.
func (s *scanner) list(opening, closing byte, item func() bool) bool {
	if !s.next(opening) || s.depth == maxDepth {
		return false
	}
	s.depth++
	if s.next(closing) {
		s.depth--
		return true
	}
	for {
		s.space()
		if !item() {
			return false
		}
		if s.next(closing) {
			s.depth--
			return true
		}
		if !s.next(',') {
			return false
		}
	}
}

// value skips whitespace and the value that follows it.
func (s *scanner) value() bool {
	s.space()
	if s.pos == len(s.data) {
		return false
	}
	switch s.data[s.pos] {
	case '{':
		return s.object(func([]byte) bool { return s.value() })
	case '[':
		return s.array(s.value)
	case '"':
		return s.quoted()
	case 't':
		return s.literal("true")
	case 'f':
		return s.literal("false")
	case 'n':
		return s.literal("null")
	}
	return s.number()
}

// quoted skips the string at the current position.
func (s *scanner) quoted() bool {
	data, i := s.data, s.pos+1
	for {
		i += plain(data[i:])
		switch {
		case i == len(data):
			return false
		case data[i] == '"':
			s.pos = i + 1
			return true
		case data[i] == '\\':
			n := escape(data[i:])
			if n == 0 {
				return false
			}
			i += n
		default:
			return false
		}
	}
}

// number skips the number at the current position.
func (s *scanner) number() bool {
	data, i := s.data, s.pos
	if i < len(data) && data[i] == '-' {
		i++
	}
	switch {
	case i < len(data) && data[i] == '0':
		i++
	case i < len(data) && '1' <= data[i] && data[i] <= '9':
		i = digits(data, i)
	default:
		return false
	}
	if i < len(data) && data[i] == '.' {
		start := i + 1
		if i = digits(data, start); i == start {
			return false
		}
	}
	if i < len(data) && (data[i] == 'e' || data[i] == 'E') {
		if i++; i < len(data) && (data[i] == '+' || data[i] == '-') {
			i++
		}
		start := i
		if i = digits(data, start); i == start {
			return false
		}
	}
	s.pos = i
	return true
}

// literal skips word, which must be at the current position.
func (s *scanner) literal(word string) bool {
	end := s.pos + len(word)
	if end > len(s.data) || string(s.data[s.pos:end]) != word {
		return false
	}
	s.pos = end
	return true
}

// next skips whitespace and then c, reporting whether c was there.
func (s *scanner) next(c byte) bool {
	s.space()
	if !s.at(c) {
		return false
	}
	s.pos++
	return true
}

// at reports whether c is at the current position.
func (s *scanner) at(c byte) bool {
	return s.pos < len(s.data) && s.data[s.pos] == c
}

// end reports whether nothing but whitespace is left.
func (s *scanner) end() bool {
	s.space()
	return s.pos == len(s.data)
}

// space skips whitespace.
func (s *scanner) space() {
	for s.pos < len(s.data) {
		switch s.data[s.pos] {
		case ' ', '\t', '\n', '\r':
			s.pos++
		default:
			return
		}
	}
}

// plain returns the length of the run of bytes at the start of data that a
// string holds as they are: all but quotes, backslashes and control bytes.
func plain(data []byte) int {
	for i, c := range data {
		if c < ' ' || c == '"' || c == '\\' {
			return i
		}
	}
	return len(data)
}

// escape returns the length of the escape sequence at the start of data,
// or 0 when it is not one.
func escape(data []byte) int {
	if len(data) < 2 {
		return 0
	}
	switch data[1] {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		return 2
	case 'u':
		if len(data) >= 6 && hex(data[2]) && hex(data[3]) && hex(data[4]) && hex(data[5]) {
			return 6
		}
	}
	return 0
}

// digits returns the position after the run of digits at i.
func digits(data []byte, i int) int {
	for i < len(data) && '0' <= data[i] && data[i] <= '9' {
		i++
	}
	return i
}

// hex reports whether c is a hexadecimal digit.
func hex(c byte) bool {
	return '0' <= c && c <= '9' || 'a' <= c && c <= 'f' || 'A' <= c && c <= 'F'
}

// keep reads the value at the current position and keeps it in field when
// it is a string, as encoding/json decodes into a string field: null and
// values of other types leave the field be.
func (s *scanner) keep(field *[]byte) bool {
	start := s.pos
	if !s.value() {
		return false
	}
	if s.data[start] == '"' {
		*field = s.data[start:s.pos]
	}
	return true
}

// decode reads the value at the current position into v as encoding/json
// decodes into a struct field: a value of another type leaves v be.
func (s *scanner) decode(v any) bool {
	start := s.pos
	if !s.value() {
		return false
	}
	_ = json.Unmarshal(s.data[start:s.pos], v)
	return true
}

// stringy reports whether the value at the current position decodes into
// a Go string: a string, or null, which leaves it be.
func (s *scanner) stringy() bool {
	return s.at('"') || s.at('n')
}

// named reports whether a raw JSON key names the struct field name, which
// encoding/json matches without regard to case.
func named(key []byte, name string) bool {
	return bytes.EqualFold(unquote(key), []byte(name))
}

// unquote returns the text of a raw JSON string for comparison, decoding
// it only when it holds escapes; it returns nil for anything else.
func unquote(raw []byte) []byte {
	switch {
	case len(raw) < 2 || raw[0] != '"':
		return nil
	case bytes.IndexByte(raw, '\\') < 0:
		return raw[1 : len(raw)-1]
	}
	return []byte(str(raw))
}

// str returns the Go string a raw JSON value decodes into: the text of a
// string, or "" for anything else.
func str(raw []byte) string {
	var text string
	_ = json.Unmarshal(raw, &text)
	return text
}
