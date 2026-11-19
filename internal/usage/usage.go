// Package usage reads token usage from model responses on their way to the
// client and prices it, so the router can report what each thread costs.
package usage

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Tokens is the token usage of one response. Input excludes cached input,
// which CacheRead counts; CacheWrite is input written to the prompt cache.
type Tokens struct {
	Input      int64
	Output     int64
	CacheRead  int64
	CacheWrite int64
}

// Total is every token the response accounted for.
func (t Tokens) Total() int64 {
	return t.Input + t.Output + t.CacheRead + t.CacheWrite
}

// Add sums two usages.
func (t Tokens) Add(other Tokens) Tokens {
	return Tokens{t.Input + other.Input, t.Output + other.Output, t.CacheRead + other.CacheRead, t.CacheWrite + other.CacheWrite}
}

// Price is a model's list price in US dollars per million tokens. A zero
// cache price falls back to the input price.
type Price struct {
	Input      float64 `yaml:"input"`
	Output     float64 `yaml:"output"`
	CacheRead  float64 `yaml:"cache_read"`
	CacheWrite float64 `yaml:"cache_write"`
}

// Cost prices the usage in US dollars.
func (p Price) Cost(t Tokens) float64 {
	cacheRead, cacheWrite := p.CacheRead, p.CacheWrite
	if cacheRead == 0 {
		cacheRead = p.Input
	}
	if cacheWrite == 0 {
		cacheWrite = p.Input
	}
	return (float64(t.Input)*p.Input + float64(t.Output)*p.Output +
		float64(t.CacheRead)*cacheRead + float64(t.CacheWrite)*cacheWrite) / 1e6
}

const (
	// maxBody bounds a buffered non-streaming response.
	maxBody = 8 << 20
	// maxLine bounds one buffered server-sent event line.
	maxLine = 4 << 20
)

// Meter is fed a copy of a Messages API response body and keeps, for every
// field, the latest non-zero usage it has seen: streams report cumulative
// counts (message_start, then message_delta), and a plain JSON response
// reports them once. The latest report wins rather than the largest because
// CLIProxy starts a stream it translated from another API (Codex, OpenAI
// compatible) with an estimate of the whole prompt as input and reports the
// real split between input and cache reads only in message_delta.
type Meter struct {
	stream   bool
	pending  []byte
	overflow bool
	tokens   Tokens
	// event is the type of the server-sent event being read, from its
	// event field or its data's type, and data whether it has data yet.
	event string
	data  bool
	// stopped is set by a stream's complete message_stop event, failed by
	// its complete error event.
	stopped, failed bool
}

// Finished reports whether the response ran to its end: a stream that sent
// a complete message_stop event and no error event, or any whole body. An
// event still unfinished when the stream ended counts for nothing.
func (m *Meter) Finished() bool {
	if !m.stream {
		return true
	}
	return m.stopped && !m.failed
}

// NewMeter returns a meter for a response of the given content type.
func NewMeter(contentType string) *Meter {
	return &Meter{stream: strings.Contains(contentType, "text/event-stream")}
}

// Write observes response bytes; it never fails.
func (m *Meter) Write(p []byte) (int, error) {
	if !m.stream {
		if len(m.pending)+len(p) > maxBody {
			m.overflow, m.pending = true, nil
		} else if !m.overflow {
			m.pending = append(m.pending, p...)
		}
		return len(p), nil
	}
	for rest := p; len(rest) > 0; {
		i := bytes.IndexByte(rest, '\n')
		if i < 0 {
			m.buffer(rest)
			break
		}
		m.buffer(rest[:i])
		m.line(true)
		rest = rest[i+1:]
	}
	return len(p), nil
}

// Tokens is the usage seen so far, finishing any buffered input.
func (m *Meter) Tokens() Tokens {
	if m.stream {
		m.line(false)
	} else if !m.overflow && len(m.pending) > 0 {
		m.observe(m.pending)
		m.pending = nil
	}
	return m.tokens
}

func (m *Meter) buffer(p []byte) {
	if len(m.pending)+len(p) > maxLine {
		m.overflow = true
		return
	}
	m.pending = append(m.pending, p...)
}

// line handles the buffered server-sent event line: a complete one, ended
// by a newline, or the unfinished rest of a stream, whose usage still
// counts but which never completes an event.
func (m *Meter) line(complete bool) {
	line, overflow := m.pending, m.overflow
	m.pending, m.overflow = m.pending[:0], false
	if overflow {
		return
	}
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		// A blank line ends the event.
		if !complete {
			return
		}
		if m.data {
			switch m.event {
			case "message_stop":
				m.stopped = true
			case "error":
				m.failed = true
			}
		}
		m.event, m.data = "", false
		return
	}
	if event, ok := bytes.CutPrefix(line, []byte("event:")); ok {
		if complete {
			m.event = string(bytes.TrimSpace(event))
		}
		return
	}
	data, ok := bytes.CutPrefix(line, []byte("data:"))
	if !ok {
		return
	}
	data = bytes.TrimSpace(data)
	m.data = m.data || complete
	// Only data that may carry usage or end the stream is decoded.
	if bytes.Contains(data, []byte(`"usage"`)) || bytes.Contains(data, []byte("message_stop")) ||
		bytes.Contains(data, []byte(`"error"`)) {
		if kind := m.observe(data); kind != "" && complete {
			m.event = kind
		}
	}
}

// fields is an Anthropic usage object: cache reads and writes are
// reported beside the input.
type fields struct {
	InputTokens   int64 `json:"input_tokens"`
	OutputTokens  int64 `json:"output_tokens"`
	CacheCreation int64 `json:"cache_creation_input_tokens"`
	CacheRead     int64 `json:"cache_read_input_tokens"`
}

type wrapped struct {
	Usage *fields `json:"usage"`
}

// observe folds the usage objects of one JSON document into the meter: a
// top-level usage (message_delta events and whole bodies) and
// message.usage (the message_start event). It returns the document's
// top-level type.
func (m *Meter) observe(document []byte) string {
	var envelope struct {
		wrapped
		Type    string   `json:"type"`
		Message *wrapped `json:"message"`
	}
	if json.Unmarshal(document, &envelope) != nil {
		return ""
	}
	found := []*fields{envelope.Usage}
	if envelope.Message != nil {
		found = append(found, envelope.Message.Usage)
	}
	for _, u := range found {
		if u != nil {
			m.merge(Tokens{Input: u.InputTokens, Output: u.OutputTokens, CacheRead: u.CacheRead, CacheWrite: u.CacheCreation})
		}
	}
	return envelope.Type
}

func (m *Meter) merge(t Tokens) {
	m.tokens = Tokens{latest(m.tokens.Input, t.Input), latest(m.tokens.Output, t.Output),
		latest(m.tokens.CacheRead, t.CacheRead), latest(m.tokens.CacheWrite, t.CacheWrite)}
}

// latest is a usage field after a new report: the new count, unless the
// report leaves the field out or at zero.
func latest(previous, reported int64) int64 {
	if reported > 0 {
		return reported
	}
	return previous
}
