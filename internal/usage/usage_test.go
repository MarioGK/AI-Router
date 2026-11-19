package usage

import (
	"math"
	"strings"
	"testing"
)

func TestMeterTellsAFinishedStream(t *testing.T) {
	const start = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n\n"
	for _, tc := range []struct {
		name, contentType, body string
		want                    bool
	}{
		{"whole body", "application/json", `{"usage":{"input_tokens":3}}`, true},
		{"stopped", "text/event-stream", start + "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", true},
		{"stopped without event lines", "text/event-stream", start + "data: {\"type\":\"message_stop\"}\n\n", true},
		{"spaced type", "text/event-stream", start + "data: {\"type\": \"message_stop\"}\n\n", true},
		{"type not first", "text/event-stream", start + "data: {\"extra\":0,\"type\":\"message_stop\"}\n\n", true},
		{"CRLF", "text/event-stream", "event: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n", true},
		{"cut", "text/event-stream", start, false},
		{"stop header only", "text/event-stream", start + "event: message_stop\n", false},
		{"stop header before a blank line without data", "text/event-stream", start + "event: message_stop\n\n", false},
		{"stop data without its blank line", "text/event-stream", start + "event: message_stop\ndata: {\"type\":\"message_stop\"}\n", false},
		{"stop data cut mid-line", "text/event-stream", start + "event: message_stop\ndata: {\"type\":\"message_stop\"}", false},
		{"error", "text/event-stream", start + "event: error\ndata: {\"type\":\"error\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", false},
		{"spaced error", "text/event-stream", start + "data: {\"error\":{\"type\":\"overloaded_error\"}, \"type\": \"error\"}\n\ndata: {\"type\":\"message_stop\"}\n\n", false},
		{"quoted stop", "text/event-stream", start + "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"{\\\"type\\\":\\\"message_stop\\\"}\"}}\n\n", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			meter := NewMeter(tc.contentType)
			for i := range len(tc.body) {
				meter.Write([]byte(tc.body[i : i+1]))
			}
			if got := meter.Finished(); got != tc.want {
				t.Fatalf("Finished = %v, want %v", got, tc.want)
			}
			meter.Tokens()
		})
	}
}

func feed(contentType, body string, chunk int) Tokens {
	meter := NewMeter(contentType)
	for len(body) > 0 {
		n := min(chunk, len(body))
		meter.Write([]byte(body[:n]))
		body = body[n:]
	}
	return meter.Tokens()
}

func TestMeterReadsTheMessagesAPI(t *testing.T) {
	cases := map[string]struct {
		contentType, body string
		want              Tokens
	}{
		"anthropic stream": {"text/event-stream", "event: message_start\n" +
			`data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":12,"cache_creation_input_tokens":300,"cache_read_input_tokens":4000,"output_tokens":1}}}` + "\n\n" +
			"event: content_block_delta\n" + `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi \"usage\""}}` + "\n\n" +
			"event: message_delta\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":87}}` + "\n\n",
			Tokens{Input: 12, Output: 87, CacheRead: 4000, CacheWrite: 300}},
		// CLIProxy's translation of a Codex or OpenAI-compatible stream: the
		// start estimates the whole prompt, the delta splits it for real.
		"translated stream": {"text/event-stream", "event: message_start\n" +
			`data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":100000,"output_tokens":0}}}` + "\n\n" +
			"event: message_delta\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10000,"output_tokens":500,"cache_read_input_tokens":90000}}` + "\n\n",
			Tokens{Input: 10000, Output: 500, CacheRead: 90000}},
		"anthropic body": {"application/json",
			`{"id":"m","type":"message","usage":{"input_tokens":5,"output_tokens":9,"cache_read_input_tokens":100}}`,
			Tokens{Input: 5, Output: 9, CacheRead: 100}},
		"no usage":        {"text/event-stream", "data: {\"x\":1}\n\n", Tokens{}},
		"count tokens":    {"application/json", `{"input_tokens":42}`, Tokens{}},
		"unterminated":    {"text/event-stream", `data: {"usage":{"input_tokens":3,"output_tokens":4}}`, Tokens{Input: 3, Output: 4}},
		"not json at all": {"application/json", `<html>`, Tokens{}},
	}
	for name, c := range cases {
		for _, chunk := range []int{1, 7, 1 << 20} {
			if got := feed(c.contentType, c.body, chunk); got != c.want {
				t.Errorf("%s (chunks of %d): Tokens = %+v, want %+v", name, chunk, got, c.want)
			}
		}
	}
}

func TestMeterBoundsItsBuffers(t *testing.T) {
	meter := NewMeter("application/json")
	meter.Write([]byte(`{"usage":{"input_tokens":1,"output_tokens":1},"pad":"`))
	meter.Write([]byte(strings.Repeat("x", maxBody)))
	if got := meter.Tokens(); got != (Tokens{}) {
		t.Fatalf("an oversized body was parsed: %+v", got)
	}
	stream := NewMeter("text/event-stream")
	stream.Write([]byte("data: " + strings.Repeat("y", maxLine+1)))
	stream.Write([]byte("\n" + `data: {"usage":{"input_tokens":2,"output_tokens":3}}` + "\n"))
	if got := stream.Tokens(); got != (Tokens{Input: 2, Output: 3}) {
		t.Fatalf("the line after an oversized one was lost: %+v", got)
	}
}

func TestPriceCost(t *testing.T) {
	opus := Price{Input: 4, Output: 20, CacheRead: 0.2, CacheWrite: 5}
	got := opus.Cost(Tokens{Input: 1_000, Output: 2_000, CacheRead: 100_000, CacheWrite: 10_000})
	want := (1_000*4 + 2_000*20 + 100_000*0.2 + 10_000*5) / 1e6
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("Cost = %v, want %v", got, want)
	}
	if got := (Price{Input: 1, Output: 2}).Cost(Tokens{CacheRead: 1e6, CacheWrite: 1e6}); got != 2 {
		t.Fatalf("cache prices default to input: %v", got)
	}
}
