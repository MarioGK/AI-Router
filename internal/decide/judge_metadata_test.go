package decide

import (
	"encoding/json/v2"
	"io"
	"net/http"
	"strings"
	"testing"
)

type judgeMetadataTransport func(*http.Request) (*http.Response, error)

func (f judgeMetadataTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestJudgePreservesUsageOnInvalidAnswers(t *testing.T) {
	for _, tc := range []struct {
		name   string
		reply  string
		status int
	}{
		{"missing choices", `{"model":"judge-v1","choices":[],"usage":{"prompt_tokens":17,"completion_tokens":3}}`, http.StatusOK},
		{"invalid judgement", `{"model":"judge-v1","choices":[{"message":{"content":"not a judgement"}}],"usage":{"prompt_tokens":17,"completion_tokens":3}}`, http.StatusOK},
		{"provider failure", `{"model":"judge-v1","error":"echoed private task","usage":{"prompt_tokens":17,"completion_tokens":3}}`, http.StatusInternalServerError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &http.Client{Transport: judgeMetadataTransport(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: tc.status, Body: io.NopCloser(strings.NewReader(tc.reply)), Header: http.Header{}}, nil
			})}
			judge := NewJudge("judge", "https://judge.invalid/chat", "configured-judge", "key", client)
			verdict, err := judge.Decide(t.Context(), "task", []Candidate{{ID: "target"}}, nil)
			if err == nil || verdict.Model != "judge-v1" || verdict.InputTokens != 17 || verdict.OutputTokens != 3 || !verdict.UsageKnown || verdict.Digest == "" {
				t.Fatalf("verdict=%+v, err=%v", verdict, err)
			}
			if tc.status != http.StatusOK && (verdict.Response != "" || strings.Contains(err.Error(), "private task")) {
				t.Fatalf("provider failure body retained: %+v, err=%v", verdict, err)
			}
		})
	}
}

func TestJudgeReportedUsageCompleteness(t *testing.T) {
	for _, tc := range []struct {
		name  string
		usage map[string]any
		known bool
	}{
		{"both zero", map[string]any{"prompt_tokens": 0, "completion_tokens": 0}, true},
		{"partial", map[string]any{"prompt_tokens": 17}, false},
		{"missing", nil, false},
		{"negative", map[string]any{"prompt_tokens": 17, "completion_tokens": -1}, false},
		{"malformed optional", map[string]any{"prompt_tokens": 17, "completion_tokens": "unknown"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"choices": []map[string]any{{"message": map[string]string{"content": `{"probabilities":{"target":1}}`}}},
				"usage":   tc.usage,
			})
			if err != nil {
				t.Fatal(err)
			}
			client := &http.Client{Transport: judgeMetadataTransport(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(string(body))), Header: http.Header{}}, nil
			})}
			judge := NewJudge("judge", "https://judge.invalid/chat", "configured-judge", "key", client)
			verdict, err := judge.Decide(t.Context(), "task", []Candidate{{ID: "target"}}, nil)
			if err != nil || verdict.UsageKnown != tc.known || verdict.OutputTokens < 0 || verdict.Model != "configured-judge" {
				t.Fatalf("verdict=%+v, err=%v", verdict, err)
			}
			if input, ok := tc.usage["prompt_tokens"].(int); ok && input > 0 && verdict.InputTokens != int64(input) {
				t.Fatalf("partial reported input lost: %+v", verdict)
			}
		})
	}
}
