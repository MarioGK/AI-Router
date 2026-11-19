package decide

import (
	"bytes"
	"context"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

type systemOneRoundTrip func(*http.Request) (*http.Response, error)

func (f systemOneRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

type systemOneTrackedBody struct {
	reader io.Reader
	read   int
	closed bool
}

type systemOneReadFunc func([]byte) (int, error)

func (f systemOneReadFunc) Read(p []byte) (int, error) {
	return f(p)
}

func (b *systemOneTrackedBody) Read(p []byte) (int, error) {
	n, err := b.reader.Read(p)
	b.read += n
	return n, err
}

func (b *systemOneTrackedBody) Close() error {
	b.closed = true
	return nil
}

func systemOneFixture(status int, reply string) (*http.Client, *systemOneTrackedBody) {
	body := &systemOneTrackedBody{reader: strings.NewReader(reply)}
	client := &http.Client{Transport: systemOneRoundTrip(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: status, Body: body, Header: http.Header{}, Request: r}, nil
	})}
	return client, body
}

func TestSystemOneContractWireAndDigests(t *testing.T) {
	const state = "images=no\n\nadd a --dry-run flag"
	var sent [][]byte
	var bodies []*systemOneTrackedBody
	client := &http.Client{Transport: systemOneRoundTrip(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodPost || r.URL.String() != "https://classifier.invalid/v1/systemone" ||
			r.Header.Get("Content-Type") != "application/json" || r.Header.Get("Authorization") != "Bearer test-key" {
			t.Error("unexpected classifier request method, endpoint or headers")
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		sent = append(sent, body)
		reply := `{"model":"jev-test","answers":{"model":{"choice":"glm-5.3","probabilities":{"glm-5.3":1}},"reasoning":{"probabilities":{"2":1}}},"usage":{"input_tokens":17,"output_tokens":5}}`
		if len(sent) == 2 {
			reply = `{"model":"jev-test","answers":{"pair":{"choice":"glm-5.3","probabilities":{"glm-5.3":0.8,"gpt-6.1-sol":0.2}}},"usage":{"input_tokens":11,"output_tokens":3}}`
		}
		responseBody := &systemOneTrackedBody{reader: strings.NewReader(reply)}
		bodies = append(bodies, responseBody)
		return &http.Response{StatusCode: http.StatusOK, Body: responseBody, Header: http.Header{}, Request: r}, nil
	})}
	decider := NewSystemOne("jev", "https://classifier.invalid/v1/systemone", "jev-latest", "test-key", client)
	policy := NewChain([]Step{{Decider: decider, Timeout: time.Second}}, 0.1, DoubtPair, efforts).Policy(three)
	verdict, err := decider.Decide(t.Context(), state, three, efforts)
	if err != nil || verdict.Choice != "glm-5.3" || verdict.Levels["high"] != 1 ||
		verdict.Model != "jev-test" || verdict.InputTokens != 17 || verdict.OutputTokens != 5 || !verdict.UsageKnown {
		t.Fatalf("choice verdict = %+v, error %v", verdict, err)
	}
	rebuilt, ok := policy.Request("jev", state)
	if !ok || !bytes.Equal(rebuilt, sent[0]) || verdict.Digest != Digest(sent[0]) {
		t.Fatal("choice request and digest do not match the recorded policy")
	}
	var request systemOneRequest
	if err := json.Unmarshal(sent[0], &request); err != nil || request.Model != "jev-latest" || request.State != state ||
		len(request.Questions) != 2 || request.Questions[modelQuestionID].Type != "choice" ||
		request.Questions[reasoningQuestionID].Type != "score" {
		t.Fatalf("choice request = %+v, error %v", request, err)
	}

	pair, err := decider.Pair(t.Context(), state, three[1], three[2])
	if err != nil || pair.Choice != "glm-5.3" || pair.InputTokens != 11 || !pair.UsageKnown {
		t.Fatalf("pair verdict = %+v, error %v", pair, err)
	}
	rebuilt, ok = policy.PairRequest("jev", state, three[1].ID, three[2].ID)
	if !ok || !bytes.Equal(rebuilt, sent[1]) || pair.Digest != Digest(sent[1]) {
		t.Fatal("pair request and digest do not match the recorded policy")
	}
	request = systemOneRequest{}
	if err := json.Unmarshal(sent[1], &request); err != nil || len(request.Questions) != 1 {
		t.Fatalf("pair request = %+v, error %v", request, err)
	}
	framing := request.Questions[pairQuestionID].Instructions
	if !strings.HasPrefix(framing.Focus, policy.Instructions.Focus) || framing.Probabilities != policy.Instructions.Probabilities ||
		!strings.Contains(framing.Question, three[1].ID) || !strings.Contains(framing.Question, three[2].ID) {
		t.Fatal("pair request lost routing or probability instructions")
	}
	for _, body := range bodies {
		if !body.closed {
			t.Error("response body was not closed")
		}
	}
}

func TestSystemOneContractOptionalAnswersDoNotPoisonChoice(t *testing.T) {
	cases := []struct {
		name, reasoning string
	}{
		{"missing", ""},
		{"null", `null`},
		{"wrong answer type", `"private-marker"`},
		{"wrong score type", `{"score":"private-marker"}`},
		{"wrong probability type", `{"probabilities":{"2":"private-marker"}}`},
		{"wrong probability container", `{"probabilities":[1]}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			answers := `"model":{"choice":"glm-5.3","probabilities":{"glm-5.3":1}},"unasked":["private-marker"]`
			if c.reasoning != "" {
				answers += `,"reasoning":` + c.reasoning
			}
			reply := `{"answers":{` + answers + `},"usage":{"input_tokens":7,"output_tokens":2}}`
			client, body := systemOneFixture(http.StatusOK, reply)
			verdict, err := NewSystemOne("jev", "https://classifier.invalid", "jev", "test-key", client).
				Decide(t.Context(), "state", three, efforts)
			if err != nil || verdict.Choice != "glm-5.3" || verdict.Levels != nil || !verdict.UsageKnown || !body.closed {
				t.Fatalf("optional answer rejected choice: %+v, error %v, closed %v", verdict, err, body.closed)
			}
		})
	}
}

func TestSystemOneContractRequiredAnswerFailuresKeepMetadata(t *testing.T) {
	cases := []struct {
		name, answer string
	}{
		{"missing", ""},
		{"null", `null`},
		{"empty", `{}`},
		{"wrong answer type", `"private-marker"`},
		{"wrong choice type", `{"choice":["private-marker"]}`},
		{"wrong probability type", `{"probabilities":{"glm-5.3":"private-marker"}}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			answers := ""
			if c.answer != "" {
				answers = `"model":` + c.answer
			}
			reply := `{"model":"jev-test","answers":{` + answers + `},"usage":{"input_tokens":19,"output_tokens":4}}`
			client, body := systemOneFixture(http.StatusOK, reply)
			verdict, err := NewSystemOne("jev", "https://classifier.invalid", "jev", "test-key", client).
				Decide(t.Context(), "state", three, efforts)
			if err == nil || strings.Contains(err.Error(), "private-marker") || verdict.Model != "jev-test" ||
				verdict.InputTokens != 19 || verdict.OutputTokens != 4 || !verdict.UsageKnown ||
				verdict.Response != reply || verdict.Digest == "" || !body.closed {
				t.Fatalf("failure metadata = %+v, error %v, closed %v", verdict, err, body.closed)
			}
		})
	}
}

func TestSystemOneContractNumericLevelIDsAndIndexPrecedence(t *testing.T) {
	cases := []struct {
		name   string
		levels []Level
		key    string
		want   string
	}{
		{"token budget id", []Level{{ID: "1024"}, {ID: "4096"}}, "4096", "4096"},
		{"token budget index", []Level{{ID: "1024"}, {ID: "4096"}}, "0", "1024"},
		{"index takes precedence", []Level{{ID: "1"}, {ID: "4096"}}, "1", "4096"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			reply := `{"answers":{"model":{"choice":"glm-5.3"},"reasoning":{"probabilities":{"` + c.key + `":1}}}}`
			client, _ := systemOneFixture(http.StatusOK, reply)
			verdict, err := NewSystemOne("jev", "https://classifier.invalid", "jev", "test-key", client).
				Decide(t.Context(), "state", three, c.levels)
			if err != nil || len(verdict.Levels) != 1 || verdict.Levels[c.want] != 1 {
				t.Fatalf("numeric level mapping = %v, error %v", verdict.Levels, err)
			}
		})
	}
}

func TestSystemOneContractUsagePresenceAndPartialCounts(t *testing.T) {
	cases := []struct {
		name, raw     string
		input, output int64
		known         bool
	}{
		{"missing", "", 0, 0, false},
		{"null", `null`, 0, 0, false},
		{"empty", `{}`, 0, 0, false},
		{"wrong container", `"private-marker"`, 0, 0, false},
		{"reported zero", `{"input_tokens":0,"output_tokens":0}`, 0, 0, true},
		{"reported both", `{"input_tokens":13,"output_tokens":4}`, 13, 4, true},
		{"partial", `{"input_tokens":13}`, 13, 0, false},
		{"negative", `{"input_tokens":13,"output_tokens":-1}`, 13, 0, false},
		{"wrong type", `{"input_tokens":"private-marker","output_tokens":4}`, 0, 4, false},
		{"fractional", `{"input_tokens":1.5,"output_tokens":4}`, 0, 4, false},
		{"null counter", `{"input_tokens":null,"output_tokens":4}`, 0, 4, false},
		{"overflow", `{"input_tokens":9223372036854775808,"output_tokens":4}`, 0, 4, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			reply := `{"answers":{"model":{"choice":"glm-5.3"}}`
			if c.raw != "" {
				reply += `,"usage":` + c.raw
			}
			reply += `}`
			client, body := systemOneFixture(http.StatusOK, reply)
			verdict, err := NewSystemOne("jev", "https://classifier.invalid", "jev", "test-key", client).
				Decide(t.Context(), "state", three, nil)
			if err != nil || verdict.Choice != "glm-5.3" || !body.closed ||
				verdict.InputTokens != c.input || verdict.OutputTokens != c.output || verdict.UsageKnown != c.known {
				t.Fatalf("usage verdict = %+v, error %v; want %d, %d, %v", verdict, err, c.input, c.output, c.known)
			}
		})
	}
}

func TestSystemOneContractResponseBoundsAndPrivacy(t *testing.T) {
	const valid = `{"answers":{"model":{"choice":"glm-5.3"}},"usage":{"input_tokens":5,"output_tokens":2}}`
	cases := []struct {
		name   string
		status int
		reply  string
		wantOK bool
		known  bool
	}{
		{"exact bound", http.StatusOK, valid + strings.Repeat(" ", maxDeciderBody-len(valid)), true, true},
		{"one byte over", http.StatusOK, valid + strings.Repeat(" ", maxDeciderBody+1-len(valid)), false, false},
		{"large status body", http.StatusRequestEntityTooLarge, strings.Repeat("private-marker", maxDeciderBody), false, false},
		{"status body", http.StatusBadRequest, `{"error":"prompt-marker key-marker","usage":{"input_tokens":5,"output_tokens":2}}`, false, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			client, body := systemOneFixture(c.status, c.reply)
			verdict, err := NewSystemOne("jev", "https://classifier.invalid?secret=query-marker", "jev", "key-marker", client).
				Decide(t.Context(), "prompt-marker", three, nil)
			if (err == nil) != c.wantOK || verdict.UsageKnown != c.known || !body.closed || body.read > maxDeciderBody+1 {
				t.Fatalf("bounded response: error %v, known %v, closed %v, read %d", err, verdict.UsageKnown, body.closed, body.read)
			}
			if c.wantOK && len(verdict.Response) != maxResponse {
				t.Fatalf("retained response has %d bytes", len(verdict.Response))
			}
			if !c.wantOK {
				if verdict.Response != "" {
					t.Fatal("non-200 or oversized body retained as a response")
				}
				for _, marker := range []string{"private-marker", "prompt-marker", "key-marker", "query-marker", "classifier.invalid"} {
					if strings.Contains(err.Error(), marker) {
						t.Fatalf("classifier error exposed %s", marker)
					}
				}
				if c.status != http.StatusOK && !strings.Contains(err.Error(), fmt.Sprintf("status %d", c.status)) {
					t.Fatal("classifier error lost HTTP status")
				}
			}
		})
	}
}

func TestSystemOneContractRequestAndDecodeErrorsArePrivate(t *testing.T) {
	cases := []struct {
		name, url, reply string
		transportError   error
	}{
		{"invalid URL", "://classifier.invalid?secret=query-marker", "", nil},
		{"transport", "https://classifier.invalid?secret=query-marker", "", errors.New("prompt-marker key-marker query-marker")},
		{"decode", "https://classifier.invalid?secret=query-marker", `{"answers":"prompt-marker key-marker"}`, nil},
		{"trailing data", "https://classifier.invalid?secret=query-marker", `{"answers":{"model":{"choice":"glm-5.3"}}} prompt-marker key-marker`, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			client, body := systemOneFixture(http.StatusOK, c.reply)
			if c.transportError != nil {
				client.Transport = systemOneRoundTrip(func(*http.Request) (*http.Response, error) {
					return nil, c.transportError
				})
			}
			verdict, err := NewSystemOne("jev", c.url, "jev", "key-marker", client).Decide(t.Context(), "prompt-marker", three, nil)
			if err == nil || verdict.Digest == "" {
				t.Fatalf("request failure: error %v, digest %q", err, verdict.Digest)
			}
			for _, marker := range []string{"prompt-marker", "key-marker", "query-marker", "classifier.invalid"} {
				if strings.Contains(err.Error(), marker) {
					t.Fatalf("classifier error exposed %s", marker)
				}
			}
			if c.reply != "" && (!body.closed || verdict.Response != c.reply) {
				t.Fatal("decode failure lost its bounded response or left its body open")
			}
		})
	}
}

func TestSystemOneContractCancellationIdentity(t *testing.T) {
	cases := []struct {
		name string
		ctx  func(*testing.T) context.Context
		want error
	}{
		{"canceled", func(t *testing.T) context.Context {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			return ctx
		}, context.Canceled},
		{"deadline", func(t *testing.T) context.Context {
			ctx, cancel := context.WithDeadline(t.Context(), time.Now().Add(-time.Second))
			t.Cleanup(cancel)
			return ctx
		}, context.DeadlineExceeded},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			called := false
			client := &http.Client{Transport: systemOneRoundTrip(func(r *http.Request) (*http.Response, error) {
				called = true
				<-r.Context().Done()
				return nil, fmt.Errorf("query-marker key-marker: %w", r.Context().Err())
			})}
			verdict, err := NewSystemOne("jev", "https://classifier.invalid?secret=query-marker", "jev", "key-marker", client).
				Decide(c.ctx(t), "prompt-marker", three, nil)
			if !called || !errors.Is(err, c.want) || verdict.Digest == "" ||
				strings.Contains(err.Error(), "marker") || strings.Contains(err.Error(), "classifier.invalid") {
				t.Fatalf("cancellation = %v, called %v, digest %q", err, called, verdict.Digest)
			}
		})
	}
}

func TestSystemOneContractCanceledSuccessClosesBody(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	body := &systemOneTrackedBody{reader: strings.NewReader(`{"answers":{"model":{"choice":"glm-5.3"}}}`)}
	client := &http.Client{Transport: systemOneRoundTrip(func(r *http.Request) (*http.Response, error) {
		cancel()
		return &http.Response{StatusCode: http.StatusOK, Body: body, Header: http.Header{}, Request: r}, nil
	})}
	verdict, err := NewSystemOne("jev", "https://classifier.invalid", "jev", "test-key", client).Decide(ctx, "state", three, nil)
	if !errors.Is(err, context.Canceled) || !body.closed || body.read != 0 || verdict.Choice != "" {
		t.Fatalf("canceled success: error %v, closed %v, read %d, choice %q", err, body.closed, body.read, verdict.Choice)
	}
}

func TestSystemOneContractReadErrorsCloseBodyAndStayPrivate(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want error
	}{
		{"read failure", errors.New("prompt-marker key-marker query-marker"), nil},
		{"cancellation", fmt.Errorf("key-marker: %w", context.Canceled), context.Canceled},
		{"deadline", fmt.Errorf("query-marker: %w", context.DeadlineExceeded), context.DeadlineExceeded},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body := &systemOneTrackedBody{reader: systemOneReadFunc(func([]byte) (int, error) {
				return 0, c.err
			})}
			client := &http.Client{Transport: systemOneRoundTrip(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Body: body, Header: http.Header{}, Request: r}, nil
			})}
			verdict, err := NewSystemOne("jev", "https://classifier.invalid?secret=query-marker", "jev", "key-marker", client).
				Decide(t.Context(), "prompt-marker", three, nil)
			if err == nil || c.want != nil && !errors.Is(err, c.want) || !body.closed ||
				verdict.Digest == "" || strings.Contains(err.Error(), "marker") || strings.Contains(err.Error(), "classifier.invalid") {
				t.Fatalf("read failure: error %v, closed %v, digest %q", err, body.closed, verdict.Digest)
			}
		})
	}
}
