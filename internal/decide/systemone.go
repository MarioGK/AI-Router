package decide

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"strconv"
)

// SystemOne asks a decision model speaking TypeSafe's /v1/systemone API a
// choice question over the candidates and, with the reasoning chooser on, a
// score question over the reasoning levels in the same request. The
// configured endpoint must support this request and response schema.
type SystemOne struct {
	name, url, model, key string
	client                *http.Client
	framing               *Instructions
}

// NewSystemOne returns a systemone decider.
func NewSystemOne(name, url, model, key string, client *http.Client) *SystemOne {
	return &SystemOne{name: name, url: url, model: model, key: key, client: client}
}

// Name is the configured decider name.
func (s *SystemOne) Name() string {
	return s.name
}

// Question ids: the choice over every candidate, the reasoning score, and
// the pair question that settles a doubt.
const (
	modelQuestionID     = "model"
	reasoningQuestionID = "reasoning"
	pairQuestionID      = "pair"
)

type systemOneRequest struct {
	Model     string                       `json:"model"`
	State     string                       `json:"state"`
	Questions map[string]systemOneQuestion `json:"questions"`
}

// systemOneQuestion is a choice question, whose criteria map each option to
// its criterion, or a score question, whose criteria list the levels
// lightest first.
type systemOneQuestion struct {
	Type         string       `json:"type"`
	Instructions Instructions `json:"instructions"`
	Criteria     any          `json:"criteria"`
}

// systemOneAnswer is the answer to a choice question, or to a score
// question, whose probabilities are keyed by level index.
type systemOneAnswer struct {
	Choice        string             `json:"choice"`
	Score         *float64           `json:"score"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
}

type systemOneResponse struct {
	Model   string                     `json:"model"`
	Answers map[string]json.RawMessage `json:"answers"`
	Usage   json.RawMessage            `json:"usage"`
}

// Info names the decider for the policy record.
func (s *SystemOne) Info() DeciderInfo {
	return DeciderInfo{Name: s.name, Kind: KindSystemOne, Model: s.model}
}

// Decide asks the choice question and, with levels, the reasoning score.
func (s *SystemOne) Decide(ctx context.Context, state string, candidates []Candidate, levels []Level) (Verdict, error) {
	reasoning := reasoningQuestion(levels)
	verdict, response, err := s.ask(ctx, modelQuestionID, systemOneBody(s.model, state, decisionQuestions(s.instructions(), candidates, reasoning)))
	if err == nil && reasoning != nil {
		var answer systemOneAnswer
		if json.Unmarshal(response.Answers[reasoningQuestionID], &answer) == nil {
			verdict.Levels = scoreLevels(answer, levels)
		}
	}
	return verdict, err
}

// Pair asks which of two ladder steps should do the request.
func (s *SystemOne) Pair(ctx context.Context, state string, cheap, strong Candidate) (Verdict, error) {
	framing := pairFraming(pairQuestion, s.instructions(), cheap.ID, strong.ID)
	verdict, _, err := s.ask(ctx, pairQuestionID, systemOneBody(s.model, state, pairQuestions(framing, []Candidate{cheap, strong})))
	return verdict, err
}

// ask posts a request and reads the answer to question id.
func (s *SystemOne) ask(ctx context.Context, id string, body []byte) (Verdict, systemOneResponse, error) {
	var response systemOneResponse
	sent := Verdict{Digest: Digest(body)}
	raw, err := post(ctx, s.client, s.url, s.key, body, &response)
	sent.Model, sent.Response = response.Model, excerpt(raw)
	sent.InputTokens, sent.OutputTokens, sent.UsageKnown = reportedUsage(response.Usage, "input_tokens", "output_tokens")
	if err != nil {
		return sent, response, err
	}
	rawAnswer, ok := response.Answers[id]
	if !ok {
		return sent, response, fmt.Errorf("response has no %q answer", id)
	}
	var answer systemOneAnswer
	if json.Unmarshal(rawAnswer, &answer) != nil || answer.Choice == "" && len(answer.Probabilities) == 0 {
		return sent, response, fmt.Errorf("response has invalid %q answer", id)
	}
	sent.Choice, sent.Probabilities, sent.Confidence = answer.Choice, answer.Probabilities, answer.Confidence
	return sent, response, nil
}

// reportedUsage preserves each valid token count, but marks usage known
// only when both counters are reported as nonnegative integers.
func reportedUsage(raw json.RawMessage, inputKey, outputKey string) (input, output int64, known bool) {
	var usage map[string]json.RawMessage
	if json.Unmarshal(raw, &usage) != nil {
		return 0, 0, false
	}
	count := func(raw json.RawMessage) (int64, bool) {
		var tokens *int64
		if json.Unmarshal(raw, &tokens) != nil || tokens == nil || *tokens < 0 {
			return 0, false
		}
		return *tokens, true
	}
	input, inputOK := count(usage[inputKey])
	output, outputOK := count(usage[outputKey])
	return input, output, inputOK && outputOK
}

// scoreLevels maps a score answer to a probability per level: its
// probabilities, keyed by level index or level id, or, when they hold
// nothing, all on the level its score rounds to. Canonical indexes in range
// take precedence over numeric level ids. It is nil for a missing answer.
func scoreLevels(answer systemOneAnswer, levels []Level) map[string]float64 {
	if len(levels) == 0 {
		return nil
	}
	probabilities, total := map[string]float64{}, 0.0
	for key, p := range answer.Probabilities {
		i, err := strconv.Atoi(key)
		if err != nil || strconv.Itoa(i) != key || i < 0 || i >= len(levels) {
			i = slices.IndexFunc(levels, func(l Level) bool { return l.ID == key })
		}
		if i >= 0 && i < len(levels) && p > 0 && !math.IsInf(p, 0) {
			probabilities[levels[i].ID] += p
			total += p
		}
	}
	if total > 0 {
		return probabilities
	}
	if answer.Score == nil || math.IsNaN(*answer.Score) {
		return nil
	}
	// Clamp before converting: a huge score must not overflow the index.
	score := min(max(*answer.Score+0.5, 0), float64(len(levels)-1))
	return map[string]float64{levels[int(score)].ID: 1}
}

// maxDeciderBody bounds the raw response read from a classifier.
const maxDeciderBody = 1 << 20

// post sends a JSON request with a Bearer key, decodes the JSON response
// into out and returns its bounded raw body for HTTP 200 responses,
// including on decode errors. Errors expose only fixed categories, HTTP
// statuses and context cancellation; provider bodies and URL-bearing
// transport errors can contain secrets.
func post(ctx context.Context, client *http.Client, url, key string, payload []byte, out any) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, deciderRequestError("create decider request", err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+key)
	response, err := client.Do(request)
	if err != nil {
		return nil, deciderRequestError("send decider request", err)
	}
	defer response.Body.Close()
	if err := ctx.Err(); err != nil {
		return nil, deciderRequestError("send decider request", err)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxDeciderBody+1))
	if len(body) > maxDeciderBody {
		return nil, fmt.Errorf("status %d: decider response exceeds %d bytes", response.StatusCode, maxDeciderBody)
	}
	if err != nil {
		_ = json.Unmarshal(body, out)
		if response.StatusCode != http.StatusOK {
			body = nil
		}
		return body, deciderRequestError("read decider response", err)
	}
	if err := ctx.Err(); err != nil {
		_ = json.Unmarshal(body, out)
		if response.StatusCode != http.StatusOK {
			body = nil
		}
		return body, deciderRequestError("read decider response", err)
	}
	if response.StatusCode != http.StatusOK {
		_ = json.Unmarshal(body, out)
		return nil, fmt.Errorf("status %d", response.StatusCode)
	}
	if err := json.Unmarshal(body, out); err != nil {
		return body, deciderRequestError("decode decider response", err)
	}
	return body, nil
}

// deciderRequestError keeps cancellation identity without retaining an
// arbitrary provider, request URL or transport error message.
func deciderRequestError(operation string, err error) error {
	switch {
	case errors.Is(err, context.Canceled):
		return fmt.Errorf("%s: %w", operation, context.Canceled)
	case errors.Is(err, context.DeadlineExceeded):
		return fmt.Errorf("%s: %w", operation, context.DeadlineExceeded)
	default:
		return fmt.Errorf("%s failed", operation)
	}
}
