package decide

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Judge uses any chat model served by CLIProxy (gpt-6-luna, deepseek-flash,
// a Claude model...) as the decider. It asks for a probability per
// candidate, and per reasoning level with the chooser on, as a JSON object.
type Judge struct {
	name, url, model, key string
	client                *http.Client
	framing               *Instructions
}

// NewJudge returns an LLM judge posting to a chat completions URL.
func NewJudge(name, url, model, key string, client *http.Client) *Judge {
	return &Judge{name: name, url: url, model: model, key: key, client: client}
}

// Name is the configured decider name.
func (j *Judge) Name() string {
	return j.name
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

const judgeFormat = ` Reply with only a JSON object of the form ` +
	`{"probabilities": {"<model id>": <number>}, "reason": "<one short sentence>"}, giving for every model ` +
	`the probability that it is the right choice (the numbers sum to 1) and the deciding factor for the ` +
	`most likely model, without quoting the request. Use exactly the offered model IDs. Every probability ` +
	`must be a finite JSON number between 0 and 1; the map must sum to 1.`

// judgeReasoningFormat is the reply format when the judge also answers the
// reasoning question.
const judgeReasoningFormat = ` Reply with only a JSON object of the form ` +
	`{"probabilities": {"<model id>": <number>}, "reasoning": {"<level id>": <number>}, ` +
	`"reason": "<one short sentence>"}, giving for every model the probability that it is the right choice, ` +
	`for every reasoning level the probability that it is the minimum sufficient level (each set of numbers sums to 1), ` +
	`and the deciding factor for the most likely model, without quoting the request. Use exactly the offered model ` +
	`and reasoning level IDs. Every probability must be a finite JSON number between 0 and 1.`

// maxReason bounds a judge's stored reason, in runes.
const maxReason = 300

// Info names the decider for the policy record.
func (j *Judge) Info() DeciderInfo {
	return DeciderInfo{Name: j.name, Kind: KindLLM, Model: j.model}
}

// Decide asks the judge model to choose among the candidates and, with
// levels, how much reasoning the request needs.
func (j *Judge) Decide(ctx context.Context, state string, candidates []Candidate, levels []Level) (Verdict, error) {
	reasoning := reasoningQuestion(levels)
	return j.ask(ctx, judgeBody(j.model, judgeSystem(j.instructions(), judgeFormat, reasoning), state, candidates, reasoning), candidates)
}

// Pair asks the judge model which of two ladder steps should do the
// request.
func (j *Judge) Pair(ctx context.Context, state string, cheap, strong Candidate) (Verdict, error) {
	framing := pairFraming(pairQuestion, j.instructions(), cheap.ID, strong.ID)
	pair := []Candidate{cheap, strong}
	return j.ask(ctx, judgeBody(j.model, judgeSystem(framing, judgeFormat, nil), state, pair, nil), pair)
}

// ask posts a chat completion and parses the judgement of the candidates in
// its reply.
func (j *Judge) ask(ctx context.Context, payload []byte, candidates []Candidate) (Verdict, error) {
	var response struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage json.RawMessage `json:"usage"`
	}
	sent := Verdict{Digest: Digest(payload)}
	raw, err := post(ctx, j.client, j.url, j.key, payload, &response)
	sent.Model = cmp.Or(response.Model, j.model)
	sent.Response = excerpt(raw)
	sent.InputTokens, sent.OutputTokens, sent.UsageKnown = reportedUsage(response.Usage, "prompt_tokens", "completion_tokens")
	if err != nil {
		return sent, err
	}
	if len(response.Choices) == 0 {
		return sent, fmt.Errorf("response has no choices")
	}
	reply := messageText(response.Choices[0].Message.Content)
	sent.Response = excerpt([]byte(reply))
	verdict, err := parseJudgement(reply, candidates)
	if err != nil {
		return sent, err
	}
	verdict.Digest = sent.Digest
	verdict.Response = sent.Response
	verdict.Model = sent.Model
	verdict.InputTokens, verdict.OutputTokens = sent.InputTokens, sent.OutputTokens
	verdict.UsageKnown = sent.UsageKnown
	return verdict, nil
}

// messageText flattens a chat message content that is a string or a list
// of text parts.
func messageText(raw json.RawMessage) string {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var parts []struct {
		Text string `json:"text"`
	}
	_ = json.Unmarshal(raw, &parts)
	var joined strings.Builder
	for _, p := range parts {
		joined.WriteString(p.Text)
	}
	return joined.String()
}

// thinking matches the reasoning blocks some models put before their answer.
var thinking = regexp.MustCompile(`(?s)<(think|thinking)>.*?</(think|thinking)>`)

// maxObjects bounds the braces a reply is searched from for its judgement.
const maxObjects = 64

// parseJudgement reads the first JSON object in the reply that judges the
// candidates, past any reasoning block, stray braces and objects naming no
// candidate: either {"probabilities": {...}, "reasoning": {...}, "choice":
// "..."} or a bare {"id": p} map. Probabilities may be numbers, numeric
// strings or percentages; the reasoning answer may also name one level.
func parseJudgement(reply string, candidates []Candidate) (Verdict, error) {
	reply = thinking.ReplaceAllString(reply, "")
	found := false
	for offset, tried := 0, 0; tried < maxObjects; tried++ {
		start := strings.IndexByte(reply[offset:], '{')
		if start < 0 {
			break
		}
		start += offset
		offset = start + 1
		var object map[string]json.RawMessage
		if json.NewDecoder(strings.NewReader(reply[start:])).Decode(&object) != nil {
			continue
		}
		found = true
		if verdict, ok := judgement(object, candidates); ok {
			return verdict, nil
		}
	}
	if found {
		return Verdict{}, fmt.Errorf("reply judges none of the candidates")
	}
	return Verdict{}, fmt.Errorf("reply has no JSON object")
}

// judgement reads a verdict from a reply object; ok is false when it puts
// no probability on a candidate and chooses none.
func judgement(object map[string]json.RawMessage, candidates []Candidate) (Verdict, bool) {
	var verdict Verdict
	_, structured := object["probabilities"]
	_ = json.Unmarshal(object["choice"], &verdict.Choice)
	if structured || verdict.Choice != "" {
		verdict.Probabilities = probabilities(object["probabilities"])
		_ = json.Unmarshal(object["reason"], &verdict.Reason)
		verdict.Reason = oneLine(verdict.Reason, maxReason)
		verdict.Levels = judgedLevels(object["reasoning"])
	} else {
		verdict.Probabilities = probabilities(json.RawMessage(nil))
		for id, raw := range object {
			if p, ok := number(raw); ok {
				verdict.Probabilities[id] = p
			}
		}
	}
	judged := slices.ContainsFunc(candidates, func(c Candidate) bool {
		_, weighed := verdict.Probabilities[c.ID]
		return weighed || c.ID == verdict.Choice
	})
	return verdict, judged
}

// probabilities reads a {"id": p} object, skipping values that are not
// numbers; empty for anything else.
func probabilities(raw json.RawMessage) map[string]float64 {
	result := map[string]float64{}
	var values map[string]json.RawMessage
	if json.Unmarshal(raw, &values) != nil {
		return result
	}
	for id, value := range values {
		if p, ok := number(value); ok {
			result[id] = p
		}
	}
	return result
}

// number reads a probability written as a number, a numeric string or a
// percentage ("80%").
func number(raw json.RawMessage) (float64, bool) {
	var p float64
	if json.Unmarshal(raw, &p) == nil {
		return p, true
	}
	var text string
	if json.Unmarshal(raw, &text) != nil {
		return 0, false
	}
	text = strings.TrimSpace(text)
	percent := strings.HasSuffix(text, "%")
	p, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimSuffix(text, "%")), 64)
	if err != nil {
		return 0, false
	}
	if percent {
		p /= 100
	}
	return p, true
}

// judgedLevels reads a judge's reasoning answer: a probability per level,
// or one level's id, whatever their case. It is nil when the judge gave
// neither.
func judgedLevels(raw json.RawMessage) map[string]float64 {
	levels := map[string]float64{}
	for id, p := range probabilities(raw) {
		levels[strings.ToLower(strings.TrimSpace(id))] += p
	}
	var level string
	if json.Unmarshal(raw, &level) == nil {
		if level = strings.ToLower(strings.TrimSpace(level)); level != "" {
			levels[level] = 1
		}
	}
	if len(levels) == 0 {
		return nil
	}
	return levels
}

// oneLine collapses whitespace and cuts text to at most limit runes.
func oneLine(text string, limit int) string {
	text = strings.Join(strings.Fields(text), " ")
	if utf8.RuneCountInString(text) > limit {
		text = string([]rune(text)[:limit-1]) + "…"
	}
	return text
}
