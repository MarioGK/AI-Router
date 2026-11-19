package app

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/usage"
)

const proxyKey = "proxy-key-for-tests"

// upstream is a fake CLIProxy that records every request.
type upstream struct {
	t        *testing.T
	server   *httptest.Server
	mu       sync.Mutex
	requests []recorded
	// respond answers a routed request; nil means 200 with a small body.
	respond func(w http.ResponseWriter, r recorded)
}

type recorded struct {
	Method, Path, Query string
	Header              http.Header
	Body                []byte
	Model               string
}

func newUpstream(t *testing.T) *upstream {
	u := &upstream{t: t}
	u.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var fields struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &fields)
		rec := recorded{Method: r.Method, Path: r.URL.Path, Query: r.URL.RawQuery, Header: r.Header.Clone(), Body: body, Model: fields.Model}
		if r.URL.Path == "/v1/models" {
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"deepseek-flash","object":"model"},{"id":"glm-5.3","object":"model"},` +
				`{"id":"glm-5.3-flash","object":"model"},{"id":"gpt-6.1-sol","object":"model"},{"id":"gpt-6-luna","object":"model"},` +
				`{"id":"claude-opus-5-5","object":"model"}]}`))
			return
		}
		u.mu.Lock()
		u.requests = append(u.requests, rec)
		respond := u.respond
		u.mu.Unlock()
		if respond != nil {
			respond(w, rec)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"model":%q,"ok":true}`, rec.Model)
	}))
	t.Cleanup(u.server.Close)
	return u
}

func (u *upstream) routed() []recorded {
	u.mu.Lock()
	defer u.mu.Unlock()
	return append([]recorded(nil), u.requests...)
}

// jev is a fake systemone endpoint.
type jev struct {
	server *httptest.Server
	calls  atomic.Int32
	mu     sync.Mutex
	status int
	answer string
	// pair answers pair questions, in the answer's "id=p,id=p" form.
	pair string
	// reasoning answers the reasoning score, as its probabilities by level
	// index ({"2": 1}); empty leaves it unanswered.
	reasoning string
	states    []string
	keys      [][]string
	bodies    [][]byte
}

func newJev(t *testing.T) *jev {
	j := &jev{status: http.StatusOK, answer: "glm-5.3"}
	j.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		j.calls.Add(1)
		var request struct {
			State     string `json:"state"`
			Questions map[string]struct {
				Criteria json.RawMessage `json:"criteria"`
			} `json:"questions"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &request)
		time.Sleep(20 * time.Millisecond)
		j.mu.Lock()
		defer j.mu.Unlock()
		j.states = append(j.states, request.State)
		j.bodies = append(j.bodies, body)
		var criteria map[string]json.RawMessage
		_ = json.Unmarshal(request.Questions["model"].Criteria, &criteria)
		var keys []string
		for id := range criteria {
			keys = append(keys, id)
		}
		j.keys = append(j.keys, keys)
		if j.status != http.StatusOK {
			w.WriteHeader(j.status)
			return
		}
		question, spec := "model", j.answer
		if _, ok := request.Questions["pair"]; ok {
			question, spec = "pair", j.pair
		}
		answer := spec
		probabilities := map[string]float64{answer: 1}
		if strings.Contains(spec, "=") {
			// "choice=p,other=p" form for escalation tests.
			probabilities = map[string]float64{}
			for i, pair := range strings.Split(spec, ",") {
				id, p, _ := strings.Cut(pair, "=")
				var value float64
				_, _ = fmt.Sscan(p, &value)
				probabilities[id] = value
				if i == 0 {
					answer = id
				}
			}
		}
		answers := map[string]any{question: map[string]any{
			"type": "choice", "choice": answer, "probabilities": probabilities, "confidence": 0.9,
		}}
		if _, asked := request.Questions["reasoning"]; asked && j.reasoning != "" {
			answers["reasoning"] = map[string]any{"type": "score", "probabilities": json.RawMessage(j.reasoning)}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"answers": answers})
	}))
	t.Cleanup(j.server.Close)
	return j
}

type harness struct {
	t        *testing.T
	cpa      *upstream
	jev      *jev
	app      *App
	server   *httptest.Server
	logs     *syncBuffer
	database string
	yaml     string
}

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

func newHarness(t *testing.T, extra string) *harness {
	t.Helper()
	t.Setenv("CLIPROXYAPI_API_KEY", proxyKey)
	t.Setenv("JEV_API_KEY", "jev-key")
	h := &harness{t: t, cpa: newUpstream(t), jev: newJev(t), logs: &syncBuffer{}, database: filepath.Join(t.TempDir(), "router.db")}
	h.yaml = fmt.Sprintf(`
database: %s
upstream: {base_url: %q}
auto:
  default_model: glm-5.3
  aux_model: glm-5.3-flash
  candidates:
    - {id: deepseek-flash, context: 1000000, output: 64000, vision: true, description: "quick questions and small edits"}
    - {id: glm-5.3, context: 200000, output: 128000, description: "everyday feature work"}
    - {id: gpt-6.1-sol, context: 400000, output: 128000, vision: true, description: "hard problems"}
  fallback: {gpt-6.1-sol: glm-5.3, glm-5.3: deepseek-flash, deepseek-flash: glm-5.3}
%s
deciders:
  - {name: jev, type: systemone, url: %q, model: jev-latest, api_key_env: JEV_API_KEY, timeout: 2s}
  - {name: judge, type: llm, model: gpt-6-luna, timeout: 2s}
`, h.database, h.cpa.server.URL, extra, h.jev.server.URL+"/v1/systemone")
	h.start()
	return h
}

func (h *harness) start() {
	cfg, err := config.Parse([]byte(h.yaml))
	if err != nil {
		h.t.Fatal(err)
	}
	h.app, err = New(cfg, slog.New(slog.NewJSONHandler(h.logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	if err != nil {
		h.t.Fatal(err)
	}
	if err := h.app.catalog.Refresh(h.t.Context()); err != nil {
		h.t.Fatal(err)
	}
	h.server = httptest.NewServer(h.app)
	h.t.Cleanup(func() {
		h.server.Close()
		h.app.Close()
	})
}

func (h *harness) restart() {
	h.server.Close()
	h.app.Close()
	h.start()
}

func (h *harness) do(method, path, body string, header ...string) *http.Response {
	h.t.Helper()
	request, err := http.NewRequest(method, h.server.URL+path, strings.NewReader(body))
	if err != nil {
		h.t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+proxyKey)
	request.Header.Set("Content-Type", "application/json")
	for i := 0; i < len(header); i += 2 {
		request.Header.Set(header[i], header[i+1])
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		h.t.Fatal(err)
	}
	h.t.Cleanup(func() { response.Body.Close() })
	return response
}

// claudeCode names the Claude Code version the tests send.
const claudeCode = "claude-cli/2.1.284"

// claude sends a Claude Code turn of session for the virtual model.
func (h *harness) claude(session, prompt string, header ...string) *http.Response {
	return h.do(http.MethodPost, "/v1/messages",
		fmt.Sprintf(`{"model":"auto","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":%q}],`+tools+`}`, prompt),
		append([]string{"User-Agent", claudeCode, "X-Claude-Code-Session-Id", session}, header...)...)
}

// screenshot is a Claude Code turn for the virtual model carrying an image.
// tools is what every turn of Claude Code's agent loop offers the model.
const tools = `"tools":[{"name":"Bash","description":"Run a shell command","input_schema":{"type":"object"}}]`

const screenshot = `{"model":"auto","max_tokens":1024,` + tools + `,"messages":[{"role":"user","content":[` +
	`{"type":"text","text":"what is on this screenshot"},` +
	`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]}]}`

// claudeImage sends session's screenshot turn.
func (h *harness) claudeImage(session string) *http.Response {
	return h.do(http.MethodPost, "/v1/messages", screenshot, "User-Agent", claudeCode, "X-Claude-Code-Session-Id", session)
}

// logged stops serving, writes the queued request log and returns it,
// oldest first.
func (h *harness) logged() []pin.RequestLog {
	h.t.Helper()
	h.server.Close()
	h.app.history.Close()
	requests, err := h.app.store.Requests(h.t.Context(), pin.RequestFilter{})
	if err != nil {
		h.t.Fatal(err)
	}
	slices.Reverse(requests)
	return requests
}

// anthropicError reads an error answer, which always has the Anthropic
// Messages API's shape, and returns its type and message.
func anthropicError(t *testing.T, response *http.Response) (kind, message string) {
	t.Helper()
	var body struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil || body.Type != "error" ||
		response.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("error answer %d %q = %+v, %v", response.StatusCode, response.Header.Get("Content-Type"), body, err)
	}
	return body.Error.Type, body.Error.Message
}

func readAll(t *testing.T, response *http.Response) string {
	t.Helper()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

func TestExplicitModelsPassThroughByteForByte(t *testing.T) {
	h := newHarness(t, "")
	body := "{ \"model\": \"claude-sonnet-5\",\n  \"messages\": [{\"role\":\"user\",\"content\":\"hi\"}], \"x_future_field\": {\"a\": [1,2]} }"
	response := h.do(http.MethodPost, "/v1/messages?beta=true", body,
		"User-Agent", "claude-cli/2.1.283 (external, cli)", "Anthropic-Beta", "context-management-2025-06-27",
		"X-Claude-Code-Session-Id", "cc-1", "Connection", "X-Drop-Me", "X-Drop-Me", "secret", "X-Custom", "kept")
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", response.StatusCode)
	}
	routed := h.cpa.routed()
	if len(routed) != 1 {
		t.Fatalf("routed %d requests", len(routed))
	}
	got := routed[0]
	if string(got.Body) != body || got.Path != "/v1/messages" || got.Query != "beta=true" {
		t.Fatalf("forwarded %s?%s %q", got.Path, got.Query, got.Body)
	}
	for name, want := range map[string]string{
		"User-Agent": "claude-cli/2.1.283 (external, cli)", "Anthropic-Beta": "context-management-2025-06-27",
		"X-Claude-Code-Session-Id": "cc-1", "Authorization": "Bearer " + proxyKey, "X-Custom": "kept", "X-Drop-Me": "",
	} {
		if got.Header.Get(name) != want {
			t.Errorf("header %s = %q, want %q", name, got.Header.Get(name), want)
		}
	}
	if h.jev.calls.Load() != 0 {
		t.Fatal("an explicit model called the decider")
	}
}

func TestAutoPinsAThreadOnItsFirstPromptOnly(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	first := readAll(t, h.claude("ses_1", "design the storage layer"))
	h.jev.answer = "deepseek-flash"
	second := readAll(t, h.claude("ses_1", "now rename a variable"))
	other := readAll(t, h.claude("ses_2", "what is a mutex?"))
	if !strings.Contains(first, `"gpt-6.1-sol"`) || !strings.Contains(second, `"gpt-6.1-sol"`) || !strings.Contains(other, `"deepseek-flash"`) {
		t.Fatalf("responses: %s | %s | %s", first, second, other)
	}
	if h.jev.calls.Load() != 2 {
		t.Fatalf("decider called %d times, want once per thread", h.jev.calls.Load())
	}
	routed := h.cpa.routed()
	want := `{"system":"` + modelNote("gpt-6.1-sol") + `","model":"gpt-6.1-sol","max_tokens":1024,"stream":true,` +
		`"messages":[{"role":"user","content":"design the storage layer"}],` + tools + `}`
	if string(routed[0].Body) != want {
		t.Fatalf("routed body = %s", routed[0].Body)
	}
	if state := h.jev.states[0]; state != "images=no\n\ndesign the storage layer" {
		t.Fatalf("decider state = %q", state)
	}

	h.restart()
	h.jev.answer = "deepseek-flash"
	if again := readAll(t, h.claude("ses_1", "one more")); !strings.Contains(again, `"gpt-6.1-sol"`) {
		t.Fatalf("pin did not survive a restart: %s", again)
	}
	if h.jev.calls.Load() != 2 {
		t.Fatal("a restarted router decided a pinned thread again")
	}
}

func TestEscalatesWhenStrongerModelsHoldTenPercent(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "deepseek-flash=0.88,glm-5.3=0.12"
	if body := readAll(t, h.claude("ses_esc", "add a CLI flag")); !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("not escalated: %s", body)
	}
	h.jev.answer = "glm-5.3=0.60,deepseek-flash=0.40"
	if body := readAll(t, h.claude("ses_down", "add a CLI flag")); !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("doubt towards a cheaper model changed the choice: %s", body)
	}
}

func TestDecisionsKeepTheirTrailAndInput(t *testing.T) {
	h := newHarness(t, "  doubt: pair\n  record_prompts: true")
	h.jev.answer = "deepseek-flash=0.80,glm-5.3=0.20"
	h.jev.pair = "deepseek-flash=0.66,glm-5.3=0.34"
	if body := readAll(t, h.claude("ses_pair", "rename a variable")); !strings.Contains(body, `"deepseek-flash"`) {
		t.Fatalf("the pair answer was not taken: %s", body)
	}
	threads, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || len(threads.Threads) != 1 {
		t.Fatalf("threads = %+v, %v", threads, err)
	}
	decision := threads.Threads[0].Decision
	if decision == nil || decision.Decider != "jev" || decision.Model != "deepseek-flash" || len(decision.Attempts) != 2 {
		t.Fatalf("decision = %+v", decision)
	}
	if jev := decision.Attempts[0]; jev.Decider != "jev" || jev.Outcome != decide.Doubted || jev.Choice != "deepseek-flash" ||
		math.Abs(jev.Above-0.20) > 1e-9 {
		t.Fatalf("jev attempt = %+v", jev)
	}
	if pair := decision.Attempts[1]; pair.Decider != "jev" || pair.Outcome != decide.Decided ||
		!slices.Equal(pair.Pair, []string{"deepseek-flash", "glm-5.3"}) || math.Abs(pair.Above-0.34) > 1e-9 {
		t.Fatalf("pair attempt = %+v", pair)
	}
	basis := decision.Basis
	if basis.Kind != "messages" || basis.UserAgent != claudeCode || len(basis.Candidates) == 0 || basis.Threshold != 0.1 ||
		basis.Doubt != "pair" || basis.PromptChars != len("rename a variable") || basis.Truncated || len(basis.Policy) != 12 {
		t.Fatalf("basis = %+v", basis)
	}
	if decision.Prompt != "images=no\n\nrename a variable" {
		t.Fatalf("prompt = %q", decision.Prompt)
	}
	detail, err := h.app.store.Thread(t.Context(), threads.Threads[0].Key)
	if err != nil || detail.Policy == nil || detail.Policy.Fingerprint() != basis.Policy {
		t.Fatalf("thread policy = %+v, %v", detail, err)
	}
	if sent, ok := detail.Policy.Request("jev", decision.Prompt); !ok || !bytes.Equal(sent, h.jev.bodies[0]) {
		t.Fatalf("rebuilt jev request\n%s\nsent\n%s", sent, h.jev.bodies[0])
	}
	if sent, ok := detail.Policy.PairRequest("jev", decision.Prompt, "deepseek-flash", "glm-5.3"); !ok || !bytes.Equal(sent, h.jev.bodies[1]) {
		t.Fatalf("rebuilt pair request\n%s\nsent\n%s", sent, h.jev.bodies[1])
	}
	if strings.Contains(h.logs.String(), "rename a variable") {
		t.Fatal("a recorded prompt reached the logs")
	}
}

func TestDeciderChainFallsBackToTheJudgeThenTheDefault(t *testing.T) {
	h := newHarness(t, "")
	h.jev.status = http.StatusServiceUnavailable
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Model == "gpt-6-luna" {
			_, _ = w.Write([]byte(`{"model":"gpt-6-luna","choices":[{"message":{"content":"{\"probabilities\":{\"gpt-6.1-sol\":0.95,\"glm-5.3\":0.05},\"reason\":\"A race.\"}"}}],"usage":{"prompt_tokens":400,"completion_tokens":30}}`))
			return
		}
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}
	if body := readAll(t, h.claude("ses_judge", "debug the race")); !strings.Contains(body, `"gpt-6.1-sol"`) {
		t.Fatalf("judge verdict not used: %s", body)
	}
	threads, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || len(threads.Threads) != 1 || len(threads.Threads[0].Decision.Attempts) != 2 {
		t.Fatalf("threads = %+v, %v", threads, err)
	}
	if judge := threads.Threads[0].Decision.Attempts[1]; judge.Decider != "judge" || judge.Reason != "A race." ||
		judge.Model != "gpt-6-luna" || judge.InputTokens != 400 || judge.OutputTokens != 30 {
		t.Fatalf("judge attempt = %+v", judge)
	}
	judged := h.cpa.routed()[0]
	if judged.Path != "/v1/chat/completions" || judged.Header.Get("Authorization") != "Bearer "+proxyKey {
		t.Fatalf("judge request = %+v", judged)
	}

	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Model == "gpt-6-luna" {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}
	if body := readAll(t, h.claude("ses_default", "anything")); !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("default not used: %s", body)
	}
	if logs := h.logs.String(); !strings.Contains(logs, `"decider":"default"`) || !strings.Contains(logs, "jev: status 503") {
		t.Fatalf("logs lack the failed chain: %s", logs)
	}
}

func TestFailoverSwitchesOnceAndRepins(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	var failing sync.Map
	failing.Store("gpt-6.1-sol", true)
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if _, down := failing.Load(r.Model); down {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":"quota"}`))
			return
		}
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}
	response := h.claude("ses_fo", "refactor everything")
	if body := readAll(t, response); response.StatusCode != http.StatusOK || !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("failover response %d %s", response.StatusCode, body)
	}
	failing.Delete("gpt-6.1-sol")
	if body := readAll(t, h.claude("ses_fo", "continue")); !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("thread was not re-pinned: %s", body)
	}
	failing.Store("glm-5.3", true)
	response = h.claude("ses_fo", "continue again")
	if body := readAll(t, response); response.StatusCode != http.StatusTooManyRequests || !strings.Contains(body, "quota") {
		t.Fatalf("second failover happened: %d %s", response.StatusCode, body)
	}
	models := []string{}
	for _, r := range h.cpa.routed() {
		models = append(models, r.Model)
	}
	if strings.Join(models, ",") != "gpt-6.1-sol,glm-5.3,glm-5.3,glm-5.3" {
		t.Fatalf("routed models = %v", models)
	}
}

func TestReasoningEffortRidesTheModelSuffix(t *testing.T) {
	h := newHarness(t, "")
	h.yaml = strings.Replace(h.yaml, "{id: gpt-6.1-sol, context: 400000,", "{id: gpt-6.1-sol, reasoning: xhigh, context: 400000,", 1)
	h.restart()
	h.jev.answer = "gpt-6.1-sol"
	readAll(t, h.claude("ses_effort", "hard problem"))
	readAll(t, h.do(http.MethodPost, "/v1/messages", `{"model":"gpt-6.1-sol","max_tokens":1024,"messages":[]}`))
	routed := h.cpa.routed()
	if routed[0].Model != "gpt-6.1-sol(xhigh)" || routed[1].Model != "gpt-6.1-sol" {
		t.Fatalf("routed models = %q, %q; want the suffix on auto only", routed[0].Model, routed[1].Model)
	}
	summary, err := h.app.store.Summarize(t.Context(), time.Now().Add(-time.Hour))
	if err != nil || summary.Pins["gpt-6.1-sol"] != 1 {
		t.Fatalf("pins = %+v, %v; the pin keeps the bare id", summary.Pins, err)
	}
}

func TestTheReasoningChooserPinsAnEffortPerThread(t *testing.T) {
	h := newHarness(t, `  failover_trip: 1
  reasoning:
    levels:
      - {id: low, description: mechanical}
      - {id: medium, description: routine}
      - {id: high, description: tricky}
      - {id: xhigh, description: deep}`)
	h.yaml = strings.NewReplacer(
		"{id: glm-5.3, context: 200000,", "{id: glm-5.3, efforts: [low, medium, high], context: 200000,",
		"{id: gpt-6.1-sol, context: 400000,", "{id: gpt-6.1-sol, reasoning: high, efforts: [medium, high, xhigh], context: 400000,",
	).Replace(h.yaml)
	h.restart()
	h.jev.answer, h.jev.reasoning = "gpt-6.1-sol", `{"2":0.4,"3":0.6}`
	var failing atomic.Bool
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if failing.Load() && strings.HasPrefix(r.Model, "gpt-6.1-sol") {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}
	response := h.claude("ses_deep", "untangle the sync protocol")
	if body := readAll(t, response); !strings.Contains(body, `"gpt-6.1-sol(xhigh)"`) ||
		response.Header.Get("X-Ts-Router-Effort") != "xhigh" {
		t.Fatalf("first response %s, effort %q", body, response.Header.Get("X-Ts-Router-Effort"))
	}
	// The level outlives a restart and follows the thread to its fallback,
	// at the nearest effort the fallback accepts.
	h.restart()
	failing.Store(true)
	if body := readAll(t, h.claude("ses_deep", "continue")); !strings.Contains(body, `"glm-5.3(high)"`) {
		t.Fatalf("the failover lost the thread's level: %s", body)
	}
	// Without a level, the fallback runs at its fixed effort: none.
	h.jev.reasoning = ""
	if body := readAll(t, h.claude("ses_plain", "anything")); !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("a thread without a level got an effort: %s", body)
	}
	page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	var pins []string
	for _, thread := range page.Threads {
		pins = append(pins, thread.Model+"@"+thread.Level)
	}
	slices.Sort(pins)
	if got := strings.Join(pins, ","); err != nil || got != "glm-5.3@,glm-5.3@xhigh" {
		t.Fatalf("pins = %s, %v", got, err)
	}
	var efforts []string
	for _, r := range h.logged() {
		efforts = append(efforts, r.Model+"@"+r.Effort)
	}
	if got := strings.Join(efforts, ","); got != "gpt-6.1-sol@xhigh,glm-5.3@high,glm-5.3@" {
		t.Fatalf("logged efforts = %s", got)
	}
}

func TestClaudeCodeLearnsItsPinnedModel(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	claude := []string{"User-Agent", claudeCode, "X-Claude-Code-Session-Id", "cc-note"}
	body := `{"model":"auto","system":[{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"which model are you?"}],` + tools + `}`
	response := h.do(http.MethodPost, "/v1/messages", body, claude...)
	readAll(t, response)
	if response.Header.Get("X-Ts-Router-Model") != "gpt-6.1-sol" {
		t.Fatalf("X-Ts-Router-Model = %q", response.Header.Get("X-Ts-Router-Model"))
	}
	sent := string(h.cpa.routed()[0].Body)
	want := `{"model":"gpt-6.1-sol","system":[{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}},{"type":"text","text":"` + modelNote("gpt-6.1-sol") + `"}],"messages":`
	if !strings.HasPrefix(sent, want) {
		t.Fatalf("sent body = %s", sent)
	}
	// Counting tokens follows the pin but generates nothing, and an explicit
	// model passes through untouched: neither gets the note.
	readAll(t, h.do(http.MethodPost, "/v1/messages/count_tokens", body, claude...))
	readAll(t, h.do(http.MethodPost, "/v1/messages", strings.Replace(body, `"auto"`, `"gpt-6.1-sol"`, 1), claude...))
	for _, r := range h.cpa.routed()[1:] {
		if r.Model != "gpt-6.1-sol" || strings.Contains(string(r.Body), "Model note") {
			t.Fatalf("%s for %s got the model note: %s", r.Path, r.Model, r.Body)
		}
	}
}

func TestClaudeKeepsClaudeCodesRequestShape(t *testing.T) {
	h := newHarness(t, "")
	h.yaml = strings.Replace(h.yaml, "  fallback:",
		"    - {id: claude-opus-5-5, context: 1000000, output: 128000, vision: true, description: \"the hardest work\"}\n  fallback:", 1)
	h.restart()
	h.jev.answer = "claude-opus-5-5"
	body := `{"model":"auto","system":[{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"redesign the settings area"}],` + tools + `}`
	readAll(t, h.do(http.MethodPost, "/v1/messages", body,
		"User-Agent", claudeCode, "X-Claude-Code-Session-Id", "cc-anthropic"))
	if sent := string(h.cpa.routed()[0].Body); sent != strings.Replace(body, `"auto"`, `"claude-opus-5-5"`, 1) {
		t.Fatalf("a request for Claude changed beyond its model: %s", sent)
	}
}

func TestServerErrorsReachTheClientUntilTheyRun(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	var overloaded atomic.Bool
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if overloaded.Load() && r.Model == "gpt-6.1-sol" {
			w.WriteHeader(529)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`))
			return
		}
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}
	readAll(t, h.claude("ses_blip", "fix the race in the cursor"))
	overloaded.Store(true)
	// The first two go back to Claude Code, which retries them on the same
	// model and account; the third in a row moves the thread for good.
	for attempt := 1; attempt <= 2; attempt++ {
		response := h.claude("ses_blip", "continue")
		readAll(t, response)
		if response.StatusCode != 529 || response.Header.Get("X-Ts-Router-Model") != "gpt-6.1-sol" {
			t.Fatalf("attempt %d: status %d on %s", attempt, response.StatusCode, response.Header.Get("X-Ts-Router-Model"))
		}
	}
	response := h.claude("ses_blip", "continue")
	if body := readAll(t, response); response.StatusCode != http.StatusOK || !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("the third overload did not fail over: %d %s", response.StatusCode, body)
	}
	overloaded.Store(false)
	if body := readAll(t, h.claude("ses_blip", "continue")); !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("the failover did not hold: %s", body)
	}
}

func TestRequestsCarryUsageAndCost(t *testing.T) {
	h := newHarness(t, "")
	h.yaml += "prices:\n  gpt-6.1-sol: {input: 2, output: 10, cache_read: 0.2}\n"
	h.restart()
	h.jev.answer = "gpt-6.1-sol"
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Model == "glm-5.3-flash" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"type":"message","usage":{"input_tokens":7,"output_tokens":3}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_start\n" +
			`data: {"type":"message_start","message":{"usage":{"input_tokens":300,"cache_read_input_tokens":1200,"output_tokens":1}}}` + "\n\n" +
			"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}` + "\n\n" +
			"event: message_delta\n" +
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":40}}` + "\n\n"))
	}
	readAll(t, h.claude("ses_cost", "hello"))
	readAll(t, h.claude("ses_cost", "", "X-Claude-Code-Request-Class", "auxiliary"))
	logged := h.logged()
	routed, auxiliary := logged[0], logged[1]
	want := (300*2.0 + 40*10.0 + 1200*0.2) / 1e6
	if routed.Tokens != (usage.Tokens{Input: 300, Output: 40, CacheRead: 1200}) || !routed.Priced || math.Abs(routed.Cost-want) > 1e-12 {
		t.Fatalf("routed request = %+v", routed)
	}
	if auxiliary.Model != "glm-5.3-flash" || auxiliary.Tokens != (usage.Tokens{Input: 7, Output: 3}) || auxiliary.Priced {
		t.Fatalf("unpriced auxiliary request = %+v", auxiliary)
	}
}

func TestNonFailoverErrorsPassThrough(t *testing.T) {
	h := newHarness(t, "")
	h.cpa.respond = func(w http.ResponseWriter, _ recorded) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad tool schema"}`))
	}
	response := h.claude("ses_400", "x")
	if response.StatusCode != http.StatusBadRequest || len(h.cpa.routed()) != 1 {
		t.Fatalf("status %d after %d upstream calls", response.StatusCode, len(h.cpa.routed()))
	}
}

func TestSideAndAuxiliaryRequestsNeverCreatePins(t *testing.T) {
	h := newHarness(t, "")
	claude := []string{"User-Agent", "claude-cli/2.1.283", "X-Claude-Code-Session-Id", "cc-side"}
	compaction := append(slices.Clone(claude), "X-Claude-Code-Request-Class", "compaction")
	message := `{"model":"auto","max_tokens":10,"messages":[{"role":"user","content":"hello"}]}`
	count := readAll(t, h.do(http.MethodPost, "/v1/messages/count_tokens", message, claude...))
	compacted := readAll(t, h.do(http.MethodPost, "/v1/messages", message, compaction...))
	aux := readAll(t, h.do(http.MethodPost, "/v1/messages", message, append(claude, "X-Claude-Code-Request-Class", "auxiliary")...))
	noSession := readAll(t, h.do(http.MethodPost, "/v1/messages", message, "User-Agent", "curl/8"))
	if !strings.Contains(count, `"glm-5.3"`) || !strings.Contains(compacted, `"glm-5.3"`) || !strings.Contains(aux, `"glm-5.3-flash"`) ||
		!strings.Contains(noSession, `"glm-5.3"`) {
		t.Fatalf("responses: %s | %s | %s | %s", count, compacted, aux, noSession)
	}
	if h.jev.calls.Load() != 0 {
		t.Fatal("a side request called the decider")
	}
	h.jev.answer = "gpt-6.1-sol"
	if main := readAll(t, h.do(http.MethodPost, "/v1/messages", message, append(claude, "X-Claude-Code-Request-Class", "main")...)); !strings.Contains(main, `"gpt-6.1-sol"`) {
		t.Fatalf("main request = %s", main)
	}
	if count := readAll(t, h.do(http.MethodPost, "/v1/messages/count_tokens", message, claude...)); !strings.Contains(count, `"gpt-6.1-sol"`) {
		t.Fatalf("count_tokens does not follow the pin: %s", count)
	}
	if compacted := readAll(t, h.do(http.MethodPost, "/v1/messages", message, compaction...)); !strings.Contains(compacted, `"gpt-6.1-sol"`) {
		t.Fatalf("a compaction does not follow the pin: %s", compacted)
	}
	if page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{}); err != nil || page.Total != 1 || h.jev.calls.Load() != 1 {
		t.Fatalf("threads = %+v after %d decisions, %v", page.Threads, h.jev.calls.Load(), err)
	}
}

func TestTheSessionComesFromTheHeaderOrTheMetadata(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	readAll(t, h.claude("0b7c3c7e-5d0e-4d8f-9a47-1f6a2d9e8c11", "plan the migration"))
	h.jev.answer = "deepseek-flash"
	for _, userID := range []string{
		`user_4f1c_account_9d2e_session_0b7c3c7e-5d0e-4d8f-9a47-1f6a2d9e8c11`,
		`{\"device_id\":\"4f1c\",\"account_uuid\":\"9d2e\",\"session_id\":\"0b7c3c7e-5d0e-4d8f-9a47-1f6a2d9e8c11\"}`,
	} {
		body := `{"model":"auto","max_tokens":1024,"metadata":{"user_id":"` + userID + `"},` +
			`"messages":[{"role":"user","content":"grep for X"}]}`
		if follower := readAll(t, h.do(http.MethodPost, "/v1/messages", body, "User-Agent", claudeCode)); !strings.Contains(follower, `"gpt-6.1-sol"`) {
			t.Fatalf("metadata.user_id %s did not name the thread: %s", userID, follower)
		}
	}
	if calls := h.jev.calls.Load(); calls != 1 {
		t.Fatalf("decider called %d times for one thread", calls)
	}
}

func TestImagesRestrictAndUpgradeToVisionModels(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "deepseek-flash"
	readAll(t, h.claudeImage("ses_img"))
	if keys := h.jev.keys[0]; len(keys) != 2 || strings.Contains(strings.Join(keys, ","), "glm-5.3") {
		t.Fatalf("vision candidates = %v", keys)
	}
	if state := h.jev.states[0]; state != "images=yes\n\nwhat is on this screenshot" {
		t.Fatalf("decider state = %q", state)
	}

	h.jev.answer = "glm-5.3"
	readAll(t, h.claude("ses_text", "plain text first"))
	upgraded := readAll(t, h.claudeImage("ses_text"))
	if !strings.Contains(upgraded, `"gpt-6.1-sol"`) {
		t.Fatalf("image turn did not upgrade the glm-5.3 pin: %s", upgraded)
	}
	if later := readAll(t, h.claude("ses_text", "text again")); !strings.Contains(later, `"gpt-6.1-sol"`) {
		t.Fatalf("upgrade is not permanent: %s", later)
	}
}

func TestConcurrentFirstRequestsShareOneDecision(t *testing.T) {
	h := newHarness(t, "")
	var wg sync.WaitGroup
	for range 20 {
		wg.Go(func() { readAll(t, h.claude("ses_burst", "start")) })
	}
	wg.Wait()
	if calls := h.jev.calls.Load(); calls != 1 {
		t.Fatalf("decider called %d times", calls)
	}
}

func TestStreamsChunksAsTheyArrive(t *testing.T) {
	h := newHarness(t, "")
	release := make(chan struct{})
	h.cpa.respond = func(w http.ResponseWriter, _ recorded) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: first\n\n"))
		w.(http.Flusher).Flush()
		<-release
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}
	response := h.claude("ses_sse", "stream please")
	reader := bufio.NewReader(response.Body)
	line, err := reader.ReadString('\n')
	close(release)
	if err != nil || line != "data: first\n" || response.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("first line %q, %v", line, err)
	}
	rest, _ := io.ReadAll(reader)
	if !strings.Contains(string(rest), "[DONE]") {
		t.Fatalf("rest = %q", rest)
	}
}

func TestRejectsBadKeysAndServesModels(t *testing.T) {
	h := newHarness(t, "")
	request, _ := http.NewRequest(http.MethodPost, h.server.URL+"/v1/messages", strings.NewReader(`{"model":"auto"}`))
	request.Header.Set("X-Api-Key", "wrong")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if kind, _ := anthropicError(t, response); response.StatusCode != http.StatusUnauthorized || kind != "authentication_error" {
		t.Fatalf("bad key: %d %s", response.StatusCode, kind)
	}
	if len(h.cpa.routed()) != 0 || h.jev.calls.Load() != 0 {
		t.Fatal("an unauthenticated request reached CLIProxy or the decider")
	}

	models := readAll(t, h.do(http.MethodGet, "/v1/models", ""))
	if !strings.HasPrefix(models, `{"data":[{"context_length":200000,"created":0,"id":"auto","max_output_tokens":64000,"object":"model","owned_by":"ai-router"}`) {
		t.Fatalf("models = %s", models)
	}

	health, err := http.Get(h.server.URL + "/healthz")
	if err != nil || health.StatusCode != http.StatusOK {
		t.Fatalf("healthz = %v, %v", health, err)
	}
	health.Body.Close()
	stats := readAll(t, h.do(http.MethodGet, "/stats", ""))
	if !strings.Contains(stats, `"last_24h"`) {
		t.Fatalf("stats = %s", stats)
	}
}

func TestNoPromptTextInLogsOrDatabase(t *testing.T) {
	// Prompts reach the database only with record_prompts, which is on
	// unless turned off, and never the logs or the request log.
	h := newHarness(t, "  record_prompts: false")
	const sentinel = "SENTINEL-PROMPT-7f3a"
	h.jev.status = http.StatusTooManyRequests
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Model == "gpt-6-luna" {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		// The answer repeats the prompt, like a model quoting it back.
		_, _ = w.Write(r.Body)
	}
	readAll(t, h.claude("ses_private", "please handle "+sentinel))
	readAll(t, h.do(http.MethodPost, "/v1/messages",
		fmt.Sprintf(`{"model":"gpt-6.1-sol","max_tokens":1024,"messages":[{"role":"user","content":%q}]}`, sentinel)))
	logged := h.logged()
	if len(logged) != 1 {
		t.Fatalf("logged %d requests, want the routed one", len(logged))
	}
	for _, entry := range logged {
		if strings.Contains(fmt.Sprintf("%+v", entry), sentinel) {
			t.Fatalf("prompt text reached the request log: %+v", entry)
		}
	}
	if strings.Contains(h.logs.String(), sentinel) {
		t.Fatal("prompt text reached the logs")
	}
	for _, suffix := range []string{"", "-wal"} {
		data, err := os.ReadFile(h.database + suffix)
		if err == nil && bytes.Contains(data, []byte(sentinel)) {
			t.Fatalf("prompt text reached %s", h.database+suffix)
		}
	}
}

func TestHandledRequestsAreLoggedOnceFinished(t *testing.T) {
	h := newHarness(t, "")
	h.jev.answer = "gpt-6.1-sol"
	var quota, unreachable, broken atomic.Bool
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		switch {
		case r.Model == "gpt-6.1-sol" && quota.Load():
			w.WriteHeader(http.StatusTooManyRequests)
		case unreachable.Load():
			connection, _, _ := w.(http.Hijacker).Hijack()
			connection.Close()
		case broken.Load():
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: partial\n\n"))
			w.(http.Flusher).Flush()
			connection, _, _ := w.(http.Hijacker).Hijack()
			connection.Close()
		default:
			_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
		}
	}
	readAll(t, h.claude("ses_log", "design the storage layer"))
	// An explicit model is relayed but never logged.
	readAll(t, h.do(http.MethodPost, "/v1/messages", `{"model":"claude-sonnet-5","messages":[]}`,
		"User-Agent", "claude-cli/2.1.283", "X-Claude-Code-Session-Id", "cc-log"))
	quota.Store(true)
	readAll(t, h.claude("ses_log", "continue"))
	unreachable.Store(true)
	readAll(t, h.do(http.MethodPost, "/v1/messages", `{"model":"auto","messages":[]}`))
	unreachable.Store(false)
	broken.Store(true)
	// A compaction is a side request, charged to its thread.
	_, _ = io.ReadAll(h.claude("ses_log", "summarize the conversation", "X-Claude-Code-Request-Class", "compaction").Body)
	readAll(t, h.do(http.MethodGet, "/v1/models", ""))
	readAll(t, h.do(http.MethodGet, "/stats", ""))
	readAll(t, h.do(http.MethodGet, "/healthz", ""))
	readAll(t, h.do(http.MethodPost, "/v1/messages", `{"model":"auto"}`, "Authorization", "Bearer wrong"))
	readAll(t, h.do(http.MethodPost, "/v1/chat/completions", `{"model":"auto"}`))

	page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || len(page.Threads) != 1 {
		t.Fatalf("threads = %+v, %v", page, err)
	}
	key := page.Threads[0].Key
	logged := h.logged()
	want := []pin.RequestLog{
		{Key: key, Kind: "messages", Requested: "auto", Model: "gpt-6.1-sol", Reason: "decided", Status: 200},
		{Key: key, Kind: "messages", Requested: "auto", Model: "glm-5.3", Reason: "failover", Status: 200, Failover: true},
		{Kind: "messages", Requested: "auto", Model: "glm-5.3", Reason: "no_session", Error: "CLIProxy unreachable: EOF"},
		{Key: key, Kind: "messages", Requested: "auto", Model: "glm-5.3", Reason: "side_request", Status: 200, Error: "stream interrupted: unexpected EOF"},
	}
	if len(logged) != len(want) {
		t.Fatalf("logged %d requests: %+v", len(logged), logged)
	}
	for i, entry := range logged {
		if entry.ID == 0 || entry.At.IsZero() {
			t.Errorf("request %d = %+v", i, entry)
		}
		entry.ID, entry.At, entry.Latency = 0, time.Time{}, 0
		if entry != want[i] {
			t.Errorf("request %d = %+v, want %+v", i, entry, want[i])
		}
	}
	if logged[0].Latency < 20*time.Millisecond {
		t.Errorf("the decided request took %v, less than its decision", logged[0].Latency)
	}
}

func TestAFullRequestLogNeverBlocksARequest(t *testing.T) {
	h := newHarness(t, "")
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	h.app.history.Close()
	h.app.history = newHistory(func(context.Context, pin.RequestLog) error {
		<-release
		return nil
	}, 1, h.app.counters, h.app.log)

	const requests = 5
	statuses := make(chan int, requests)
	go func() {
		for range requests {
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"auto"}`))
			request.Header.Set("Authorization", "Bearer "+proxyKey)
			h.app.ServeHTTP(response, request)
			statuses <- response.Code
		}
	}()
	for range requests {
		select {
		case status := <-statuses:
			if status != http.StatusOK {
				t.Fatalf("status = %d", status)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("a request waited for the request log")
		}
	}
	// The stalled writer holds at most one request and the queue another.
	if dropped := h.app.counters.Snapshot()["history.dropped"]; dropped < requests-2 {
		t.Fatalf("dropped %d requests, want at least %d", dropped, requests-2)
	}
}

// glmLimits caps glm-5.3 at concurrency, spilling to gpt-6.1-sol.
func glmLimits(concurrency int) string {
	return fmt.Sprintf("limits:\n  glm-5.3: {concurrency: %d, throttle_codes: [\"1302\"], retries: 3, queue_timeout: 1s, spill: gpt-6.1-sol}\n", concurrency)
}

func TestThrottledRequestsRetryFromTheQueue(t *testing.T) {
	pause := retryPause
	retryPause = time.Millisecond
	t.Cleanup(func() { retryPause = pause })
	h := newHarness(t, "")
	h.yaml += glmLimits(2)
	h.restart()
	var throttled atomic.Int32
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Model == "glm-5.3" && throttled.Add(1) <= 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","code":"1302","message":"[1302][Rate limit reached for requests]"}}`))
			return
		}
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}
	response := h.claude("ses_throttled", "add a flag")
	if body := readAll(t, response); response.StatusCode != http.StatusOK || !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("status %d: %s", response.StatusCode, body)
	}
	if calls := len(h.cpa.routed()); calls != 3 {
		t.Fatalf("CLIProxy saw %d calls, want two throttled and one served", calls)
	}
	// A Content-Length body can reach the client before the handler's
	// deferred release runs. Observe the gate after the request completes.
	h.app.inflight.wait()
	stats := h.app.limits.Gate("glm-5.3").Stats()
	if stats.Throttles != 2 || stats.Limit != 1 || stats.InFlight != 0 {
		t.Fatalf("gate = %+v", stats)
	}
	page, _ := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if len(page.Threads) != 1 || page.Threads[0].Model != "glm-5.3" || page.Threads[0].FailedOver {
		t.Fatalf("a throttled thread failed over: %+v", page.Threads)
	}
}

func TestExplicitModelsUseConfiguredConcurrencyGate(t *testing.T) {
	h := newHarness(t, `limits:
  gpt-6.1-sol: {concurrency: 1, queue_timeout: 1s, retries: 0}`)
	hold := make(chan struct{})
	started := make(chan struct{})
	var startedOnce sync.Once
	var inFlight atomic.Int32
	var peak atomic.Int32
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Model != "gpt-6.1-sol" {
			_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
			return
		}
		current := inFlight.Add(1)
		for {
			old := peak.Load()
			if current <= old || peak.CompareAndSwap(old, current) {
				break
			}
		}
		startedOnce.Do(func() { close(started) })
		<-hold
		inFlight.Add(-1)
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}

	body := `{"model":"gpt-6.1-sol","max_tokens":8,"messages":[{"role":"user","content":"one"}]}`
	first := make(chan *http.Response, 1)
	go func() { first <- h.do(http.MethodPost, "/v1/messages", body) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("first request did not reach the upstream")
	}

	second := make(chan *http.Response, 1)
	go func() { second <- h.do(http.MethodPost, "/v1/messages", body) }()
	select {
	case response := <-second:
		response.Body.Close()
		t.Fatal("second request bypassed the concurrency gate")
	case <-time.After(50 * time.Millisecond):
	}
	close(hold)
	for _, request := range []chan *http.Response{first, second} {
		select {
		case response := <-request:
			if response.StatusCode != http.StatusOK {
				response.Body.Close()
				t.Fatalf("status = %d, want 200", response.StatusCode)
			}
			response.Body.Close()
		case <-time.After(time.Second):
			t.Fatal("gated request did not finish")
		}
	}
	if peak.Load() != 1 {
		t.Fatalf("peak concurrency = %d, want 1", peak.Load())
	}
}

func TestOtherRateLimitsStillFailOver(t *testing.T) {
	h := newHarness(t, "")
	h.yaml += glmLimits(2)
	h.restart()
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Model == "glm-5.3" {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"code":"1308","message":"usage limit reached"}}`))
			return
		}
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}
	if body := readAll(t, h.claude("ses_quota", "add a flag")); !strings.Contains(body, `"deepseek-flash"`) {
		t.Fatalf("no failover: %s", body)
	}
	if stats := h.app.limits.Gate("glm-5.3").Stats(); stats.Throttles != 0 || stats.InFlight != 0 {
		t.Fatalf("gate = %+v", stats)
	}
}

func TestABusyModelSpillsNewThreadsAndTurnsAwayWaiters(t *testing.T) {
	h := newHarness(t, "  record_prompts: true")
	h.yaml += glmLimits(1)
	h.restart()
	hold := make(chan struct{})
	h.cpa.respond = func(w http.ResponseWriter, r recorded) {
		if r.Model == "glm-5.3" {
			<-hold
		}
		_, _ = fmt.Fprintf(w, `{"model":%q}`, r.Model)
	}
	first := make(chan string)
	go func() { first <- readAll(t, h.claude("ses_busy_first", "add a flag")) }()
	for h.app.limits.Gate("glm-5.3").Stats().InFlight != 1 {
		time.Sleep(time.Millisecond)
	}

	// A new thread decided for glm-5.3 goes to the spill model.
	if body := readAll(t, h.claude("ses_busy_second", "write the tests")); !strings.Contains(body, `"gpt-6.1-sol"`) {
		t.Fatalf("no spill: %s", body)
	}
	// A request without a thread has no pin to spill and waits its queue
	// timeout out.
	started := time.Now()
	busy := h.do(http.MethodPost, "/v1/messages", `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`)
	if body := readAll(t, busy); busy.StatusCode != http.StatusTooManyRequests || busy.Header.Get("Retry-After") == "" ||
		!strings.Contains(body, "glm-5.3 is busy") || time.Since(started) < time.Second {
		t.Fatalf("busy = %d %s after %v", busy.StatusCode, body, time.Since(started))
	}
	close(hold)
	if body := <-first; !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("first = %s", body)
	}
	page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil {
		t.Fatal(err)
	}
	for _, thread := range page.Threads {
		if thread.Model == "gpt-6.1-sol" && thread.Decision.Basis.Spilled != "glm-5.3" {
			t.Fatalf("spilled decision = %+v", thread.Decision)
		}
	}
	if stats := h.app.limits.Gate("glm-5.3").Stats(); stats.InFlight != 0 || stats.Waiting != 0 {
		t.Fatalf("gate = %+v", stats)
	}
}
