package app

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/route"
	"github.com/MarioGK/AI-Router/internal/usage"
)

type policyVerdict struct {
	model         string
	level         string
	probabilities map[string]float64
}

type policyQuestion struct {
	Type         string              `json:"type"`
	Instructions decide.Instructions `json:"instructions"`
	Criteria     jsontext.Value      `json:"criteria"`
}

type policyClassifierRequest struct {
	Model     string                    `json:"model"`
	State     string                    `json:"state"`
	Questions map[string]policyQuestion `json:"questions"`
}

// policyTransport supplies both HTTP peers without opening a listener or
// dialing a socket. The real SystemOne decider still builds and parses its
// wire protocol, and the real forwarder still rewrites and relays requests.
type policyTransport struct {
	t           *testing.T
	cfg         *config.Config
	verdict     policyVerdict
	reject      string
	unavailable []string
	// fail answers every Messages request with this status when set, and
	// cut with a stream that ends before message_stop.
	fail       int
	cut        bool
	classified []policyClassifierRequest
	routed     []recorded
}

func (p *policyTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	var body []byte
	if r.Body != nil {
		defer r.Body.Close()
		var err error
		body, err = io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
	}
	switch {
	case r.URL.Host == "jev.test" && r.URL.Path == "/v1/systemone":
		if r.Header.Get("Authorization") != "Bearer jev-key" {
			p.t.Errorf("classifier authorization = %q", r.Header.Get("Authorization"))
		}
		var request policyClassifierRequest
		if err := json.Unmarshal(body, &request); err != nil {
			return nil, err
		}
		p.classified = append(p.classified, request)
		probabilities := p.verdict.probabilities
		if probabilities == nil {
			probabilities = map[string]float64{p.verdict.model: 1}
		}
		return p.response(r, http.StatusOK, map[string]any{
			"model": "jev-latest",
			"answers": map[string]any{
				"model": map[string]any{
					"choice": p.verdict.model, "probabilities": probabilities,
				},
				"reasoning": map[string]any{"probabilities": map[string]float64{p.verdict.level: 1}},
			},
		})
	case r.URL.Host == "upstream.test" && r.URL.Path == "/v1/models":
		var models []map[string]string
		for _, id := range p.cfg.Auto.Required() {
			if !slices.Contains(p.unavailable, id) {
				models = append(models, map[string]string{"id": id})
			}
		}
		return p.response(r, http.StatusOK, map[string]any{"data": models})
	case r.URL.Host == "upstream.test" && r.URL.Path == "/v1/messages":
		var fields struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(body, &fields); err != nil {
			return nil, err
		}
		p.routed = append(p.routed, recorded{
			Method: r.Method, Path: r.URL.Path, Header: r.Header.Clone(), Body: body, Model: fields.Model,
		})
		if p.fail != 0 {
			return p.response(r, p.fail, map[string]any{"error": map[string]string{"message": "invalid request"}})
		}
		if p.cut {
			stream := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":" +
				strconv.Quote(fields.Model) + "}}\n\n"
			return &http.Response{
				StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}},
				Body: io.NopCloser(strings.NewReader(stream)), ContentLength: int64(len(stream)), Request: r,
			}, nil
		}
		if config.Bare(fields.Model) == p.reject {
			// Failed attempts must not contribute usage to the final request.
			return p.response(r, http.StatusTooManyRequests, map[string]any{
				"error": map[string]string{"message": "usage limit reached"},
				"usage": map[string]int{"input_tokens": 999999, "output_tokens": 999999},
			})
		}
		return p.response(r, http.StatusOK, map[string]any{
			"model": fields.Model, "content": []map[string]string{{"type": "text", "text": "done"}},
			"usage": map[string]int{
				"input_tokens": 11, "output_tokens": 7,
				"cache_read_input_tokens": 3, "cache_creation_input_tokens": 2,
			},
		})
	default:
		return nil, fmt.Errorf("unexpected outbound request: %s %s", r.Method, r.URL)
	}
}

func (p *policyTransport) response(r *http.Request, status int, value any) (*http.Response, error) {
	body, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return &http.Response{
		StatusCode: status, Header: http.Header{"Content-Type": {"application/json"}},
		Body: io.NopCloser(strings.NewReader(string(body))), ContentLength: int64(len(body)), Request: r,
	}, nil
}

type policyHarness struct {
	t         *testing.T
	app       *App
	transport *policyTransport
}

func newPolicyHarness(t *testing.T, verdict policyVerdict, configure ...func(*config.Config)) *policyHarness {
	t.Helper()
	t.Setenv("CLIPROXYAPI_API_KEY", proxyKey)
	t.Setenv("JEV_API_KEY", "jev-key")
	document, err := os.ReadFile("../../config.example.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var example config.Config
	if err := yaml.Unmarshal(document, &example); err != nil {
		t.Fatal(err)
	}
	// Keep the example's candidate criteria, efforts, retirement mappings and
	// cost economy; replace only deployment paths and outbound endpoints.
	example.Database = filepath.Join(t.TempDir(), "router.db")
	example.SecretsFiles = nil
	example.Upstream.BaseURL = "http://upstream.test"
	example.Accounts.ManagementKeyEnv = ""
	example.Deciders = example.Deciders[:1]
	example.Deciders[0].URL = "https://jev.test/v1/systemone"
	for _, customize := range configure {
		customize(&example)
	}
	document, err = yaml.Marshal(example)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Parse(document)
	if err != nil {
		t.Fatal(err)
	}
	transport := &policyTransport{t: t, cfg: cfg, verdict: verdict}
	client := &http.Client{Transport: transport}
	a, err := newWithClients(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), client, client)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Error(err)
		}
	})
	if err := a.catalog.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	return &policyHarness{t: t, app: a, transport: transport}
}

func (h *policyHarness) messages(session, requested, prompt string) *httptest.ResponseRecorder {
	h.t.Helper()
	body := fmt.Sprintf(`{"model":%q,"max_tokens":1024,"messages":[{"role":"user","content":%q}],%s}`,
		requested, prompt, tools)
	w := h.request(session, body)
	if w.Code != http.StatusOK {
		h.t.Fatalf("messages = %d: %s", w.Code, w.Body.String())
	}
	return w
}

func (h *policyHarness) request(session, body string) *httptest.ResponseRecorder {
	h.t.Helper()
	r := httptest.NewRequestWithContext(h.t.Context(), http.MethodPost, "http://router.test/v1/messages", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer "+proxyKey)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("User-Agent", claudeCode)
	r.Header.Set("X-Claude-Code-Session-Id", session)
	w := httptest.NewRecorder()
	h.app.ServeHTTP(w, r)
	return w
}

func (h *policyHarness) checkResponse(w *httptest.ResponseRecorder, model, effort string) {
	h.t.Helper()
	if got := w.Header().Get("X-Ts-Router-Model"); got != model {
		h.t.Errorf("response model = %q, want %q", got, model)
	}
	if got := w.Header().Get("X-Ts-Router-Effort"); got != effort {
		h.t.Errorf("response effort = %q, want %q", got, effort)
	}
	var response struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || response.Model != config.Suffixed(model, effort) {
		h.t.Errorf("response body = %s, error %v", w.Body.String(), err)
	}
}

func (h *policyHarness) checkRouted(index int, session, model, effort string) {
	h.t.Helper()
	if index >= len(h.transport.routed) {
		h.t.Fatalf("routed %d requests, need request %d", len(h.transport.routed), index)
	}
	r := h.transport.routed[index]
	var body struct {
		System string `json:"system"`
	}
	if err := json.Unmarshal(r.Body, &body); err != nil {
		h.t.Fatal(err)
	}
	expectedNote := modelNote(model)
	if strings.HasPrefix(model, "claude-") {
		expectedNote = ""
	}
	if r.Model != config.Suffixed(model, effort) || body.System != expectedNote {
		h.t.Errorf("routed body = %s, want %s with provider system prompt %q", r.Body, config.Suffixed(model, effort), expectedNote)
	}
	if r.Method != http.MethodPost || r.Path != "/v1/messages" ||
		r.Header.Get("Authorization") != "Bearer "+proxyKey || r.Header.Get("User-Agent") != claudeCode ||
		r.Header.Get("X-Claude-Code-Session-Id") != session {
		h.t.Errorf("routed request = %+v", r)
	}
}

func (h *policyHarness) logged() []pin.RequestLog {
	h.t.Helper()
	h.app.history.Close()
	entries, err := h.app.store.Requests(h.t.Context(), pin.RequestFilter{})
	if err != nil {
		h.t.Fatal(err)
	}
	slices.Reverse(entries)
	return entries
}

func (h *policyHarness) checkLogged(entry pin.RequestLog, requested, model, effort, reason string, failedOver bool) {
	h.t.Helper()
	wantTokens := usage.Tokens{Input: 11, Output: 7, CacheRead: 3, CacheWrite: 2}
	price, priced := h.app.cfg.Price(model)
	if entry.Key == "" || entry.Kind != "messages" || entry.Requested != requested || entry.Model != model ||
		entry.Effort != effort || entry.Reason != reason || entry.Status != http.StatusOK ||
		entry.Failover != failedOver || entry.Error != "" || entry.Tokens != wantTokens ||
		entry.Priced != priced || math.Abs(entry.Cost-price.Cost(wantTokens)) > 1e-12 {
		h.t.Errorf("logged request = %+v", entry)
	}
}

func TestRoutingPolicySolAlwaysRunsAtHigh(t *testing.T) {
	for _, requested := range []string{"auto", "auto(low)", "auto(xhigh)", "auto(8192)"} {
		t.Run(requested, func(t *testing.T) {
			h := newPolicyHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "low"})
			for turn := range 2 {
				h.checkResponse(h.messages("fixed-sol", requested, "Fix the cursor race before today's release."), "gpt-6.1-sol", "high")
				h.checkRouted(turn, "fixed-sol", "gpt-6.1-sol", "high")
			}
			if len(h.transport.classified) != 1 {
				t.Fatalf("classifier called %d times, want one decision for the thread", len(h.transport.classified))
			}
			entries := h.logged()
			if len(entries) != 2 || entries[0].Key != entries[1].Key {
				t.Fatalf("logged requests = %+v", entries)
			}
			h.checkLogged(entries[0], requested, "gpt-6.1-sol", "high", route.ReasonDecided, false)
			h.checkLogged(entries[1], requested, "gpt-6.1-sol", "high", route.ReasonPinned, false)
			p, err := h.app.store.Get(t.Context(), entries[0].Key)
			if err != nil || p == nil || p.Level != "low" {
				t.Fatalf("fixed effort changed the judged pin level: %+v, %v", p, err)
			}
		})
	}
}

func TestRoutingPolicyFlexibleCandidatesKeepClientEffort(t *testing.T) {
	for _, effort := range []string{"low", "xhigh", "8192"} {
		t.Run(effort, func(t *testing.T) {
			h := newPolicyHarness(t, policyVerdict{model: "glm-5.3", level: "medium"})
			requested := "auto(" + effort + ")"
			h.checkResponse(h.messages("flexible", requested, "Add ordinary endpoint tests; no rush."), "glm-5.3", effort)
			h.checkRouted(0, "flexible", "glm-5.3", effort)
			entries := h.logged()
			if len(entries) != 1 {
				t.Fatalf("logged requests = %+v", entries)
			}
			h.checkLogged(entries[0], requested, "glm-5.3", effort, route.ReasonDecided, false)
		})
	}
}

func TestRoutingPolicyVisionUpgradeStaysOnIncludedGLMFlash(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "glm-5.3", level: "medium"})
	const session = "vision-upgrade"
	h.checkResponse(h.messages(session, "auto(low)", "Add an ordinary component; no rush."), "glm-5.3", "low")
	image := strings.Replace(screenshot, `"model":"auto"`, `"model":"auto(low)"`, 1)
	h.checkResponse(h.request(session, image), "glm-5.3-flash", "low")
	h.checkResponse(h.messages(session, "auto(low)", "Continue without another image."), "glm-5.3-flash", "low")
	h.checkRouted(0, session, "glm-5.3", "low")
	h.checkRouted(1, session, "glm-5.3-flash", "low")
	h.checkRouted(2, session, "glm-5.3-flash", "low")
	if len(h.transport.classified) != 1 {
		t.Fatalf("vision upgrade called the classifier %d times", len(h.transport.classified))
	}
	entries := h.logged()
	if len(entries) != 3 {
		t.Fatalf("vision requests = %+v", entries)
	}
	h.checkLogged(entries[0], "auto(low)", "glm-5.3", "low", route.ReasonDecided, false)
	for _, entry := range entries[1:] {
		h.checkLogged(entry, "auto(low)", "glm-5.3-flash", "low", route.ReasonPinned, false)
	}
	p, err := h.app.store.Get(t.Context(), entries[0].Key)
	if err != nil || p == nil || p.Model != "glm-5.3-flash" || p.Level != "medium" || !p.Upgraded || p.FailedOver {
		t.Fatalf("vision pin = %+v, %v", p, err)
	}
}

func TestRoutingPolicyVisionUpgradeUsesFixedSolEffortWithoutIncludedVision(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{model: "glm-5.3", level: "medium"}, func(cfg *config.Config) {
		for i := range cfg.Auto.Candidates {
			if cfg.Auto.Candidates[i].ID == "glm-5.3-flash" {
				cfg.Auto.Candidates[i].Vision = false
			}
		}
	})
	const session = "vision-upgrade-sol"
	h.checkResponse(h.messages(session, "auto(low)", "Add an ordinary component; no rush."), "glm-5.3", "low")
	image := strings.Replace(screenshot, `"model":"auto"`, `"model":"auto(low)"`, 1)
	h.checkResponse(h.request(session, image), "gpt-6.1-sol", "high")
	h.checkResponse(h.messages(session, "auto(low)", "Continue without another image."), "gpt-6.1-sol", "high")
	h.checkRouted(1, session, "gpt-6.1-sol", "high")
	h.checkRouted(2, session, "gpt-6.1-sol", "high")
	entries := h.logged()
	p, err := h.app.store.Get(t.Context(), entries[0].Key)
	if err != nil || p == nil || p.Model != "gpt-6.1-sol" || p.Level != "medium" || !p.Upgraded || p.FailedOver {
		t.Fatalf("vision pin = %+v, %v", p, err)
	}
}

func TestRoutingPolicyNoVisionCandidateReturns503WithoutOutboundCalls(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{}, func(cfg *config.Config) {
		for i := range cfg.Auto.Candidates {
			cfg.Auto.Candidates[i].Vision = false
		}
	})
	w := h.request("no-vision", screenshot)
	var response struct {
		Type  string `json:"type"`
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || w.Code != http.StatusServiceUnavailable ||
		response.Type != "error" || response.Error.Type != "model_unavailable_error" {
		t.Fatalf("no-vision response = %d %s, %v", w.Code, w.Body.String(), err)
	}
	if len(h.transport.classified) != 0 || len(h.transport.routed) != 0 {
		t.Fatalf("no-vision outbound calls: classifier %d, upstream %d", len(h.transport.classified), len(h.transport.routed))
	}
	if entries := h.logged(); len(entries) != 0 {
		t.Fatalf("no-vision logged a routed request: %+v", entries)
	}
}

func TestRoutingPolicyUnavailableSpillKeepsGlmCost(t *testing.T) {
	h := newPolicyHarness(t, policyVerdict{
		model: "glm-5.3", level: "high", probabilities: map[string]float64{"glm-5.3": 0.85, "claude-opus-5-5": 0.15},
	})
	h.transport.unavailable = []string{"gpt-6.1-sol"}
	if err := h.app.catalog.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	gate := h.app.limits.Gate("glm-5.3")
	for range h.app.cfg.Limits["glm-5.3"].Concurrency {
		release, err := gate.Acquire(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(release)
	}
	// Resolver.Try runs the exact production chain without trying to acquire
	// a generation slot on the deliberately saturated GLM gate.
	decision, _ := h.app.resolver.Try(t.Context(), "Add ordinary endpoint tests; no rush.", false)
	if decision.Model != "glm-5.3" || decision.Basis.Spilled != "" || decision.Basis.Costs == nil {
		t.Fatalf("unavailable spill decision = %+v", decision)
	}
	want, ok := h.app.costs.Task("glm-5.3", "high")
	if !ok || math.Abs(decision.Basis.Costs.Task["glm-5.3"]-want) > 1e-12 {
		t.Fatalf("GLM task cost = %v, want %v (known %v)", decision.Basis.Costs.Task, want, ok)
	}
	if slices.Contains(decision.Basis.Candidates, "gpt-6.1-sol") || len(h.transport.classified) != 1 {
		t.Fatalf("spill policy candidates = %v, classifier calls %d", decision.Basis.Candidates, len(h.transport.classified))
	}
}

func TestRoutingPolicyRetiredPinsMigrateToSol(t *testing.T) {
	for _, retired := range []string{"claude-sonnet-5-5"} {
		t.Run(retired, func(t *testing.T) {
			h := newPolicyHarness(t, policyVerdict{})
			const session = "retired-thread"
			// Seed a persisted pin in the same client-key/session namespace as
			// a thread created by an earlier router deployment.
			digest := sha256.Sum256([]byte(h.app.auth.Fingerprint() + "\x00claude-code\x00" + session))
			key := hex.EncodeToString(digest[:])
			_, err := h.app.store.Create(t.Context(), pin.Pin{Key: key, Model: retired, Level: "xhigh"}, pin.Event{
				Key: key, Kind: pin.Decision, Model: retired, Level: "xhigh",
			})
			if err != nil {
				t.Fatal(err)
			}
			h.checkResponse(h.messages(session, "auto(low)", "Continue the existing task."), "gpt-6.1-sol", "high")
			h.checkRouted(0, session, "gpt-6.1-sol", "high")
			if len(h.transport.classified) != 0 {
				t.Fatal("migration classified an existing thread again")
			}
			p, err := h.app.store.Get(t.Context(), key)
			if err != nil || p == nil || p.Model != "gpt-6.1-sol" || p.Level != "xhigh" ||
				p.Decided != retired || p.FailedOver || p.Upgraded {
				t.Fatalf("migrated pin = %+v, %v", p, err)
			}
			entries := h.logged()
			if len(entries) != 1 || entries[0].Key != key {
				t.Fatalf("migration requests = %+v", entries)
			}
			h.checkLogged(entries[0], "auto(low)", "gpt-6.1-sol", "high", route.ReasonPinned, false)
			detail, err := h.app.store.Thread(t.Context(), key)
			if err != nil || detail == nil || len(detail.Events) != 2 {
				t.Fatalf("migration history = %+v, %v", detail, err)
			}
			event := detail.Events[1]
			if event.Kind != pin.Migration || event.Previous != retired || event.Model != "gpt-6.1-sol" || event.Level != "xhigh" {
				t.Fatalf("migration event = %+v", event)
			}
		})
	}
}

func TestRoutingPolicyFailoverReevaluatesDestinationEffort(t *testing.T) {
	for _, tc := range []struct {
		name, from, to, requested, level, initialEffort, finalEffort string
	}{
		{"into-sol-low", "glm-5.3", "gpt-6.1-sol", "auto(low)", "xhigh", "low", "high"},
		{"into-sol-xhigh", "glm-5.3", "gpt-6.1-sol", "auto(xhigh)", "low", "xhigh", "high"},
		{"into-sol-budget", "glm-5.3", "gpt-6.1-sol", "auto(8192)", "medium", "8192", "high"},
		{"out-of-sol-low", "gpt-6.1-sol", "glm-5.3", "auto(low)", "xhigh", "high", "low"},
		{"out-of-sol-xhigh", "gpt-6.1-sol", "glm-5.3", "auto(xhigh)", "low", "high", "xhigh"},
		{"out-of-sol-budget", "gpt-6.1-sol", "glm-5.3", "auto(8192)", "medium", "high", "8192"},
		{"out-of-sol-judged-low", "gpt-6.1-sol", "glm-5.3", "auto", "low", "high", "medium"},
		{"out-of-sol-judged-xhigh", "gpt-6.1-sol", "glm-5.3", "auto", "xhigh", "high", "xhigh"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newPolicyHarness(t, policyVerdict{model: tc.from, level: tc.level})
			h.transport.reject = tc.from
			h.checkResponse(h.messages(tc.name, tc.requested, "Implement this task."), tc.to, tc.finalEffort)
			h.checkRouted(0, tc.name, tc.from, tc.initialEffort)
			h.checkRouted(1, tc.name, tc.to, tc.finalEffort)
			h.checkResponse(h.messages(tc.name, tc.requested, "Continue the same task."), tc.to, tc.finalEffort)
			h.checkRouted(2, tc.name, tc.to, tc.finalEffort)
			if len(h.transport.routed) != 3 || len(h.transport.classified) != 1 {
				t.Fatalf("outbound requests: %d upstream, %d classifier", len(h.transport.routed), len(h.transport.classified))
			}
			entries := h.logged()
			if len(entries) != 2 || entries[0].Key != entries[1].Key {
				t.Fatalf("failover requests = %+v", entries)
			}
			h.checkLogged(entries[0], tc.requested, tc.to, tc.finalEffort, route.ReasonFailover, true)
			h.checkLogged(entries[1], tc.requested, tc.to, tc.finalEffort, route.ReasonPinned, false)
			p, err := h.app.store.Get(t.Context(), entries[0].Key)
			if err != nil || p == nil || p.Model != tc.to || p.Level != tc.level || !p.FailedOver || p.FailedFrom != tc.from {
				t.Fatalf("failed-over pin = %+v, %v", p, err)
			}
		})
	}
}

func TestRoutingPolicyExampleCriteriaReachClassifierAndUpstream(t *testing.T) {
	for _, tc := range []struct {
		name, prompt, model, level, effort string
	}{
		{"quick", "Explain HTTP 204 in one sentence.", "deepseek-flash", "low", "low"},
		{"local", "Fix the form's disabled save button and add a component test; no rush.", "glm-5.3-flash", "medium", "medium"},
		{"routine", "Add pagination and endpoint tests; no rush.", "glm-5.3", "medium", "medium"},
		{"priority", "Fix the cursor race before today's release.", "gpt-6.1-sol", "high", "high"},
		{"hard", "Find why writes across regions intermittently corrupt tenant data.", "claude-opus-5-5", "xhigh", "xhigh"},
		{"substantial-ui", "Design and build a responsive five-screen onboarding flow.", "claude-opus-5-5", "high", "high"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The fixture supplies a known classifier answer; this verifies
			// policy transmission and routing rather than classifier quality.
			h := newPolicyHarness(t, policyVerdict{model: tc.model, level: tc.level})
			h.checkResponse(h.messages(tc.name, "auto", tc.prompt), tc.model, tc.effort)
			h.checkRouted(0, tc.name, tc.model, tc.effort)
			if len(h.transport.classified) != 1 {
				t.Fatalf("classifier requests = %d", len(h.transport.classified))
			}
			request := h.transport.classified[0]
			if request.Model != "jev-latest" || request.State != "images=no\n\n"+tc.prompt || request.Questions["reasoning"].Type != "score" {
				t.Errorf("classifier request = %+v", request)
			}
			question := request.Questions["model"]
			var criteria map[string]struct {
				What     string   `json:"what"`
				NotFor   string   `json:"not_for"`
				Examples []string `json:"examples"`
			}
			if err := json.Unmarshal(question.Criteria, &criteria); err != nil {
				t.Fatal(err)
			}
			wantIDs := []string{"deepseek-flash", "glm-5.3-flash", "glm-5.3", "gpt-6.1-sol", "claude-opus-5-5"}
			if !slices.Equal(h.app.cfg.Auto.CandidateIDs(), wantIDs) || len(criteria) != len(wantIDs) {
				t.Fatalf("candidate ladder = %v, criteria = %v", h.app.cfg.Auto.CandidateIDs(), criteria)
			}
			for _, candidate := range h.app.cfg.Auto.Candidates {
				got, offered := criteria[candidate.ID]
				if !offered || got.What != candidate.Description || got.NotFor != candidate.NotFor || !slices.Equal(got.Examples, candidate.Examples) {
					t.Errorf("classifier criterion for %s = %+v", candidate.ID, got)
				}
			}
			if !strings.Contains(question.Instructions.Focus, "High difficulty and substantial UI changes take precedence over urgency") {
				t.Errorf("classifier focus = %q", question.Instructions.Focus)
			}
			entries := h.logged()
			if len(entries) != 1 {
				t.Fatalf("policy requests = %+v", entries)
			}
			h.checkLogged(entries[0], "auto", tc.model, tc.effort, route.ReasonDecided, false)
			detail, err := h.app.store.Thread(t.Context(), entries[0].Key)
			if err != nil || detail == nil || detail.Decision == nil || detail.Decision.Level != tc.level ||
				detail.Decision.Decider != "jev" || !slices.Equal(detail.Decision.Basis.Candidates, wantIDs) || detail.Policy == nil {
				t.Fatalf("policy decision = %+v, %v", detail, err)
			}
		})
	}
}
