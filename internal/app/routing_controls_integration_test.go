package app

import (
	"encoding/json/v2"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/route"
)

type routingJudgeRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
}

// routingControlsTransport adds the configured LLM fallback to the existing
// socket-free peers. Both deciders still run their production wire adapters.
type routingControlsTransport struct {
	base   *policyTransport
	judged []routingJudgeRequest
}

func (p *routingControlsTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.Host != "upstream.test" || r.URL.Path != "/v1/chat/completions" {
		return p.base.RoundTrip(r)
	}
	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	var request routingJudgeRequest
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, err
	}
	p.judged = append(p.judged, request)
	answer, err := json.Marshal(map[string]any{
		"probabilities": map[string]float64{"glm-5.3": 1},
		"reasoning":     map[string]float64{"medium": 1},
		"reason":        "The task fits the routine tier.",
	})
	if err != nil {
		return nil, err
	}
	return p.base.response(r, http.StatusOK, map[string]any{
		"model": request.Model,
		"choices": []map[string]any{{"message": map[string]string{
			"role": "assistant", "content": string(answer),
		}}},
		"usage": map[string]int{"prompt_tokens": 13, "completion_tokens": 5},
	})
}

func newRoutingControlsHarness(t *testing.T, verdict policyVerdict, configure ...func(*config.Config)) (*policyHarness, *routingControlsTransport) {
	t.Helper()
	customize := append([]func(*config.Config){func(cfg *config.Config) {
		cfg.Deciders = append(cfg.Deciders, config.Decider{
			Name: "fallback-judge", Type: config.LLM, URL: "http://upstream.test/v1/chat/completions",
			Model: "deepseek-flash", APIKeyEnv: "CLIPROXYAPI_API_KEY", Timeout: time.Second,
		})
	}}, configure...)
	h := newPolicyHarness(t, verdict, customize...)
	transport := &routingControlsTransport{base: h.transport}
	restartRoutingControls(t, h, transport)
	return h, transport
}

func restartRoutingControls(t *testing.T, h *policyHarness, transport *routingControlsTransport) {
	t.Helper()
	cfg, logger := h.app.cfg, h.app.log
	if err := h.app.Close(); err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: transport}
	a, err := newWithClients(cfg, logger, client, client)
	if err != nil {
		t.Fatal(err)
	}
	h.app = a
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Error(err)
		}
	})
	if err := a.catalog.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func saveRoutingControls(t *testing.T, h *policyHarness, settings route.Settings) route.Settings {
	t.Helper()
	saved, err := h.app.resolver.SetSettings(t.Context(), settings)
	if err != nil {
		t.Fatal(err)
	}
	return saved
}

func routingControlBody(model, prompt string) string {
	return fmt.Sprintf(`{"model":%q,"max_tokens":1024,"messages":[{"role":"user","content":%q}],%s}`,
		model, prompt, tools)
}

func checkRoutingGroup(t *testing.T, request policyClassifierRequest, models []string) {
	t.Helper()
	var criteria map[string]any
	if err := json.Unmarshal(request.Questions["model"].Criteria, &criteria); err != nil {
		t.Fatal(err)
	}
	if got, want := slices.Sorted(maps.Keys(criteria)), slices.Sorted(slices.Values(models)); !slices.Equal(got, want) {
		t.Errorf("classifier group = %v, want %v", got, want)
	}
}

func checkRoutingUnavailable(t *testing.T, w *httptest.ResponseRecorder) {
	t.Helper()
	var response struct {
		Type  string `json:"type"`
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || w.Code != http.StatusServiceUnavailable ||
		response.Type != "error" || response.Error.Type != "model_unavailable_error" {
		t.Fatalf("unavailable model = %d %s, %v", w.Code, w.Body.String(), err)
	}
}

func TestRoutingControlsFixedSolBypassesEveryClassifier(t *testing.T) {
	for _, requested := range []string{"auto(low)", "auto(xhigh)", "auto(8192)"} {
		t.Run(requested, func(t *testing.T) {
			h, transport := newRoutingControlsHarness(t, policyVerdict{model: "glm-5.3", level: "medium"})
			saveRoutingControls(t, h, route.Settings{Mode: route.ModeFixed, FixedModel: "gpt-6.1-sol", JevEnabled: true})
			h.checkResponse(h.messages("fixed-control", requested, "Add an ordinary endpoint."), "gpt-6.1-sol", "high")
			h.checkRouted(0, "fixed-control", "gpt-6.1-sol", "high")
			if len(h.transport.classified) != 0 || len(transport.judged) != 0 {
				t.Fatalf("fixed classifiers: Jev %d, LLM %d", len(h.transport.classified), len(transport.judged))
			}
			page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
			if err != nil || page.Total != 0 {
				t.Fatalf("fixed mode created a pin: %+v, %v", page, err)
			}
			entries := h.logged()
			if len(entries) != 1 || entries[0].Requested != requested || entries[0].Model != "gpt-6.1-sol" ||
				entries[0].Effort != "high" || entries[0].Reason != route.ReasonFixed || entries[0].Failover ||
				entries[0].Status != http.StatusOK || entries[0].Tokens.Input != 11 || entries[0].Tokens.Output != 7 {
				t.Fatalf("fixed request log = %+v", entries)
			}
		})
	}
}

func TestPrioritySurvivesRestartReachesJevAndResetsForNewThreads(t *testing.T) {
	h, transport := newRoutingControlsHarness(t, policyVerdict{model: "glm-5.3", level: "medium"})
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeAuto, JevEnabled: true,
		Preferences: decide.Preferences{Priority: decide.PrioritySpeed, Ranking: []string{"gpt-6.1-sol", "glm-5.3"}}})
	restartRoutingControls(t, h, transport)
	h.checkResponse(h.messages("priority-control", "auto", "Add a routine endpoint."), "glm-5.3", "medium")
	request := h.transport.classified[0]
	if !strings.Contains(request.Questions["model"].Instructions.Focus, "Operator priority: speed") ||
		!strings.Contains(request.Questions["model"].Instructions.Focus, "gpt-6.1-sol -> glm-5.3") {
		t.Fatalf("persisted priority did not reach Jev: %+v", request)
	}
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeAuto, JevEnabled: true})
	h.checkResponse(h.messages("priority-control", "auto", "Continue."), "glm-5.3", "medium")
	if len(h.transport.classified) != 1 {
		t.Fatal("reset reclassified an existing thread")
	}
	h.checkResponse(h.messages("full-auto-new", "auto", "Add another routine endpoint."), "glm-5.3", "medium")
	if len(h.transport.classified) != 2 || strings.Contains(h.transport.classified[1].Questions["model"].Instructions.Focus, "Operator priority") {
		t.Fatal("full auto retained an override for new threads")
	}
}

func TestRoutingControlsFixedModePreservesAnExistingPin(t *testing.T) {
	h, transport := newRoutingControlsHarness(t, policyVerdict{model: "glm-5.3", level: "medium"})
	h.checkResponse(h.messages("existing-control", "auto", "Add a routine endpoint."), "glm-5.3", "medium")
	page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || len(page.Threads) != 1 {
		t.Fatalf("initial pin = %+v, %v", page, err)
	}
	before := page.Threads[0].Pin
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeFixed, FixedModel: "gpt-6.1-sol", JevEnabled: true})
	h.checkResponse(h.messages("existing-control", "auto(low)", "Continue the same task."), "gpt-6.1-sol", "high")
	after, err := h.app.store.Get(t.Context(), before.Key)
	if err != nil || after == nil || *after != before {
		t.Fatalf("fixed mode changed the pin: before %+v, after %+v, %v", before, after, err)
	}
	if len(h.transport.classified) != 1 || len(transport.judged) != 0 {
		t.Fatalf("fixed existing classifiers: Jev %d, LLM %d", len(h.transport.classified), len(transport.judged))
	}
	entries := h.logged()
	if len(entries) != 2 || entries[1].Key != before.Key || entries[1].Reason != route.ReasonFixed {
		t.Fatalf("existing fixed request log = %+v", entries)
	}
}

func TestRoutingControlsFixedFailureDoesNotFailOver(t *testing.T) {
	h, transport := newRoutingControlsHarness(t, policyVerdict{model: "glm-5.3", level: "medium"})
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeFixed, FixedModel: "gpt-6.1-sol", JevEnabled: true})
	h.transport.reject = "gpt-6.1-sol"
	w := h.request("fixed-failure", routingControlBody("auto(low)", "Continue this task."))
	if w.Code != http.StatusTooManyRequests || len(h.transport.routed) != 1 ||
		len(h.transport.classified) != 0 || len(transport.judged) != 0 {
		t.Fatalf("fixed failure = %d %s, outbound %d/%d/%d", w.Code, w.Body.String(),
			len(h.transport.routed), len(h.transport.classified), len(transport.judged))
	}
	h.checkRouted(0, "fixed-failure", "gpt-6.1-sol", "high")
	entries := h.logged()
	if len(entries) != 1 || entries[0].Status != http.StatusTooManyRequests || entries[0].Failover || entries[0].Model != "gpt-6.1-sol" {
		t.Fatalf("fixed failure log = %+v", entries)
	}
}

func TestRoutingControlsGroupRestrictsClassificationAndFailover(t *testing.T) {
	for _, test := range []struct {
		name  string
		group []string
		final string
	}{
		{"nearest-allowed-default", []string{"glm-5.3", "claude-opus-5-5"}, "claude-opus-5-5"},
		{"configured-default-allowed", []string{"glm-5.3", "gpt-6.1-sol"}, "gpt-6.1-sol"},
		{"single-model", []string{"glm-5.3"}, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			h, _ := newRoutingControlsHarness(t, policyVerdict{model: "glm-5.3", level: "medium"}, func(cfg *config.Config) {
				// The configured fallback is outside every selected group;
				// only an eligible default within the group may replace it.
				cfg.Auto.Fallback["glm-5.3"] = "deepseek-flash"
			})
			status, routed, final := http.StatusTooManyRequests, 1, "glm-5.3"
			if test.final != "" {
				status, routed, final = http.StatusOK, 2, test.final
			}
			saveRoutingControls(t, h, route.Settings{Mode: route.ModeAuto, Models: test.group, JevEnabled: true})
			h.transport.reject = "glm-5.3"
			w := h.request("group-failure", routingControlBody("auto", "Add a routine endpoint."))
			if w.Code != status || len(h.transport.routed) != routed {
				t.Fatalf("group failure = %d %s, routed %+v", w.Code, w.Body.String(), h.transport.routed)
			}
			h.checkRouted(0, "group-failure", "glm-5.3", "medium")
			if test.final != "" {
				h.checkRouted(1, "group-failure", final, "high")
			}
			for _, request := range h.transport.routed {
				if !slices.Contains(test.group, config.Bare(request.Model)) {
					t.Fatalf("failover escaped the selected model group: %s", request.Model)
				}
			}
			if len(h.transport.classified) != 1 {
				t.Fatalf("group classifier calls = %d", len(h.transport.classified))
			}
			checkRoutingGroup(t, h.transport.classified[0], test.group)
			page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
			if err != nil || len(page.Threads) != 1 || page.Threads[0].Model != final || page.Threads[0].FailedOver != (test.final != "") {
				t.Fatalf("group pin = %+v, %v", page, err)
			}
		})
	}
}

func TestRoutingControlsGroupConstrainsSpill(t *testing.T) {
	for _, allowed := range []bool{true, false} {
		t.Run(fmt.Sprintf("sol-allowed-%t", allowed), func(t *testing.T) {
			h, _ := newRoutingControlsHarness(t, policyVerdict{model: "glm-5.3", level: "medium"}, func(cfg *config.Config) {
				settings := cfg.Limits["glm-5.3"]
				settings.Concurrency, settings.QueueTimeout = 1, time.Second
				cfg.Limits["glm-5.3"] = settings
			})
			group := []string{"glm-5.3", "claude-opus-5-5"}
			if allowed {
				group = append(group, "gpt-6.1-sol")
			}
			saveRoutingControls(t, h, route.Settings{Mode: route.ModeAuto, Models: group, JevEnabled: true})
			release, err := h.app.limits.Gate("glm-5.3").Acquire(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			defer release()
			w := h.request("group-spill", routingControlBody("auto(low)", "Add ordinary endpoint tests."))
			wantModel := "glm-5.3"
			if allowed {
				wantModel = "gpt-6.1-sol"
				if w.Code != http.StatusOK || len(h.transport.routed) != 1 {
					t.Fatalf("allowed spill = %d %s, routed %+v", w.Code, w.Body.String(), h.transport.routed)
				}
				h.checkResponse(w, wantModel, "high")
				h.checkRouted(0, "group-spill", wantModel, "high")
			} else if w.Code != http.StatusTooManyRequests || len(h.transport.routed) != 0 {
				t.Fatalf("excluded spill = %d %s, routed %+v", w.Code, w.Body.String(), h.transport.routed)
			}
			if len(h.transport.classified) != 1 {
				t.Fatalf("spill classifier calls = %d", len(h.transport.classified))
			}
			checkRoutingGroup(t, h.transport.classified[0], group)
			page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
			if err != nil || len(page.Threads) != 1 || page.Threads[0].Model != wantModel || page.Threads[0].Decision == nil {
				t.Fatalf("spill pin = %+v, %v", page, err)
			}
			spilled := page.Threads[0].Decision.Basis.Spilled
			if allowed && spilled != "glm-5.3" || !allowed && spilled != "" {
				t.Fatalf("spill basis = %q, allowed %t", spilled, allowed)
			}
		})
	}
}

func TestRoutingControlsGroupMovesExistingPinWithoutReclassifying(t *testing.T) {
	h, _ := newRoutingControlsHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "low"})
	h.checkResponse(h.messages("group-existing", "auto", "Solve the cursor race."), "gpt-6.1-sol", "high")
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeAuto, Models: []string{"glm-5.3", "claude-opus-5-5"}, JevEnabled: true})
	h.checkResponse(h.messages("group-existing", "auto", "Continue the existing task."), "glm-5.3", "medium")
	if len(h.transport.classified) != 1 {
		t.Fatalf("group migration reclassified the thread: %d calls", len(h.transport.classified))
	}
	page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || len(page.Threads) != 1 || page.Threads[0].Model != "glm-5.3" || page.Threads[0].Level != "low" || page.Threads[0].FailedOver {
		t.Fatalf("group migrated pin = %+v, %v", page, err)
	}
}

func TestRoutingControlsJevToggleFallsThroughConfiguredJudge(t *testing.T) {
	h, transport := newRoutingControlsHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "low"})
	settings := route.Settings{Mode: route.ModeAuto, Models: []string{"glm-5.3", "gpt-6.1-sol"}, JevEnabled: false}
	saveRoutingControls(t, h, settings)
	h.checkResponse(h.messages("jev-disabled", "auto", "Add routine endpoint tests."), "glm-5.3", "medium")
	if len(h.transport.classified) != 0 || len(transport.judged) != 1 {
		t.Fatalf("disabled Jev calls: Jev %d, LLM %d", len(h.transport.classified), len(transport.judged))
	}
	judge := transport.judged[0]
	if judge.Model != "deepseek-flash" || len(judge.Messages) != 2 ||
		judge.Messages[0].Role != "system" || judge.Messages[1].Role != "user" ||
		judge.Messages[1].Content != "images=no\n\nAdd routine endpoint tests." ||
		strings.Contains(judge.Messages[0].Content, "- claude-opus-5-5:") ||
		!strings.Contains(judge.Messages[0].Content, "- glm-5.3:") || !strings.Contains(judge.Messages[0].Content, "- gpt-6.1-sol:") {
		t.Fatalf("configured judge request = %+v", judge)
	}
	settings.JevEnabled = true
	saveRoutingControls(t, h, settings)
	h.checkResponse(h.messages("jev-enabled", "auto(xhigh)", "Solve the cursor race."), "gpt-6.1-sol", "high")
	h.checkResponse(h.messages("jev-disabled", "auto", "Continue the existing routine task."), "glm-5.3", "medium")
	if len(h.transport.classified) != 1 || len(transport.judged) != 1 {
		t.Fatalf("re-enabled Jev calls: Jev %d, LLM %d", len(h.transport.classified), len(transport.judged))
	}
	entries := h.logged()
	if len(entries) != 3 {
		t.Fatalf("toggle request logs = %+v", entries)
	}
	detail, err := h.app.store.Thread(t.Context(), entries[0].Key)
	if err != nil || detail == nil || detail.Decision == nil || detail.Decision.Decider != "fallback-judge" {
		t.Fatalf("disabled Jev decision = %+v, %v", detail, err)
	}
}

func TestRoutingControlsPersistAcrossAppRestart(t *testing.T) {
	for _, mode := range []string{route.ModeAuto, route.ModeFixed} {
		t.Run(mode, func(t *testing.T) {
			h, transport := newRoutingControlsHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "low"})
			settings := route.Settings{Mode: mode, JevEnabled: false}
			wantModel, wantEffort := "glm-5.3", "medium"
			if mode == route.ModeFixed {
				settings.FixedModel = "gpt-6.1-sol"
				wantModel, wantEffort = settings.FixedModel, "high"
			} else {
				settings.Models = []string{"glm-5.3", "claude-opus-5-5"}
			}
			saved := saveRoutingControls(t, h, settings)
			restartRoutingControls(t, h, transport)
			loaded, err := h.app.resolver.Settings(t.Context())
			if err != nil || loaded.Mode != saved.Mode || loaded.FixedModel != saved.FixedModel ||
				loaded.JevEnabled != saved.JevEnabled || !slices.Equal(loaded.Models, saved.Models) {
				t.Fatalf("restarted settings = %+v, want %+v, %v", loaded, saved, err)
			}
			requested := "auto"
			if mode == route.ModeFixed {
				requested = "auto(8192)"
			}
			h.checkResponse(h.messages("after-restart", requested, "Add routine endpoint tests."), wantModel, wantEffort)
			if len(h.transport.classified) != 0 || mode == route.ModeAuto && len(transport.judged) != 1 ||
				mode == route.ModeFixed && len(transport.judged) != 0 {
				t.Fatalf("restarted classifiers: Jev %d, LLM %d", len(h.transport.classified), len(transport.judged))
			}
		})
	}
}

func TestRoutingControlsDashboardSaveChangesRuntimeAndSurvivesRestart(t *testing.T) {
	t.Setenv("TS_ROUTER_DASHBOARD_PASSWORD_HASH", dashboardPasswordHash)
	h, transport := newRoutingControlsHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "low"})
	form := url.Values{
		"mode": {route.ModeFixed}, "fixed_model": {"gpt-6.1-sol"},
		"models": {"glm-5.3", "claude-opus-5-5"},
	}
	w := appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/routing", form)
	if w.Code != http.StatusSeeOther || !strings.HasPrefix(w.Header().Get("Location"), "/dashboard/login") {
		t.Fatalf("unsigned settings save = %d %s", w.Code, w.Body.String())
	}
	settings, err := h.app.resolver.Settings(t.Context())
	if err != nil || settings.Mode != route.ModeAuto || !settings.JevEnabled || len(settings.Models) != 0 {
		t.Fatalf("unsigned save changed settings: %+v, %v", settings, err)
	}
	w = appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/login", url.Values{
		"username": {dashboardUsername}, "password": {dashboardPassword},
	})
	cookies := w.Result().Cookies()
	if w.Code != http.StatusSeeOther || len(cookies) != 1 {
		t.Fatalf("settings operator login = %d %s", w.Code, w.Body.String())
	}
	session := cookies[0]
	w = appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/routing", form, session)
	if w.Code != http.StatusSeeOther || w.Header().Get("Location") != "/dashboard/routing?saved=1" {
		t.Fatalf("fixed settings save = %d %s", w.Code, w.Body.String())
	}
	h.checkResponse(h.messages("dashboard-fixed", "auto(low)", "Add routine endpoint tests."), "gpt-6.1-sol", "high")
	if len(h.transport.classified) != 0 || len(transport.judged) != 0 {
		t.Fatal("dashboard fixed setting still called a classifier")
	}
	form.Set("mode", route.ModeAuto)
	w = appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/routing", form, session)
	if w.Code != http.StatusSeeOther || w.Header().Get("Location") != "/dashboard/routing?saved=1" {
		t.Fatalf("automatic settings save = %d %s", w.Code, w.Body.String())
	}
	h.checkResponse(h.messages("dashboard-auto", "auto", "Add routine endpoint tests."), "glm-5.3", "medium")
	if len(h.transport.classified) != 0 || len(transport.judged) != 1 {
		t.Fatalf("dashboard automatic classifiers: Jev %d, LLM %d", len(h.transport.classified), len(transport.judged))
	}
	restartRoutingControls(t, h, transport)
	form.Set("mode", route.ModeFixed)
	form.Set("fixed_model", "glm-5.3")
	w = appDashboardRequest(t, h.app, http.MethodPost, "/dashboard/routing", form, session)
	if w.Code != http.StatusSeeOther || !strings.HasPrefix(w.Header().Get("Location"), "/dashboard/login") {
		t.Fatalf("pre-restart session saved settings: %d %s", w.Code, w.Body.String())
	}
	settings, err = h.app.resolver.Settings(t.Context())
	if err != nil || settings.Mode != route.ModeAuto || settings.JevEnabled ||
		!slices.Equal(settings.Models, []string{"glm-5.3", "claude-opus-5-5"}) {
		t.Fatalf("dashboard settings after restart = %+v, %v", settings, err)
	}
	h.checkResponse(h.messages("dashboard-restarted", "auto", "Add routine endpoint tests."), "glm-5.3", "medium")
	if len(h.transport.classified) != 0 || len(transport.judged) != 2 {
		t.Fatalf("restarted dashboard classifiers: Jev %d, LLM %d", len(h.transport.classified), len(transport.judged))
	}
}

func TestRoutingControlsIncompatibleImagesReturn503(t *testing.T) {
	for _, mode := range []string{route.ModeAuto, route.ModeFixed} {
		t.Run(mode, func(t *testing.T) {
			h, transport := newRoutingControlsHarness(t, policyVerdict{model: "glm-5.3", level: "medium"})
			saveRoutingControls(t, h, route.Settings{Mode: mode, FixedModel: "glm-5.3", Models: []string{"glm-5.3"}, JevEnabled: true})
			w := h.request("incompatible-image", screenshot)
			checkRoutingUnavailable(t, w)
			if len(h.transport.classified) != 0 || len(transport.judged) != 0 || len(h.transport.routed) != 0 {
				t.Fatalf("incompatible image outbound: Jev %d, LLM %d, model %d",
					len(h.transport.classified), len(transport.judged), len(h.transport.routed))
			}
			if entries := h.logged(); len(entries) != 0 {
				t.Fatalf("incompatible image routed logs = %+v", entries)
			}
		})
	}
}

func TestRoutingControlsFixedModelHonorsAnAutomaticImageThread(t *testing.T) {
	h, transport := newRoutingControlsHarness(t, policyVerdict{model: "gpt-6.1-sol", level: "low"})
	h.checkResponse(h.request("auto-image-fixed", screenshot), "gpt-6.1-sol", "high")
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeFixed, FixedModel: "glm-5.3", JevEnabled: true})
	checkRoutingUnavailable(t, h.request("auto-image-fixed", routingControlBody("auto", "Continue without another image.")))
	if len(h.transport.routed) != 1 || len(h.transport.classified) != 1 || len(transport.judged) != 0 {
		t.Fatalf("fixed mode lost the image constraint: upstream %d, Jev %d, LLM %d",
			len(h.transport.routed), len(h.transport.classified), len(transport.judged))
	}
}

func TestRoutingControlsFixedImagesRemainRequiredAcrossModeChangeAndRestart(t *testing.T) {
	h, transport := newRoutingControlsHarness(t, policyVerdict{model: "glm-5.3", level: "medium"})
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeFixed, FixedModel: "gpt-6.1-sol", JevEnabled: true})
	h.checkResponse(h.request("fixed-image-only", screenshot), "gpt-6.1-sol", "high")
	page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || page.Total != 0 {
		t.Fatalf("fixed image created an automatic pin: %+v, %v", page, err)
	}
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeAuto, Models: []string{"glm-5.3"}, JevEnabled: true})
	checkRoutingUnavailable(t, h.request("fixed-image-only", routingControlBody("auto", "Continue without another image.")))
	restartRoutingControls(t, h, transport)
	checkRoutingUnavailable(t, h.request("fixed-image-only", routingControlBody("auto", "Continue after the restart.")))
	if len(h.transport.routed) != 1 || len(h.transport.classified) != 0 || len(transport.judged) != 0 {
		t.Fatalf("fixed image requirement was forgotten: upstream %d, Jev %d, LLM %d",
			len(h.transport.routed), len(h.transport.classified), len(transport.judged))
	}
	page, err = h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || page.Total != 0 {
		t.Fatalf("incompatible automatic turn created a pin: %+v, %v", page, err)
	}
}

func TestRoutingControlsFixedImagesPreserveAutomaticPinAndItsVisionRequirement(t *testing.T) {
	h, transport := newRoutingControlsHarness(t, policyVerdict{model: "glm-5.3", level: "medium"})
	h.checkResponse(h.messages("fixed-image-existing", "auto", "Add routine endpoint tests."), "glm-5.3", "medium")
	page, err := h.app.store.Threads(t.Context(), pin.ThreadFilter{})
	if err != nil || len(page.Threads) != 1 {
		t.Fatalf("initial automatic pin = %+v, %v", page, err)
	}
	before := page.Threads[0].Pin
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeFixed, FixedModel: "gpt-6.1-sol", JevEnabled: true})
	h.checkResponse(h.request("fixed-image-existing", screenshot), "gpt-6.1-sol", "high")
	after, err := h.app.store.Get(t.Context(), before.Key)
	if err != nil || after == nil || *after != before {
		t.Fatalf("fixed image changed the automatic pin: before %+v, after %+v, %v", before, after, err)
	}
	saveRoutingControls(t, h, route.Settings{Mode: route.ModeAuto, Models: []string{"glm-5.3"}, JevEnabled: true})
	restartRoutingControls(t, h, transport)
	checkRoutingUnavailable(t, h.request("fixed-image-existing", routingControlBody("auto", "Continue after the restart without another image.")))
	if len(h.transport.routed) != 2 || len(h.transport.classified) != 1 || len(transport.judged) != 0 {
		t.Fatalf("existing thread lost fixed image requirement: upstream %d, Jev %d, LLM %d",
			len(h.transport.routed), len(h.transport.classified), len(transport.judged))
	}
	after, err = h.app.store.Get(t.Context(), before.Key)
	if err != nil || after == nil || after.Model != before.Model || after.Level != before.Level {
		t.Fatalf("incompatible automatic turn changed its pin: %+v, %v", after, err)
	}
}
