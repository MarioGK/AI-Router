package dashboard

import (
	"errors"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/route"
)

func routingDashboard(t *testing.T) *Dashboard {
	t.Helper()
	d := newDashboard(t, history())
	d.auto.ModelID = "work-auto"
	d.auto.Candidates = []config.Candidate{
		{ID: "deepseek-flash", Vision: true, Description: "Tiny explicit edits and direct answers."},
		{ID: "glm-5.3-flash", Vision: true, Description: "Small routine coding and native visual tasks."},
		{ID: "glm-5.3", Description: "Routine work that can wait."},
		{ID: "gpt-6.1-sol", Vision: true, Efforts: []string{"high"}, Description: "Tricky or urgent work."},
		{ID: "claude-opus-5-5", Vision: true, Description: "Difficult work and substantial UI design."},
	}
	d.auto.Retired = []string{"retired-model"}
	return d
}

func TestRoutingPriorityRankingAndFullAutoReset(t *testing.T) {
	d := routingDashboard(t)
	classifier := d.classifier.(*fakeClassifier)
	cookie := signIn(t, d)
	form := url.Values{"mode": {route.ModeAuto}, "models": d.auto.CandidateIDs(), "priority": {decide.PrioritySpeed},
		"rank:claude-opus-5-5": {"1"}, "rank:gpt-6.1-sol": {"2"}, "jev_enabled": {"true"}}
	response := serve(d, http.MethodPost, "/dashboard/routing", form.Encode(), cookie)
	if response.Code != http.StatusSeeOther || classifier.routing.Preferences.Priority != decide.PrioritySpeed ||
		!slices.Equal(classifier.routing.Preferences.Ranking, []string{"claude-opus-5-5", "gpt-6.1-sol"}) {
		t.Fatalf("priority save = %d, %+v", response.Code, classifier.routing)
	}
	body := page(t, d, cookie, "/dashboard/routing", http.StatusOK)
	contains(t, "priority controls", body, "What matters most?", "Enabled models & preferences", "Full auto", `value="speed" checked`,
		`name="rank:claude-opus-5-5"`, `value="1" selected`, `value="2" selected`, `role="tooltip"`)
	for _, test := range []struct{ field, value string }{{"priority", "unknown"}, {"rank:unknown", "1"},
		{"rank:glm-5.3", "1"}, {"rank:retired-model", "1"}, {"rank:glm-5.3", "-1"}, {"rank:glm-5.3", "6"}, {"rank:glm-5.3", "NaN"}} {
		invalid := form.Clone()
		invalid.Set(test.field, test.value)
		before := len(classifier.savedRouting)
		response = serve(d, http.MethodPost, "/dashboard/routing", invalid.Encode(), cookie)
		if response.Code != http.StatusBadRequest || len(classifier.savedRouting) != before {
			t.Fatalf("invalid %s was saved: %d", test.field, response.Code)
		}
	}
	form.Set("reset", "full_auto")
	form.Set("mode", route.ModeFixed)
	form.Set("fixed_model", "gpt-6.1-sol")
	form.Del("jev_enabled")
	response = serve(d, http.MethodPost, "/dashboard/routing", form.Encode(), cookie)
	got := classifier.routing
	if response.Code != http.StatusSeeOther || got.Mode != route.ModeAuto || !got.JevEnabled || got.FixedModel != "" ||
		len(got.Models) != 0 || got.Preferences.Priority != "" || len(got.Preferences.Ranking) != 0 {
		t.Fatalf("full auto retained overrides: %d %+v", response.Code, got)
	}
}

func TestRoutingSettingsShowDefaultPoolAndScopedControls(t *testing.T) {
	d := routingDashboard(t)
	body := page(t, d, signIn(t, d), "/dashboard/routing", http.StatusOK)
	contains(t, "routing settings", body, "Routing for work-auto", "Explicit model requests keep their own route",
		`action="/dashboard/routing"`, `method="post"`, `aria-current="page"`, "All configured models",
		"Enabled models & preferences", "GLM 5.3 Flash", "Enable Jev", "bypasses every decider", "saved across router restarts",
		"GPT-6.1 Sol keeps its configured high reasoning effort", `for="routing-mode-auto"`, `for="routing-fixed-model"`,
		`aria-describedby="routing-group-help"`)
	if count := strings.Count(body, `name="models"`); count != len(d.auto.Candidates) {
		t.Fatalf("group has %d controls, want %d", count, len(d.auto.Candidates))
	}
	if count := strings.Count(body, " checked"); count != len(d.auto.Candidates)+3 {
		t.Fatalf("default group and Jev have %d checked controls, want %d", count, len(d.auto.Candidates)+1)
	}
	if strings.Contains(body, `value="retired-model"`) {
		t.Error("the retired model is selectable in routing controls")
	}
}

func TestRoutingSettingsSaveFixedModeAndAutomaticSubset(t *testing.T) {
	d := routingDashboard(t)
	classifier := d.classifier.(*fakeClassifier)
	cookie := signIn(t, d)
	fixed := url.Values{"mode": {route.ModeFixed}, "fixed_model": {"gpt-6.1-sol"}, "models": d.auto.CandidateIDs()}
	response := serve(d, http.MethodPost, "/dashboard/routing", fixed.Encode(), cookie)
	if response.Code != http.StatusSeeOther || response.Header().Get("Location") != "/dashboard/routing?saved=1" {
		t.Fatalf("fixed settings save = %d %v: %s", response.Code, response.Header(), response.Body)
	}
	if got := classifier.routing; got.Mode != route.ModeFixed || got.FixedModel != "gpt-6.1-sol" ||
		got.JevEnabled || len(got.Models) != 0 {
		t.Fatalf("saved fixed settings = %+v", got)
	}
	body := page(t, d, cookie, response.Header().Get("Location"), http.StatusOK)
	contains(t, "fixed settings", body, "Routing settings saved", "Requests go to", "gpt-6.1-sol",
		"All deciders are bypassed while fixed mode is active", `value="fixed" checked`)

	auto := url.Values{"mode": {route.ModeAuto}, "fixed_model": {""},
		"models": {"claude-opus-5-5", "glm-5.3"}, "jev_enabled": {"true"}}
	response = serve(d, http.MethodPost, "/dashboard/routing", auto.Encode(), cookie)
	if response.Code != http.StatusSeeOther {
		t.Fatalf("automatic settings save = %d: %s", response.Code, response.Body)
	}
	if got := classifier.routing; got.Mode != route.ModeAuto || got.FixedModel != "" || !got.JevEnabled ||
		!slices.Equal(got.Models, []string{"glm-5.3", "claude-opus-5-5"}) {
		t.Fatalf("saved automatic settings = %+v", got)
	}
	body = page(t, d, cookie, "/dashboard/routing", http.StatusOK)
	contains(t, "restricted settings", body, "glm-5.3, claude-opus-5-5", `value="auto" checked`)
	if count := strings.Count(body, " checked"); count != 5 {
		t.Fatalf("restricted group and Jev have %d checked controls, want five", count)
	}

	all := url.Values{"mode": {route.ModeAuto}, "fixed_model": {""}, "models": d.auto.CandidateIDs(), "jev_enabled": {"true"}}
	response = serve(d, http.MethodPost, "/dashboard/routing", all.Encode(), cookie)
	if response.Code != http.StatusSeeOther || len(classifier.routing.Models) != 0 {
		t.Fatalf("selecting all models did not restore the unrestricted default: %d %+v", response.Code, classifier.routing)
	}
}

func TestRoutingSettingsRejectInvalidFormsWithoutSaving(t *testing.T) {
	for _, test := range []struct {
		name, body, contentType string
		status                  int
	}{
		{"unknown mode", "mode=surprise&models=glm-5.3", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"missing mode", "models=glm-5.3", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"repeated mode", "mode=auto&mode=fixed&models=glm-5.3", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"fixed model missing", "mode=fixed", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"retired fixed model", "mode=fixed&fixed_model=retired-model", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"empty automatic group", "mode=auto", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"unknown group model", "mode=auto&models=unknown-model", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"repeated group model", "mode=auto&models=glm-5.3&models=glm-5.3", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"invalid Jev value", "mode=auto&models=glm-5.3&jev_enabled=false", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"unknown field", "mode=auto&models=glm-5.3&unexpected=true", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"malformed encoding", "mode=auto&models=%zz", "application/x-www-form-urlencoded", http.StatusBadRequest},
		{"oversized form", "mode=auto&models=glm-5.3&padding=" + strings.Repeat("x", maxRoutingBody), "application/x-www-form-urlencoded", http.StatusRequestEntityTooLarge},
		{"wrong content type", `{"mode":"fixed"}`, "application/json", http.StatusUnsupportedMediaType},
	} {
		t.Run(test.name, func(t *testing.T) {
			d := routingDashboard(t)
			classifier := d.classifier.(*fakeClassifier)
			response := serve(d, http.MethodPost, "/dashboard/routing", test.body, signIn(t, d), "Content-Type", test.contentType)
			if response.Code != test.status || len(classifier.savedRouting) != 0 {
				t.Fatalf("invalid form = %d, saved=%v: %s", response.Code, classifier.savedRouting, response.Body)
			}
			contains(t, "invalid form", response.Body.String(), `role="alert"`, `id="routing-problem"`, "Active settings")
		})
	}
}

func TestRoutingSettingsRequireSessionAndSameOrigin(t *testing.T) {
	d := routingDashboard(t)
	classifier := d.classifier.(*fakeClassifier)
	form := url.Values{"mode": {route.ModeFixed}, "fixed_model": {"gpt-6.1-sol"}}.Encode()
	for _, method := range []string{http.MethodGet, http.MethodPost} {
		response := serve(d, method, "/dashboard/routing", form, nil)
		if response.Code != http.StatusSeeOther || !strings.HasPrefix(response.Header().Get("Location"), "/dashboard/login?") {
			t.Errorf("unsigned %s routing settings = %d %v", method, response.Code, response.Header())
		}
	}
	cookie := signIn(t, d)
	for _, headers := range [][]string{{"Origin", "https://evil.example"}, {"Sec-Fetch-Site", "same-site"}, {"Sec-Fetch-Site", "cross-site"}} {
		response := serve(d, http.MethodPost, "/dashboard/routing", form, cookie, headers...)
		if response.Code != http.StatusForbidden {
			t.Errorf("cross-origin routing update = %d: %s", response.Code, response.Body)
		}
	}
	if len(classifier.savedRouting) != 0 {
		t.Fatal("an unauthenticated or cross-origin request changed routing settings")
	}
}

func TestRoutingSettingsReportReadAndSaveFailuresSafely(t *testing.T) {
	d := routingDashboard(t)
	classifier := d.classifier.(*fakeClassifier)
	cookie := signIn(t, d)
	classifier.routingErr = errors.New("private-database-detail")
	body := page(t, d, cookie, "/dashboard/routing", http.StatusInternalServerError)
	contains(t, "routing read failure", body, "Routing settings could not be read", "Changes are disabled", "<fieldset disabled")
	if strings.Contains(body, "private-database-detail") {
		t.Error("the settings page exposed a backend error")
	}
	classifier.routingErr = nil
	classifier.routing = route.Settings{Mode: route.ModeFixed, FixedModel: "gpt-6.1-sol", JevEnabled: true}
	classifier.saveRoutingErr = errors.New("private-write-detail")
	form := url.Values{"mode": {route.ModeAuto}, "fixed_model": {""}, "models": d.auto.CandidateIDs()}.Encode()
	response := serve(d, http.MethodPost, "/dashboard/routing", form, cookie)
	if response.Code != http.StatusInternalServerError || len(classifier.savedRouting) != 0 {
		t.Fatalf("routing save failure = %d, saved=%v", response.Code, classifier.savedRouting)
	}
	contains(t, "routing save failure", response.Body.String(), "Your previous settings are still active", "Requests go to", "gpt-6.1-sol")
	if strings.Contains(response.Body.String(), "private-write-detail") || strings.Count(response.Body.String(), " checked") != len(d.auto.Candidates)+2 {
		t.Error("the failed save exposed its backend error or lost the submitted all-model selection")
	}
}
