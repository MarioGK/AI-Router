package dashboard

import (
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"

	"github.com/MarioGK/AI-Router/internal/bench"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
)

func TestOperatorsReviewDecisionsByTheirFirstPrompt(t *testing.T) {
	source := history()
	source.detail.Decision.FirstPrompt = "Find the race in the settlement service.\n\nIt double-charges under load."
	d := newDashboard(t, source)
	cookie := signIn(t, d)
	key := source.detail.Key

	body := page(t, d, cookie, "/dashboard/threads/"+key, http.StatusOK)
	contains(t, "thread page", body, "First prompt, as the user wrote it", "It double-charges under load.",
		`id="review"`, "Not reviewed yet", "Right as decided", `/dashboard/sse/threads/`+key+`/review`)

	post := func(signals string) string {
		t.Helper()
		return serve(d, http.MethodPost, "/dashboard/sse/threads/"+key+"/review", signals, cookie,
			"Datastar-Request", "true", "Content-Type", "application/json").Body.String()
	}
	if answer := post(`{"reviewModel":"gpt-6.1-sol","reviewLevel":"","reviewNote":"  Sol finds races  "}`); !strings.Contains(answer, "/dashboard/threads/"+key+"#review") {
		t.Fatalf("a review did not reload the thread:\n%s", answer)
	}
	if review := source.reviews[1]; review.Model != "gpt-6.1-sol" || review.Note != "Sol finds races" || review.Event != 1 {
		t.Fatalf("saved review = %+v", review)
	}
	if answer := post(`{"reviewModel":"gpt-9"}`); !strings.Contains(answer, `id="review-problem"`) || !strings.Contains(answer, "is not a candidate") {
		t.Fatalf("an unknown model was accepted:\n%s", answer)
	}

	source.detail.Decision.Review = &pin.Review{Event: 1, Model: "gpt-6.1-sol", Note: "Sol finds races"}
	body = page(t, d, cookie, "/dashboard/threads/"+key, http.StatusOK)
	contains(t, "reviewed thread page", body, "Should have been", "“Sol finds races”", "Forget the review")

	decisions := page(t, d, cookie, "/dashboard/decisions", http.StatusOK)
	contains(t, "decisions page", decisions, "reviewed", "model right", "most missed:", "claude-opus-5-5 → gpt-6.1-sol",
		"Download as bench cases", "Reviewed wrong")

	export := serve(d, http.MethodGet, "/dashboard/decisions/cases.yaml", "", cookie)
	if export.Code != http.StatusOK || !strings.HasPrefix(export.Header().Get("Content-Disposition"), "attachment") {
		t.Fatalf("export = %d %v", export.Code, export.Header())
	}
	var file struct {
		Cases []bench.Case `yaml:"cases"`
	}
	if err := yaml.Unmarshal(export.Body.Bytes(), &file); err != nil || len(file.Cases) != 1 {
		t.Fatalf("export = %v:\n%s", err, export.Body)
	}
	if c := file.Cases[0]; c.Expect[0] != "gpt-6.1-sol" || c.Note != "Sol finds races" || c.Reasoning != nil ||
		c.Prompt != source.detail.Decision.FirstPrompt || !strings.HasPrefix(c.Name, "review/") {
		t.Fatalf("exported case = %+v", c)
	}

	if post(`{"reviewForget":true}`); len(source.reviews) != 0 {
		t.Fatalf("a forgotten review stayed: %+v", source.reviews)
	}
}

func TestReviewsUseTheRecordedOfferBeforeSpill(t *testing.T) {
	source := history()
	source.detail.Decision.Model = "gpt-6.1-sol"
	source.detail.Decision.Basis.Spilled = "glm-5.3"
	source.detail.Decision.Basis.Candidates = []string{"glm-5.3", "claude-opus-5-5"}
	d := newDashboard(t, source)
	cookie := signIn(t, d)
	key := source.detail.Key
	view := d.reviewOf(key, *source.detail.Decision)
	if view.Model != "glm-5.3" || !slices.Equal(view.Models, []option{{"glm-5.3", "glm-5.3"}, {"claude-opus-5-5", "claude-opus-5-5"}}) {
		t.Fatalf("review choices before spill = %+v", view)
	}
	post := func(signals string) string {
		t.Helper()
		return serve(d, http.MethodPost, "/dashboard/sse/threads/"+key+"/review", signals, cookie,
			"Datastar-Request", "true", "Content-Type", "application/json").Body.String()
	}
	answer := post(`{"reviewModel":"gpt-6.1-sol"}`)
	contains(t, "withheld review", answer, `id="review-problem"`, "offered for this decision", "glm-5.3", "claude-opus-5-5")
	if len(source.reviews) != 0 {
		t.Fatal("a model outside the recorded offer was saved as a review")
	}
	if answer := post(`{"reviewModel":"glm-5.3"}`); !strings.Contains(answer, "/dashboard/threads/"+key+"#review") {
		t.Fatalf("the pre-spill model was rejected: %s", answer)
	}
	export := serve(d, http.MethodGet, "/dashboard/decisions/cases.yaml", "", cookie)
	if export.Code != http.StatusOK {
		t.Fatalf("export = %d: %s", export.Code, export.Body)
	}
	path := filepath.Join(t.TempDir(), "reviews.yaml")
	if err := os.WriteFile(path, export.Body.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	cases, err := bench.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	candidates := make([]decide.Candidate, len(d.auto.Candidates))
	for i, candidate := range d.auto.Candidates {
		candidates[i] = candidate.Criterion()
	}
	if err := bench.Validate(cases, candidates, d.auto.Reasoning.Criteria()); err != nil {
		t.Fatalf("saved review failed benchmark preflight: %v", err)
	}
	if len(cases) != 1 || !slices.Equal(cases[0].Expect, []string{"glm-5.3"}) ||
		!slices.Equal(cases[0].Offered, source.detail.Decision.Basis.Candidates) {
		t.Fatalf("reviewed pre-spill case = %+v", cases)
	}
}

func TestReviewsWithoutRecordedCandidatesUseCurrentCandidates(t *testing.T) {
	source := history()
	source.detail.Decision.Basis.Candidates = nil
	source.detail.Decision.Model = "retired-model"
	d := newDashboard(t, source)
	view := d.reviewOf(source.detail.Key, *source.detail.Decision)
	if !slices.Contains(view.Models, option{"gpt-6.1-sol", "gpt-6.1-sol"}) ||
		slices.Contains(view.Models, option{"retired-model", "retired-model"}) {
		t.Fatalf("review without a recorded offer = %+v", view)
	}
	if err := d.checkReview("gpt-6.1-sol", "", "", *source.detail.Decision); err != nil {
		t.Errorf("current candidate review without a recorded offer = %v", err)
	}
	for _, model := range []string{"unknown-model", "retired-model"} {
		if err := d.checkReview(model, "", "", *source.detail.Decision); err == nil {
			t.Errorf("inactive %s was accepted without a recorded offer", model)
		}
	}
}

func TestReviewsExcludeRetiredRecordedCandidates(t *testing.T) {
	source := history()
	source.detail.Decision.Model = "retired-model"
	source.detail.Decision.Basis.Candidates = []string{"retired-model", "glm-5.3", "claude-opus-5-5"}
	d := newDashboard(t, source)
	view := d.reviewOf(source.detail.Key, *source.detail.Decision)
	if slices.Contains(view.Models, option{"retired-model", "retired-model"}) || view.Model != "glm-5.3" ||
		!strings.Contains(view.Problem, "no longer configured") {
		t.Fatalf("retired recorded model review = %+v", view)
	}
	if err := d.checkReview("retired-model", "", "", *source.detail.Decision); err == nil || !strings.Contains(err.Error(), "not an active candidate") {
		t.Fatalf("retired recorded model review validation = %v", err)
	}
	body := page(t, d, signIn(t, d), "/dashboard/threads/"+source.detail.Key, http.StatusOK)
	if strings.Contains(body, "Right as decided") {
		t.Error("a retired decision model can still be selected by the agreement action")
	}
}

func TestIncompatibleSavedReviewCanBeCorrected(t *testing.T) {
	source := history()
	source.detail.Decision.Basis.Candidates = []string{"glm-5.3", "claude-opus-5-5"}
	source.detail.Decision.Review = &pin.Review{Event: 1, Model: "gpt-6.1-sol"}
	d := newDashboard(t, source)
	view := d.reviewOf(source.detail.Key, *source.detail.Decision)
	if slices.Contains(view.Models, option{"gpt-6.1-sol", "gpt-6.1-sol"}) || view.Model != "claude-opus-5-5" {
		t.Fatalf("incompatible saved review choices = %+v", view)
	}
	body := page(t, d, signIn(t, d), "/dashboard/threads/"+source.detail.Key, http.StatusOK)
	contains(t, "incompatible review", body, "was not offered for this decision", "Choose an offered model", "Forget the review")
}
