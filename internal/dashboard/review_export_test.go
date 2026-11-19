package dashboard

import (
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/bench"
	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
)

func TestReviewExportPreservesBenchmarkInputs(t *testing.T) {
	all := []config.Candidate{
		{ID: "deepseek-flash", Vision: true},
		{ID: "glm-5.3"},
		{ID: "gpt-6.1-sol", Vision: true},
		{ID: "claude-opus-5-5", Vision: true},
	}
	source := &fakeSource{
		decisions: []pin.Event{
			{
				ID: 1, At: time.Date(2026, time.September, 29, 12, 0, 0, 0, time.UTC),
				Model: "gpt-6.1-sol", Level: "medium",
				FirstPrompt: "Compare the attached checkout error with our UI.",
				Prompt:      "images=yes\n\nCompare the attached checkout error with our UI.",
				Basis:       pin.Basis{Images: true, Candidates: []string{"gpt-6.1-sol", "claude-opus-5-5"}},
			},
			{
				ID: 2, At: time.Date(2026, time.September, 29, 13, 0, 0, 0, time.UTC),
				Model: "glm-5.3", Level: "medium",
				FirstPrompt: "Add validation for the email field in our existing form.",
				Prompt:      "images=no\n\nAdd validation for the email field in our existing form.",
				Basis:       pin.Basis{Candidates: []string{"glm-5.3", "claude-opus-5-5"}},
			},
		},
		reviews: map[int64]pin.Review{
			1: {Event: 1, Model: "gpt-6.1-sol", Level: "medium", Note: "The screenshot requires vision."},
			2: {Event: 2, Model: "glm-5.3", Level: "medium", Note: "Routine form validation."},
		},
	}
	d := newDashboard(t, source)
	d.auto.Candidates = all
	d.auto.Reasoning.Levels = []config.Level{{ID: "medium"}}
	export := serve(d, http.MethodGet, "/dashboard/decisions/cases.yaml", "", signIn(t, d))
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
	if len(cases) != len(source.decisions) {
		t.Fatalf("loaded %d cases, want %d", len(cases), len(source.decisions))
	}
	for i, c := range cases {
		e := source.decisions[i]
		review := source.reviews[e.ID]
		if c.Images != e.Basis.Images || !slices.Equal(c.Offered, e.Basis.Candidates) ||
			c.Prompt != e.FirstPrompt || !slices.Equal(c.Expect, []string{review.Model}) ||
			!slices.Equal(c.Reasoning, []string{review.Level}) || c.Note != review.Note {
			t.Errorf("loaded case %d lost decision or review data: %+v", i, c)
		}
	}
	candidates := make([]decide.Candidate, len(all))
	for i, candidate := range all {
		candidates[i] = candidate.Criterion()
	}
	if err := bench.Validate(cases, candidates, d.auto.Reasoning.Criteria()); err != nil {
		t.Fatalf("export failed benchmark preflight: %v", err)
	}

	builder := decide.NewStateBuilder(nil, 12_000)
	inputs, err := bench.Prepare(cases, func(c bench.Case) bench.Input {
		input := bench.Input{State: builder.Build(c.Prompt, c.Images).Text}
		for _, candidate := range all {
			if (!c.Images || candidate.Vision) && (len(c.Offered) == 0 || slices.Contains(c.Offered, candidate.ID)) {
				input.Candidates = append(input.Candidates, candidate.Criterion())
			}
		}
		return input
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, input := range inputs {
		e := source.decisions[i]
		if input.State != e.Prompt {
			t.Errorf("case %d prepared state = %q, want recorded %q", i, input.State, e.Prompt)
		}
		ids := make([]string, len(input.Candidates))
		for j, candidate := range input.Candidates {
			ids[j] = candidate.ID
		}
		if !slices.Equal(ids, e.Basis.Candidates) {
			t.Errorf("case %d prepared candidates = %v, want recorded %v", i, ids, e.Basis.Candidates)
		}
	}
}

func TestReviewExportRejectsUnreplayableReviews(t *testing.T) {
	source := history()
	source.detail.Decision.Basis.Candidates = []string{"glm-5.3", "claude-opus-5-5"}
	source.reviews = map[int64]pin.Review{
		1: {Event: 1, Model: "retired-model"},
		3: {Event: 3, Model: "glm-5.3"},
	}
	d := newDashboard(t, source)
	export := serve(d, http.MethodGet, "/dashboard/decisions/cases.yaml", "", signIn(t, d))
	if export.Code != http.StatusConflict || export.Header().Get("Content-Disposition") != "" {
		t.Fatalf("unreplayable export = %d %v: %s", export.Code, export.Header(), export.Body)
	}
	contains(t, "export diagnostic", export.Body.String(), "cannot be replayed", "review/", "retired-model", "not offered",
		"Correct or forget")
}

func TestHistoricalReviewExportsRemainDownloadable(t *testing.T) {
	source := history()
	source.detail.Decision.Basis.Candidates = []string{"glm-5.3", "retired-model"}
	source.reviews = map[int64]pin.Review{1: {Event: 1, Model: "glm-5.3", Level: "retired-level"}}
	d := newDashboard(t, source)
	export := serve(d, http.MethodGet, "/dashboard/decisions/cases.yaml", "", signIn(t, d))
	if export.Code != http.StatusOK {
		t.Fatalf("historical export = %d: %s", export.Code, export.Body)
	}
	path := filepath.Join(t.TempDir(), "historical-reviews.yaml")
	if err := os.WriteFile(path, export.Body.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	cases, err := bench.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	current := make([]decide.Candidate, len(d.auto.Candidates))
	for i, candidate := range d.auto.Candidates {
		current[i] = candidate.Criterion()
	}
	if err := bench.Validate(cases, current, d.auto.Reasoning.Criteria()); err == nil {
		t.Fatal("current benchmark configuration silently accepted retired candidates")
	}
	historical := []decide.Candidate{{ID: "glm-5.3"}, {ID: "retired-model"}}
	if err := bench.Validate(cases, historical, []decide.Level{{ID: "retired-level"}}); err != nil {
		t.Fatalf("matching historical benchmark configuration rejected export: %v", err)
	}
	if len(cases) != 1 || !slices.Equal(cases[0].Offered, source.detail.Decision.Basis.Candidates) ||
		!slices.Equal(cases[0].Reasoning, []string{"retired-level"}) {
		t.Fatalf("historical review metadata = %+v", cases)
	}
}
