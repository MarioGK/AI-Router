package dashboard

import (
	"cmp"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/starfederation/datastar-go/datastar"
	"go.yaml.in/yaml/v3"

	"github.com/MarioGK/AI-Router/internal/bench"
	"github.com/MarioGK/AI-Router/internal/pin"
)

const (
	// maxReviewBody bounds a review request's body: a model, a level and a
	// note.
	maxReviewBody = 16 << 10
	// maxNote bounds a review's note, in runes.
	maxNote = 2000
)

// reviewOptions filter decisions by review.
var reviewOptions = []option{{pin.Unreviewed, "Unreviewed"}, {pin.Right, "Reviewed right"}, {pin.Wrong, "Reviewed wrong"}}

// reviewView is the review card of a thread's decision: what it decided,
// its review, and the form's choices, which start from the review or else
// from the decision.
type reviewView struct {
	Key                   string
	Decided, DecidedLevel string
	Review                *pin.Review
	Model, Level, Note    string
	Models, Levels        []option
	Problem               string
	CanAgree              bool
}

func (d *Dashboard) reviewOf(key string, decision pin.Event) reviewView {
	view := reviewView{Key: key, Decided: decision.Decided(), DecidedLevel: decision.Level, Review: decision.Review,
		Model: decision.Decided(), Level: decision.Level}
	if r := decision.Review; r != nil {
		view.Model, view.Level, view.Note = r.Model, r.Level, r.Note
	}
	models := d.reviewModels(decision)
	view.CanAgree = slices.Contains(models, view.Decided)
	for _, model := range models {
		view.Models = append(view.Models, option{model, model})
	}
	if !slices.Contains(models, view.Model) {
		switch {
		case len(models) == 0:
			view.Problem = "None of this decision's offered models is still configured. Existing reviews can be forgotten or exported for replay with a matching configuration."
		case len(decision.Basis.Candidates) > 0 && !slices.Contains(decision.Basis.Candidates, view.Model):
			view.Problem = fmt.Sprintf("%q was not offered for this decision. Choose an offered model to correct the review.", view.Model)
		default:
			view.Problem = fmt.Sprintf("%q is no longer configured. Choose an offered model still in service to correct the review.", view.Model)
		}
		view.Model = ""
		if len(models) > 0 {
			view.Model = decision.Decided()
			if !slices.Contains(models, view.Model) {
				view.Model = models[0]
			}
		}
	}
	for _, level := range d.auto.Reasoning.IDs() {
		view.Levels = append(view.Levels, option{level, level})
	}
	return view
}

// reviewModels keeps the recorded offer's order, restricted to models
// still configured. Without a recorded offer, it uses current candidates.
func (d *Dashboard) reviewModels(decision pin.Event) []string {
	if len(decision.Basis.Candidates) == 0 {
		return d.auto.CandidateIDs()
	}
	var models []string
	for _, model := range decision.Basis.Candidates {
		if d.auto.Rank(model) >= 0 {
			models = append(models, model)
		}
	}
	return models
}

// right is how a review judged the decision: on the model and on the
// reasoning level.
func (v reviewView) right() (model, level bool) {
	return v.Review.Model == v.Decided, v.Review.Level == v.DecidedLevel
}

// review records, or forgets, an operator's review of a thread's decision,
// then reloads the thread's page.
func (d *Dashboard) review(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxReviewBody)
	var signals struct {
		Model  string `json:"reviewModel"`
		Level  string `json:"reviewLevel"`
		Note   string `json:"reviewNote"`
		Forget bool   `json:"reviewForget"`
	}
	if err := datastar.ReadSignals(r, &signals); err != nil {
		status := http.StatusBadRequest
		if _, tooLarge := errors.AsType[*http.MaxBytesError](err); tooLarge {
			status = http.StatusRequestEntityTooLarge
		}
		http.Error(w, err.Error(), status)
		return
	}
	key := r.PathValue("key")
	sse := datastar.NewSSE(w, r)
	detail, err := d.source.Thread(r.Context(), key)
	switch {
	case err != nil:
		d.log.Warn("dashboard review failed", "error", err)
		d.patch(sse, reviewProblem("the thread could not be read"))
		return
	case detail == nil || detail.Decision == nil:
		d.patch(sse, reviewProblem("this thread has no decision to review"))
		return
	}
	event := detail.Decision.ID
	if signals.Forget {
		err = d.source.DeleteReview(r.Context(), event)
	} else if err = d.checkReview(signals.Model, signals.Level, signals.Note, *detail.Decision); err == nil {
		err = d.source.SaveReview(r.Context(), pin.Review{Event: event, Model: signals.Model, Level: signals.Level,
			Note: strings.TrimSpace(signals.Note)})
		if err == nil && d.outcomes != nil {
			d.outcomes.Reviewed(event)
		}
	}
	if err != nil {
		d.log.Warn("dashboard review failed", "error", err)
		d.patch(sse, reviewProblem(err.Error()))
		return
	}
	if err := sse.Redirect(string(threadURL(key)) + "#review"); err != nil && !sse.IsClosed() {
		d.log.Warn("dashboard patch failed", "error", err)
	}
}

// checkReview refuses a model outside the decision's offer, an unknown
// reasoning level, or a long note.
func (d *Dashboard) checkReview(model, level, note string, decision pin.Event) error {
	switch {
	case len(decision.Basis.Candidates) > 0 && !slices.Contains(decision.Basis.Candidates, model):
		return fmt.Errorf("%q is not a candidate offered for this decision; choose one of: %s", model,
			strings.Join(d.reviewModels(decision), ", "))
	case d.auto.Rank(model) < 0:
		return fmt.Errorf("%q is not an active candidate; select an offered model still in service", model)
	case level != "" && d.auto.Reasoning.Rank(level) < 0:
		return fmt.Errorf("%q is not a reasoning level", level)
	case utf8.RuneCountInString(note) > maxNote:
		return fmt.Errorf("the note is longer than %d characters", maxNote)
	}
	return nil
}

// cases serves the reviewed decisions as labelled bench cases: each first
// prompt with the model and reasoning level its review says it should
// have had, for `ai-router bench --cases`.
func (d *Dashboard) cases(w http.ResponseWriter, r *http.Request) {
	reviewed, err := d.source.Reviewed(r.Context())
	if err != nil {
		d.log.Warn("dashboard case export failed", "error", err)
		http.Error(w, "the reviewed decisions could not be read", http.StatusInternalServerError)
		return
	}
	file := struct {
		Cases []bench.Case `yaml:"cases"`
	}{Cases: []bench.Case{}}
	for _, e := range reviewed {
		prompt := firstPrompt(e)
		if prompt == "" {
			continue
		}
		c := bench.Case{Name: "review/" + e.At.UTC().Format("2006-01-02") + "-" + strconv.FormatInt(e.ID, 10),
			Expect: []string{e.Review.Model}, Images: e.Basis.Images, Offered: slices.Clone(e.Basis.Candidates),
			Note: e.Review.Note, Prompt: prompt}
		if e.Review.Level != "" {
			c.Reasoning = []string{e.Review.Level}
		}
		if len(c.Offered) > 0 && !slices.Contains(c.Offered, e.Review.Model) {
			http.Error(w, fmt.Sprintf("Reviewed case %q cannot be replayed: model %q was not offered for this decision. Correct or forget this review before exporting.",
				c.Name, e.Review.Model), http.StatusConflict)
			return
		}
		file.Cases = append(file.Cases, c)
	}
	encoded, err := yaml.Marshal(file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	now := time.Now().UTC()
	w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="ai-router-reviews-`+now.Format("20060102")+`.yaml"`)
	fmt.Fprintf(w, "# %d reviewed ai-router decisions, exported %s.\n# ai-router bench --cases <this file>\n",
		len(file.Cases), now.Format(time.RFC3339))
	_, _ = w.Write(encoded)
}

// firstPrompt is a decision's first prompt as the user wrote it, or, for
// a decision recorded before it was kept, the prompt as judged.
func firstPrompt(e pin.Event) string {
	return cmp.Or(e.FirstPrompt, promptText(e.Prompt))
}

// reviewNote is a review in one line, for a tooltip.
func reviewNote(e pin.Event) string {
	r := e.Review
	model, level := r.Right(e)
	text := "Reviewed right"
	if !model || !level {
		text = "Should have been " + r.Model
		if r.Level != "" {
			text += " at " + r.Level
		}
	}
	if r.Note != "" {
		text += ": " + r.Note
	}
	return text
}
