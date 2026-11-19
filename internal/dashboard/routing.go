package dashboard

import (
	"cmp"
	"errors"
	"mime"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/route"
)

// maxRoutingBody bounds the routing form, including its model group.
const maxRoutingBody = 16 << 10

type routingView struct {
	Current    route.Settings
	Draft      route.Settings
	Candidates []config.Candidate
	Available  bool
	Saved      bool
	Problem    string
	SolHigh    bool
	ModelID    string
	Ranks      map[string]string
}

func (d *Dashboard) routingView(settings route.Settings) routingView {
	draft := settings
	draft.Models = slices.Clone(settings.Models)
	draft.Preferences.Ranking = slices.Clone(settings.Preferences.Ranking)
	if len(draft.Models) == 0 {
		draft.Models = d.auto.CandidateIDs()
	}
	ranks := make(map[string]string, len(settings.Preferences.Ranking))
	for i, model := range settings.Preferences.Ranking {
		ranks[model] = strconv.Itoa(i + 1)
	}
	return routingView{Current: settings, Draft: draft, Candidates: d.auto.Candidates, Available: true, Ranks: ranks,
		SolHigh: d.auto.Rank("gpt-6.1-sol") >= 0 && d.auto.Effort("gpt-6.1-sol", "") == "high",
		ModelID: cmp.Or(d.auto.ModelID, "auto")}
}

// routing shows the persisted controls for subsequent automatic-model requests.
func (d *Dashboard) routing(w http.ResponseWriter, r *http.Request) {
	settings, err := d.classifier.Settings(r.Context())
	view := d.routingView(settings)
	status := http.StatusOK
	if err != nil {
		d.log.Warn("dashboard routing settings read failed", "error", err)
		view.Available = false
		view.Problem = "Routing settings could not be read. Try again before making changes."
		status = http.StatusInternalServerError
	}
	view.Saved = view.Available && r.URL.Query().Get("saved") == "1"
	d.render(w, r, status, routingPage(newFrame(r, "Routing settings", navRouting), view))
}

// saveRouting validates a bounded native form before persisting any changes.
func (d *Dashboard) saveRouting(w http.ResponseWriter, r *http.Request) {
	settings, err := d.classifier.Settings(r.Context())
	view := d.routingView(settings)
	if err != nil {
		d.log.Warn("dashboard routing settings read failed", "error", err)
		view.Available = false
		view.Problem = "Routing settings could not be read. Your changes were not saved."
		d.render(w, r, http.StatusInternalServerError, routingPage(newFrame(r, "Routing settings", navRouting), view))
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxRoutingBody)
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/x-www-form-urlencoded" {
		view.Problem = "Submit routing settings using the form on this page."
		d.render(w, r, http.StatusUnsupportedMediaType, routingPage(newFrame(r, "Routing settings", navRouting), view))
		return
	}
	if err := r.ParseForm(); err != nil {
		status := http.StatusBadRequest
		view.Problem = "The routing form could not be read. Check your selections and try again."
		if _, tooLarge := errors.AsType[*http.MaxBytesError](err); tooLarge {
			status = http.StatusRequestEntityTooLarge
			view.Problem = "The routing form is too large. Your changes were not saved."
		}
		d.render(w, r, status, routingPage(newFrame(r, "Routing settings", navRouting), view))
		return
	}
	draft, err := d.readRoutingForm(r.PostForm)
	view.Draft = draft
	view.Ranks = make(map[string]string, len(view.Candidates))
	for _, candidate := range view.Candidates {
		view.Ranks[candidate.ID] = r.PostForm.Get("rank:" + candidate.ID)
	}
	if err != nil {
		view.Problem = err.Error()
		d.render(w, r, http.StatusBadRequest, routingPage(newFrame(r, "Routing settings", navRouting), view))
		return
	}
	view.Draft = d.routingView(draft).Draft
	if _, err := d.classifier.SetSettings(r.Context(), draft); err != nil {
		d.log.Warn("dashboard routing settings save failed", "error", err)
		view.Problem = "Routing settings could not be saved. Your previous settings are still active. Try again."
		d.render(w, r, http.StatusInternalServerError, routingPage(newFrame(r, "Routing settings", navRouting), view))
		return
	}
	http.Redirect(w, r, "/dashboard/routing?saved=1", http.StatusSeeOther)
}

func (d *Dashboard) readRoutingForm(values url.Values) (route.Settings, error) {
	settings := route.Settings{Mode: values.Get("mode"), FixedModel: values.Get("fixed_model"),
		Models: slices.Clone(values["models"]), JevEnabled: values.Get("jev_enabled") == "true"}
	settings.Preferences.Priority = values.Get("priority")
	for name, entries := range values {
		if model, rank := strings.CutPrefix(name, "rank:"); rank {
			if len(entries) != 1 || d.auto.Rank(model) < 0 || d.auto.IsRetired(model) {
				return settings, errors.New("Choose one preference per active model using the dropdowns.")
			}
			continue
		}
		switch name {
		case "mode", "fixed_model", "priority":
			if len(entries) != 1 {
				return settings, errors.New("Choose one routing mode and one fixed model.")
			}
		case "jev_enabled":
			if len(entries) != 1 || entries[0] != "true" {
				return settings, errors.New("The Jev selection is invalid. Use the checkbox on this page.")
			}
		case "models":
		case "reset":
			if len(entries) != 1 || entries[0] != "full_auto" {
				return settings, errors.New("Use the Full auto button to reset routing settings.")
			}
		default:
			return settings, errors.New("The routing form contains an unknown field. Use the controls on this page.")
		}
	}
	if values.Get("reset") == "full_auto" {
		return route.Settings{Mode: route.ModeAuto, JevEnabled: true}, nil
	}
	ordered := make([]string, len(d.auto.Candidates))
	for _, candidate := range d.auto.Candidates {
		value := values.Get("rank:" + candidate.ID)
		if value == "" {
			continue
		}
		position, err := strconv.Atoi(value)
		if err != nil || position < 1 || position > len(ordered) {
			return settings, errors.New("Choose a model preference from the dropdown.")
		}
		if ordered[position-1] != "" {
			return settings, errors.New("Two models have the same preference. Choose a different position for each, or No preference.")
		}
		ordered[position-1] = candidate.ID
	}
	for _, model := range ordered {
		if model != "" {
			settings.Preferences.Ranking = append(settings.Preferences.Ranking, model)
		}
	}
	if err := settings.Preferences.Validate(); err != nil {
		return settings, err
	}
	if settings.Mode != route.ModeAuto && settings.Mode != route.ModeFixed {
		return settings, errors.New("Choose automatic routing or a fixed model.")
	}
	if settings.Mode == route.ModeFixed && settings.FixedModel == "" {
		return settings, errors.New("Choose the fixed model requests should use.")
	}
	if settings.FixedModel != "" && (d.auto.Rank(settings.FixedModel) < 0 || d.auto.IsRetired(settings.FixedModel)) {
		return settings, errors.New("The fixed model is not an active routing candidate. Choose a model from the list.")
	}
	if settings.Mode == route.ModeAuto && len(settings.Models) == 0 {
		return settings, errors.New("Select at least one model for automatic routing.")
	}
	seen := map[string]bool{}
	for _, model := range settings.Models {
		if seen[model] || d.auto.Rank(model) < 0 || d.auto.IsRetired(model) {
			return settings, errors.New("The automatic model group contains an unknown, retired, or repeated model. Choose models from the list.")
		}
		seen[model] = true
	}
	// A complete selection keeps the default pool, including its configured
	// auxiliary routing; a smaller selection is an explicit restriction.
	if len(settings.Models) == len(d.auto.Candidates) {
		settings.Models = nil
	} else {
		settings.Models = nil
		for _, candidate := range d.auto.Candidates {
			if seen[candidate.ID] {
				settings.Models = append(settings.Models, candidate.ID)
			}
		}
	}
	return settings, nil
}

func (v routingView) groupLabel() string {
	if len(v.Current.Models) == 0 {
		return "All configured models"
	}
	return strings.Join(v.Current.Models, ", ")
}

func (v routingView) priorityLabel() string {
	switch v.Current.Preferences.Priority {
	case "quality":
		return "Quality"
	case "speed":
		return "Speed"
	case "cost":
		return "Cost"
	default:
		return "Let Jev decide"
	}
}

func routingModelName(model string) string {
	switch model {
	case "deepseek-flash":
		return "DeepSeek Flash"
	case "glm-5.3-flash":
		return "GLM 5.3 Flash"
	case "glm-5.3":
		return "GLM 5.3"
	case "gpt-6.1-sol":
		return "GPT 6.1 Sol"
	case "claude-opus-5-5":
		return "Claude Opus 5.5"
	default:
		return model
	}
}

func routingRankLabel(position int) string {
	if position == 1 {
		return "1 · Highest preference"
	}
	return strconv.Itoa(position) + " · Preference"
}
