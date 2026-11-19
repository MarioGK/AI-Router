package dashboard

import (
	"net/http"
	"slices"
)

// Pages in the navigation.
const (
	navOverview  = "overview"
	navThreads   = "threads"
	navDecisions = "decisions"
	navLogs      = "logs"
	navRouting   = "routing"
)

// frame is what the page chrome needs.
type frame struct {
	Title string
	Nav   string
	Theme string
}

func newFrame(r *http.Request, title, nav string) frame {
	theme := "dark"
	if cookie, err := r.Cookie(themeCookie); err == nil {
		theme = themeName(cookie.Value)
	}
	return frame{Title: title, Nav: nav, Theme: theme}
}

// option is one entry of a select.
type option struct {
	Value, Label string
}

func modelOptions(models []string) []option {
	options := make([]option, len(models))
	for i, model := range models {
		options[i] = option{model, model}
	}
	return options
}

// historyModels includes retired models so their threads and requests
// remain selectable after they leave the active catalog requirements.
func (d *Dashboard) historyModels() []string {
	models := append(d.auto.Required(), d.auto.Retired...)
	slices.Sort(models)
	return slices.Compact(models)
}
