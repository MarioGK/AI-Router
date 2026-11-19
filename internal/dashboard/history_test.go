package dashboard

import (
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/catalog"
	"github.com/MarioGK/AI-Router/internal/config"
)

type checkedCatalog struct{ required []string }

func (c *checkedCatalog) Check(required []string) catalog.Status {
	c.required = slices.Clone(required)
	return catalog.Status{Loaded: true}
}

func TestRetiredModelsRemainSelectableInHistory(t *testing.T) {
	source := history()
	source.threads[0].Model = "retired-model"
	source.threads[1].Model = "claude-sonnet-5-5"
	source.requests[0].Model = "retired-model"
	source.requests[1].Model = "claude-sonnet-5-5"
	d := newDashboard(t, source)
	d.auto = config.Auto{
		DefaultModel: "glm-5.3", AuxModel: "glm-5.3-flash",
		Candidates: []config.Candidate{{ID: "glm-5.3"}, {ID: "gpt-6.1-sol"}, {ID: "claude-opus-5-5"}},
		Fallback:   map[string]string{"retired-model": "gpt-6.1-sol", "claude-sonnet-5-5": "gpt-6.1-sol"},
		Retired:    []string{"retired-model", "claude-sonnet-5-5"},
	}
	models := &checkedCatalog{}
	d.catalog = models
	cookie := signIn(t, d)

	for _, path := range []string{"/dashboard/threads", "/dashboard/logs"} {
		body := page(t, d, cookie, path, http.StatusOK)
		for _, model := range d.auto.Retired {
			if count := strings.Count(body, `value="`+model+`"`); count != 1 {
				t.Errorf("%s has %d filter options for retired %s, want one", path, count, model)
			}
		}
	}

	for i, model := range d.auto.Retired {
		body := page(t, d, cookie, "/dashboard/threads?model="+model, http.StatusOK)
		contains(t, "retired threads", body, `value="`+model+`" selected`, shortKey(source.threads[i].Key))
		if strings.Contains(body, shortKey(source.threads[1-i].Key)) {
			t.Errorf("%s thread filter kept another model's thread", model)
		}
		if got := source.threadFilters[len(source.threadFilters)-1].Model; got != model {
			t.Errorf("thread filter = %q, want %q", got, model)
		}

		body = page(t, d, cookie, "/dashboard/logs?model="+model, http.StatusOK)
		contains(t, "retired requests", body, `value="`+model+`" selected`, `id="request-`+number(i+1)+`"`)
		if got := source.lastRequestFilter().Model; got != model {
			t.Errorf("request filter = %q, want %q", got, model)
		}
	}

	day := d.stats(t.Context())
	for _, model := range d.auto.Retired {
		if slices.Contains(models.required, model) || slices.Contains(day.Required, model) {
			t.Errorf("retired %s is still required for catalog health", model)
		}
	}
	if !slices.Contains(models.required, "gpt-6.1-sol") {
		t.Error("the active replacement is absent from catalog requirements")
	}
}
