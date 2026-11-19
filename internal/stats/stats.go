// Package stats serves /healthz and /stats: catalog health, pins by model,
// the decision log and in-memory request counters.
package stats

import (
	"encoding/json"
	"maps"
	"net/http"
	"sync"
	"time"

	"github.com/MarioGK/AI-Router/internal/catalog"
	"github.com/MarioGK/AI-Router/internal/costs"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/learn"
	"github.com/MarioGK/AI-Router/internal/outcome"
	"github.com/MarioGK/AI-Router/internal/pin"
)

// Counters are in-memory request counters since the process started.
type Counters struct {
	mu     sync.Mutex
	values map[string]int64
}

// Add increments a counter.
func (c *Counters) Add(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.values == nil {
		c.values = map[string]int64{}
	}
	c.values[name]++
}

// Snapshot copies the counters.
func (c *Counters) Snapshot() map[string]int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return maps.Clone(c.values)
}

// Source is what the endpoints report on.
type Source struct {
	Store    *pin.Store
	Catalog  *catalog.Catalog
	Required []string
	Counters *Counters
	// Deciders is the decider chain's health, and Costs what a task is
	// expected to cost on each candidate at each reasoning level.
	Deciders func() []decide.DeciderHealth
	Costs    func() []costs.Estimate
	// Outcomes is how the settled threads went: how many showed each sign
	// of a misroute, and the time per task and miss rate learned per
	// candidate and level; nil reports none.
	Outcomes func() Outcomes
	// Learn is what routing learned from its own history: the labels by
	// kind and the weights the similar-prompt prior earned; nil reports
	// none.
	Learn   func() learn.Summary
	Started time.Time
}

// Outcomes is how the settled threads went.
type Outcomes struct {
	outcome.Counts
	Estimates []outcome.Estimate `json:"estimates"`
}

// Health answers 200 when CLIProxy serves every configured model, else
// 503, also while the router drains. Anyone may ask, so a refresh error is
// reported without its details, which name CLIProxy's address.
func (s Source) Health(w http.ResponseWriter, _ *http.Request, draining bool) {
	status := s.Catalog.Check(s.Required)
	code := http.StatusOK
	if !status.Healthy() || draining {
		code = http.StatusServiceUnavailable
	}
	if status.Error != "" {
		status.Error = "the latest model list refresh failed"
	}
	write(w, code, struct {
		catalog.Status
		Draining bool `json:"draining,omitempty"`
	}{status, draining})
}

// Stats reports pins, decisions over the last day and week, how the
// reviewed decisions fared, and counters.
func (s Source) Stats(w http.ResponseWriter, r *http.Request) {
	now := time.Now()
	day, err := s.Store.Summarize(r.Context(), now.Add(-24*time.Hour))
	var week pin.Summary
	var reviews pin.ReviewSummary
	if err == nil {
		week, err = s.Store.Summarize(r.Context(), now.Add(-7*24*time.Hour))
	}
	if err == nil {
		reviews, err = s.Store.Reviews(r.Context(), time.Time{})
	}
	if err != nil {
		write(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	body := map[string]any{
		"started":  s.Started,
		"catalog":  s.Catalog.Check(s.Required),
		"last_24h": day,
		"last_7d":  week,
		"reviews":  reviews,
		"deciders": s.Deciders(),
		"costs":    s.Costs(),
		"requests": s.Counters.Snapshot(),
	}
	if s.Outcomes != nil {
		body["outcomes"] = s.Outcomes()
	}
	if s.Learn != nil {
		body["learn"] = s.Learn()
	}
	write(w, http.StatusOK, body)
}

func write(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	_ = encoder.Encode(body)
}
