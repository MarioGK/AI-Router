package dashboard

import (
	"cmp"
	"errors"
	"net/http"
	"slices"

	"github.com/starfederation/datastar-go/datastar"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
)

// maxRepinBody bounds a re-pin request's body: two short signals.
const maxRepinBody = 16 << 10

// thread shows one thread: its pin, decision, history and requests.
func (d *Dashboard) thread(w http.ResponseWriter, r *http.Request) {
	key := r.PathValue("key")
	detail, err := d.source.Thread(r.Context(), key)
	f := newFrame(r, "Thread "+shortKey(key), navThreads)
	switch {
	case err != nil:
		d.log.Warn("dashboard thread failed", "error", err)
		d.render(w, r, http.StatusInternalServerError, threadMissing(f, key, err))
	case detail == nil:
		d.render(w, r, http.StatusNotFound, threadMissing(f, key, nil))
	default:
		var review reviewView
		if detail.Decision != nil {
			review = d.reviewOf(key, *detail.Decision)
		}
		d.render(w, r, http.StatusOK, threadPage(f, *detail, d.effort(detail.Pin), d.judged(detail.Decision),
			d.repinOf(detail.Pin), review))
	}
}

// repinView is what the re-pin card offers a thread: every candidate and,
// with the reasoning chooser on, every level.
type repinView struct {
	Key, Model, Level string
	Models, Levels    []option
}

func (d *Dashboard) repinOf(p pin.Pin) repinView {
	// The form starts from the pin, as far as the configuration still has
	// its model and level.
	view := repinView{Key: p.Key, Model: p.Model, Level: p.Level}
	if d.auto.Rank(p.Model) < 0 {
		view.Model = d.auto.DefaultModel
	}
	if d.auto.Reasoning.Rank(p.Level) < 0 {
		view.Level = ""
	}
	for _, candidate := range d.auto.Candidates {
		view.Models = append(view.Models, option{candidate.ID, candidate.ID})
	}
	for _, level := range d.auto.Reasoning.Levels {
		view.Levels = append(view.Levels, option{level.ID, level.ID})
	}
	return view
}

// repin re-pins a thread on the operator's word, then reloads its page.
func (d *Dashboard) repin(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxRepinBody)
	var signals struct {
		Model string `json:"repinModel"`
		Level string `json:"repinLevel"`
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
	if _, err := d.classifier.Override(r.Context(), key, signals.Model, signals.Level); err != nil {
		d.log.Warn("dashboard re-pin failed", "error", err)
		d.patch(sse, repinProblem(err.Error()))
		return
	}
	if err := sse.Redirect(string(threadURL(key))); err != nil && !sse.IsClosed() {
		d.log.Warn("dashboard patch failed", "error", err)
	}
}

// effort is the reasoning effort a pinned thread's model runs at; empty
// when the client's own setting applies.
func (d *Dashboard) effort(p pin.Pin) string {
	return d.auto.Effort(p.Model, p.Level)
}

// odds is one candidate's or reasoning level's probability in a decision.
type odds struct {
	Model       string
	Probability float64
}

// judged is how a decision weighed the candidates, with each ladder
// step's expected cost to get the task done when it weighed cost, and, with
// the reasoning chooser on, the reasoning levels.
type judged struct {
	Candidates []odds
	Done       map[string]float64
	// Costs is the whole judgement of cost, with the learned time and miss
	// rates that entered each step's score.
	Costs  *decide.Costs
	Levels []odds
}

// judged lists a decision's probabilities along the ladder and the
// levels, each followed by any the configuration no longer has, by name.
func (d *Dashboard) judged(decision *pin.Event) judged {
	if decision == nil {
		return judged{}
	}
	j := judged{Candidates: ordered(decision.Probabilities, d.auto.CandidateIDs()),
		Levels: ordered(decision.Levels, d.auto.Reasoning.IDs())}
	if decision.Basis.Costs != nil {
		j.Done, j.Costs = decision.Basis.Costs.Done, decision.Basis.Costs
	}
	return j
}

// ordered lists probabilities in the order of known ids, then the others
// by name.
func ordered(probabilities map[string]float64, known []string) []odds {
	var list []odds
	for _, id := range known {
		if p, ok := probabilities[id]; ok {
			list = append(list, odds{id, p})
		}
	}
	var rest []odds
	for id, p := range probabilities {
		if !slices.Contains(known, id) {
			rest = append(rest, odds{id, p})
		}
	}
	slices.SortFunc(rest, func(a, b odds) int { return cmp.Compare(a.Model, b.Model) })
	return append(list, rest...)
}

// eventGlyph and eventTone draw an event's mark in the history.
func eventGlyph(kind string) string {
	switch kind {
	case pin.Failover, pin.Migration:
		return "swap"
	case pin.CapabilityUpgrade:
		return "image"
	case pin.Override:
		return "key"
	}
	return "spark"
}

func eventTone(kind string) string {
	if kind == pin.Failover {
		return "border-serious/40 bg-serious/10 text-serious"
	}
	return "border-line bg-raised text-fg-2"
}

// eventAbout is the line under an event.
func eventAbout(e pin.Event) string {
	switch e.Kind {
	case pin.Decision:
		about := "took " + duration(e.Latency)
		if e.Level != "" {
			about += ", " + e.Level + " reasoning"
		}
		if e.Escalated {
			about += ", escalated"
		}
		if e.Consulted() {
			about += ", after a second opinion"
		}
		return about
	case pin.CapabilityUpgrade:
		return "the thread carried " + e.Detail
	case pin.Redecision:
		return "took " + duration(e.Latency) + ", " + e.Detail
	}
	return e.Detail
}
