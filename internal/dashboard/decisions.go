package dashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/a-h/templ"
	"github.com/starfederation/datastar-go/datastar"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/costs"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/learn"
	"github.com/MarioGK/AI-Router/internal/outcome"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/route"
)

// Classifier decides prompts the way the router does; *route.Resolver is
// one.
type Classifier interface {
	// Policy is the policy a new thread is decided under.
	Policy(images bool) decide.Policy
	// Try decides a prompt without pinning anything.
	Try(ctx context.Context, prompt string, images bool) (pin.Event, decide.Policy)
	// Health is the decider chain's health.
	Health() []decide.DeciderHealth
	// Override re-pins a thread on an operator's word.
	Override(ctx context.Context, key, model, level string) (*pin.Pin, error)
	// Settings reads the persistent routing controls.
	Settings(ctx context.Context) (route.Settings, error)
	// SetSettings validates and persists the routing controls.
	SetSettings(ctx context.Context, settings route.Settings) (route.Settings, error)
}

const (
	// decisionsPerPage is the length of a page of decisions.
	decisionsPerPage = 25
	// mixWindow is how far back the decisions per model reach.
	mixWindow = 30 * 24 * time.Hour
	// maxTrialChars bounds a trial prompt; the router cuts what it judges
	// to state_max_chars anyway.
	maxTrialChars = 100_000
	// trialTimeout bounds a trial's decider chain.
	trialTimeout = time.Minute
	// maxTrialBody bounds a trial request's body: the prompt and the other
	// signals.
	maxTrialBody = 1 << 20
	// maxTrials is how many trials may run at once.
	maxTrials = 4
	// queueLength is how many likely misrouted decisions the review queue
	// lists.
	queueLength = 15
)

// decisionQuery selects a page of decisions.
type decisionQuery struct {
	Search  string `json:"search"`
	Model   string `json:"model"`
	Decider string `json:"decider"`
	Review  string `json:"review"`
	Page    int    `json:"-"`
}

// values is the query as page URL parameters.
func (q decisionQuery) values() url.Values {
	values := url.Values{}
	for name, value := range map[string]string{"model": q.Model, "decider": q.Decider, "review": q.Review, "search": q.Search} {
		if value != "" {
			values.Set(name, value)
		}
	}
	if q.Page > 1 {
		values.Set("page", strconv.Itoa(q.Page))
	}
	return values
}

// decisionsView is one page of decisions.
type decisionsView struct {
	Query decisionQuery
	pin.DecisionPage
	Err error
}

func (v decisionsView) pages() int {
	return max(1, (v.Total+decisionsPerPage-1)/decisionsPerPage)
}

func (v decisionsView) first() int {
	return (v.Query.Page-1)*decisionsPerPage + 1
}

func (v decisionsView) last() int {
	return v.first() + len(v.Decisions) - 1
}

// modelMix is one candidate's share of the decisions, what the policy
// sends it, the efforts it runs at and the latest prompts that went to it.
type modelMix struct {
	Model       string
	Description string
	Efforts     string
	Count       int
	Share       float64
	Examples    []pin.Event
}

// decisionsPageView is everything the decisions page shows.
type decisionsPageView struct {
	Mix       []modelMix
	MixErr    error
	Policy    decide.Policy
	Costs     costView
	Queue     queueView
	Reviews   pin.ReviewSummary
	ReviewErr error
	// Learning is what routing learned from its own history.
	Learning learn.Summary
	Table    decisionsView
	Models   []string
	Deciders []string
}

// costView is what a task is expected to cost on each candidate at each
// reasoning level, and whether decisions weigh it.
type costView struct {
	Weighed bool
	Miss    float64
	// TimeValue is what an hour of a task's time is worth; Learning is set
	// when thread outcomes teach the time and miss rate of each cell.
	TimeValue float64
	Learning  bool
	Levels    []string
	Rows      []costRow
	// Threads is how many settled threads the costs were learned from,
	// Learned when.
	Threads int
	Learned time.Time
}

// costRow is one candidate's expected cost per task at each level, and
// what thread outcomes taught about each of those cells.
type costRow struct {
	Model   string
	Cells   []costs.Estimate
	Learned []outcome.Estimate
}

// queueView is the unreviewed decisions most likely misrouted, and how
// many wait in all.
type queueView struct {
	Enabled bool
	Items   []outcome.Outcome
	Queued  int
	Threads int
}

// queueOf reads the review queue.
func (d *Dashboard) queueOf() queueView {
	if d.outcomes == nil {
		return queueView{}
	}
	counts := d.outcomes.Counts()
	return queueView{Enabled: true, Items: d.outcomes.Queue(queueLength), Queued: counts.Queued, Threads: counts.Threads}
}

// learningOf is what routing learned from its own history; off without a
// learner.
func (d *Dashboard) learningOf() learn.Summary {
	if d.learning == nil {
		return learn.Summary{}
	}
	return d.learning()
}

// labelKinds counts the labels by kind, strongest first: "3 reviewed,
// 12 sufficed".
func labelKinds(s learn.Summary) string {
	var parts []string
	for _, kind := range learn.Kinds {
		if n := s.Kinds[kind]; n > 0 {
			parts = append(parts, number(n)+" "+strings.ReplaceAll(kind, "_", " "))
		}
	}
	if len(parts) == 0 {
		return "no settled thread labelled yet"
	}
	return strings.Join(parts, ", ")
}

// weightText is a blend weight with two decimals.
func weightText(w float64) string {
	return strconv.FormatFloat(w, 'f', 2, 64)
}

// costsOf lays out the expected cost per task of every ladder candidate.
func (d *Dashboard) costsOf() costView {
	view := costView{Weighed: d.auto.Cost.Enabled, Levels: d.auto.Reasoning.IDs(), TimeValue: d.auto.Cost.TimeValue,
		Learning: d.outcomes != nil}
	if d.auto.Cost.Miss != nil {
		view.Miss = *d.auto.Cost.Miss
	}
	models := d.auto.CandidateIDs()
	estimates := d.costs.Estimates(models, view.Levels)
	for _, model := range models {
		row := costRow{Model: model}
		for _, estimate := range estimates {
			if estimate.Model == model {
				row.Cells = append(row.Cells, estimate)
				if d.outcomes != nil {
					row.Learned = append(row.Learned, d.outcomes.Estimate(model, estimate.Level))
				}
			}
		}
		if len(row.Cells) > 0 {
			view.Rows = append(view.Rows, row)
		}
	}
	view.Threads, view.Learned = d.costs.Learned()
	return view
}

// decisions shows how auto classifies prompts: the decisions per model
// with example prompts, the classifier's input, a trial box and every
// decision.
func (d *Dashboard) decisions(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	page, _ := strconv.Atoi(query.Get("page"))
	view := decisionsPageView{
		Table: d.listDecisions(r.Context(), decisionQuery{
			Model: query.Get("model"), Decider: query.Get("decider"), Review: query.Get("review"), Search: query.Get("search"), Page: page,
		}),
		Policy:   d.classifier.Policy(false),
		Costs:    d.costsOf(),
		Queue:    d.queueOf(),
		Learning: d.learningOf(),
		Models:   d.auto.CandidateIDs(),
	}
	if view.Reviews, view.ReviewErr = d.source.Reviews(r.Context(), time.Time{}); view.ReviewErr != nil {
		d.log.Warn("dashboard reviews failed", "error", view.ReviewErr)
	}
	for _, decider := range view.Policy.Deciders {
		view.Deciders = append(view.Deciders, decider.Name)
	}
	view.Deciders = append(view.Deciders, config.DefaultDecider)
	view.Mix, view.MixErr = d.mix30(r.Context())
	d.render(w, r, http.StatusOK, decisionsPage(newFrame(r, "Decisions", navDecisions), view))
}

// mix30 is the last 30 days of decisions per candidate, in ladder order,
// then any model the configuration no longer has.
func (d *Dashboard) mix30(ctx context.Context) ([]modelMix, error) {
	counted, err := d.source.DecisionMix(ctx, time.Now().Add(-mixWindow))
	if err != nil {
		d.log.Warn("dashboard decision mix failed", "error", err)
		return nil, err
	}
	total := 0
	for _, m := range counted {
		total += m.Count
	}
	find := func(model string) pin.ModelDecisions {
		if i := slices.IndexFunc(counted, func(m pin.ModelDecisions) bool { return m.Model == model }); i >= 0 {
			return counted[i]
		}
		return pin.ModelDecisions{Model: model}
	}
	var mix []modelMix
	for _, candidate := range d.auto.Candidates {
		m := find(candidate.ID)
		mix = append(mix, modelMix{Model: candidate.ID, Description: candidate.Description, Efforts: efforts(candidate),
			Count: m.Count, Share: ratio(m.Count, total), Examples: m.Examples})
	}
	for _, m := range counted {
		if d.auto.Rank(m.Model) < 0 {
			mix = append(mix, modelMix{Model: m.Model, Count: m.Count, Share: ratio(m.Count, total), Examples: m.Examples})
		}
	}
	return mix, nil
}

func ratio(part, whole int) float64 {
	if whole == 0 {
		return 0
	}
	return float64(part) / float64(whole)
}

// decisionRows re-renders the decision table for the filters and page
// asked for.
func (d *Dashboard) decisionRows(w http.ResponseWriter, r *http.Request) {
	var query decisionQuery
	if err := datastar.ReadSignals(r, &query); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	query.Page, _ = strconv.Atoi(r.URL.Query().Get("page"))
	view := d.listDecisions(r.Context(), query)
	sse := datastar.NewSSE(w, r)
	d.patch(sse, decisionsTable(view))
	d.replaceURL(sse, "/dashboard/decisions", view.Query.values())
}

// listDecisions reads a page of decisions; a page past the end shows the
// last.
func (d *Dashboard) listDecisions(ctx context.Context, query decisionQuery) decisionsView {
	query.Page = max(1, query.Page)
	view := decisionsView{Query: query}
	for {
		filter := pin.DecisionFilter{Model: query.Model, Decider: query.Decider, Review: query.Review, Search: query.Search,
			Limit: decisionsPerPage, Offset: (view.Query.Page - 1) * decisionsPerPage}
		view.DecisionPage, view.Err = d.source.Decisions(ctx, filter)
		if view.Err != nil {
			d.log.Warn("dashboard decisions failed", "error", view.Err)
			return view
		}
		if len(view.Decisions) > 0 || view.Query.Page <= view.pages() {
			return view
		}
		view.Query.Page = view.pages()
	}
}

// try decides the prompt in the trial box without pinning anything. Every
// trial spends decider calls, so at most maxTrials run at once; the others
// get 429.
func (d *Dashboard) try(w http.ResponseWriter, r *http.Request) {
	select {
	case d.trials <- struct{}{}:
		defer func() { <-d.trials }()
	default:
		http.Error(w, "too many trials at once; try again shortly", http.StatusTooManyRequests)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxTrialBody)
	var signals struct {
		Prompt string `json:"prompt"`
		Images bool   `json:"images"`
	}
	if err := datastar.ReadSignals(r, &signals); err != nil {
		status := http.StatusBadRequest
		if _, tooLarge := errors.AsType[*http.MaxBytesError](err); tooLarge {
			status = http.StatusRequestEntityTooLarge
		}
		http.Error(w, err.Error(), status)
		return
	}
	prompt := strings.TrimSpace(signals.Prompt)
	if utf8.RuneCountInString(prompt) > maxTrialChars {
		prompt = string([]rune(prompt)[:maxTrialChars])
	}
	sse := datastar.NewSSE(w, r)
	if prompt == "" {
		d.patch(sse, trialResult(nil, decide.Policy{}, judged{}, ""))
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), trialTimeout)
	defer cancel()
	event, policy := d.classifier.Try(ctx, prompt, signals.Images)
	d.patch(sse, trialResult(&event, policy, d.judged(&event), d.auto.Effort(event.Model, event.Level)))
}

// exchange is what one decider was sent and what it answered.
type exchange struct {
	Attempt decide.Attempt
	Info    decide.DeciderInfo
	Sent    string
	// Rebuilt is false when the request cannot be shown: the prompt or
	// the policy was not recorded, or the decider was not asked.
	Rebuilt bool
	// Drifted is set when the rebuilt request differs from the one sent,
	// by its digest: the router's request format changed since.
	Drifted bool
}

// exchanges pairs each attempt of a decision with the request its decider
// was sent, rebuilt from the recorded prompt and policy.
func exchanges(policy *decide.Policy, decision pin.Event) []exchange {
	var list []exchange
	for _, attempt := range decision.Attempts {
		e := exchange{Attempt: attempt, Info: decide.DeciderInfo{Name: attempt.Decider}}
		if policy != nil {
			if i := slices.IndexFunc(policy.Deciders, func(info decide.DeciderInfo) bool { return info.Name == attempt.Decider }); i >= 0 {
				e.Info = policy.Deciders[i]
			}
			if decision.Prompt != "" && attempt.Outcome != decide.Skipped {
				body, ok := policy.Request(attempt.Decider, decision.Prompt)
				if len(attempt.Pair) == 2 {
					body, ok = policy.PairRequest(attempt.Decider, decision.Prompt, attempt.Pair[0], attempt.Pair[1])
				}
				if ok {
					e.Sent, e.Rebuilt = pretty(body), true
					e.Drifted = attempt.Digest != "" && attempt.Digest != decide.Digest(body)
				}
			}
		}
		list = append(list, e)
	}
	return list
}

// pretty indents JSON text, and leaves any other text as it is.
func pretty(text []byte) string {
	var indented bytes.Buffer
	if json.Indent(&indented, text, "", "  ") != nil {
		return string(text)
	}
	return indented.String()
}

// decisionURL is the decision of a thread on its page.
func decisionURL(key string) templ.SafeURL {
	return templ.SafeURL("/dashboard/threads/" + url.PathEscape(key) + "#decision")
}

// pairQuestionText is the pair question with placeholders for the two
// ladder steps it compares.
func pairQuestionText(question string) string {
	return fmt.Sprintf(question, "‹choice›", "‹next step›")
}

// promptText is the user text of a decider state, without the images line
// the router puts first (a client too, before it served only Claude Code).
func promptText(state string) string {
	if first, rest, ok := strings.Cut(state, "\n\n"); ok && (strings.HasPrefix(first, "images=") || strings.HasPrefix(first, "client=")) {
		return rest
	}
	return state
}
