package dashboard

import (
	"cmp"
	"context"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/a-h/templ"
	"github.com/starfederation/datastar-go/datastar"

	"github.com/MarioGK/AI-Router/internal/catalog"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
)

const (
	// dayHours is the overview's period, in the timeline's hourly columns.
	dayHours = 24
	// Every so many polls a live stream re-renders its relative times, and
	// the overview its figures.
	retimeTicks  = 5
	refreshTicks = 15
)

// stats is the overview's picture of the last day.
type stats struct {
	pin.Overview
	Timeline timeline
	Catalog  catalog.Status
	Required []string
	// Deciders is the decider chain's health now.
	Deciders []decide.DeciderHealth
	Err      error
}

// mixView is the per-model share of decisions and requests over a range.
type mixView struct {
	Range     string
	Decisions []share
	Requests  []share
	Err       error
}

// since is the start of a period of n clock hours ending with the current
// one.
func since(n int) time.Time {
	return time.Now().Truncate(time.Hour).Add(-time.Duration(n-1) * time.Hour)
}

// overview shows the last day: headline figures, the hourly timeline, the
// model mix, the catalog and the live requests.
func (d *Dashboard) overview(w http.ResponseWriter, r *http.Request) {
	day := d.stats(r.Context())
	feed := &tail{source: d.source, revision: d.logRevision.Load}
	feed.load(r.Context())
	d.render(w, r, http.StatusOK, overviewPage(newFrame(r, "Overview", navOverview), day,
		d.mixOf("24h", day.Overview, day.Err), feed.view(), d.accountsOf(), d.limits.Stats()))
}

// stats reads the last day in hourly buckets.
func (d *Dashboard) stats(ctx context.Context) stats {
	start := since(dayHours)
	overview, err := d.source.Overview(ctx, start, time.Hour)
	if err != nil {
		d.log.Warn("dashboard overview failed", "error", err)
	}
	required := d.auto.Required()
	return stats{
		Overview: overview, Timeline: newTimeline(overview, start, dayHours),
		Catalog: d.catalog.Check(required), Required: required, Deciders: d.classifier.Health(), Err: err,
	}
}

// mix switches the model mix between the last day and the last week.
func (d *Dashboard) mix(w http.ResponseWriter, r *http.Request) {
	span, days := "24h", 1
	if r.URL.Query().Get("range") == "7d" {
		span, days = "7d", 7
	}
	overview, err := d.source.Overview(r.Context(), since(days*dayHours), time.Duration(days*dayHours)*time.Hour)
	if err != nil {
		d.log.Warn("dashboard model mix failed", "error", err)
	}
	d.patch(datastar.NewSSE(w, r), mixPanel(d.mixOf(span, overview, err)))
}

// mixOf shares out a period's decisions and requests by model.
func (d *Dashboard) mixOf(span string, overview pin.Overview, err error) mixView {
	requests := map[string]int{}
	for _, bucket := range overview.Timeline {
		for model, n := range bucket.Requests {
			requests[model] += n
		}
	}
	return mixView{
		Range: span, Decisions: shares(overview.Summary.Decisions), Requests: shares(requests), Err: err,
	}
}

// feed streams the overview's live requests and refreshes its figures.
func (d *Dashboard) feed(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	sse := datastar.NewSSE(w, r)
	feed := &tail{source: d.source, revision: d.logRevision.Load}
	feed.load(ctx)
	d.patch(sse, feedRows(feed.view()))
	d.follow(ctx, func(tick int) {
		if feed.poll(ctx) || tick%retimeTicks == 0 {
			d.patch(sse, feedRows(feed.view()))
		}
		if tick%refreshTicks == 0 {
			d.patch(sse, d.figures(d.stats(ctx))...)
		}
	})
}

// figures are the overview's sections a refresh replaces.
func (d *Dashboard) figures(day stats) []templ.Component {
	return []templ.Component{kpis(day), timelineCard(day.Timeline), catalogCard(day), decidersCard(day)}
}

// kpi is one headline figure.
type kpi struct {
	Label, Value, Note string
	// Tone is the status color that flags the note, if any.
	Tone string
}

// kpis are the overview's headline figures: traffic, then routing.
func (s stats) kpis() []kpi {
	summary := s.Summary
	decisions, failovers, upgrades := total(summary.Decisions), total(summary.Failovers), total(summary.Upgrades)
	figure := func(n int) string {
		if s.Err != nil {
			return "—"
		}
		return compact(n)
	}
	latency := kpi{Label: "Decider latency", Value: "—", Note: "no decisions in 24h"}
	if p50, ok := summary.LatencyMS["p50"]; ok {
		latency.Value = duration(time.Duration(p50) * time.Millisecond)
		latency.Note = "median, p95 " + duration(time.Duration(summary.LatencyMS["p95"])*time.Millisecond)
	}
	requests := kpi{Label: "Requests", Value: figure(s.Requests), Note: "no errors"}
	if s.Errors > 0 {
		requests.Note, requests.Tone = number(s.Errors)+" errors, "+percent(s.Errors, s.Requests), "bg-critical"
	}
	switches := kpi{Label: "Failovers", Value: figure(failovers), Note: busiest(summary.Failovers)}
	if failovers > 0 {
		switches.Tone = "bg-serious"
	}
	return []kpi{
		{Label: "Pinned threads", Value: figure(s.Threads), Note: number(decisions) + " new in 24h"},
		requests,
		{Label: "Decisions", Value: figure(decisions), Note: number(summary.Overrides) + " re-pinned by hand"},
		latency,
		{Label: "Escalations", Value: figure(summary.Escalated), Note: percent(summary.Escalated, decisions) + " of decisions"},
		{Label: "Second opinions", Value: figure(summary.Consulted), Note: percent(summary.Consulted, decisions) + " of decisions"},
		switches,
		{Label: "Capability upgrades", Value: figure(upgrades), Note: busiest(summary.Upgrades)},
	}
}

// busiest names the most frequent model switch of a "from->to" tally.
func busiest(switches map[string]int) string {
	if len(switches) == 0 {
		return "none in 24h"
	}
	top := ""
	for pair, count := range switches {
		if top == "" || count > switches[top] || count == switches[top] && pair < top {
			top = pair
		}
	}
	note := strings.Replace(top, "->", " → ", 1)
	if more := len(switches) - 1; more > 0 {
		note += " and " + strconv.Itoa(more) + " more"
	}
	return note
}

// rank is one entry of a breakdown with its share of the total.
type rank struct {
	Name  string
	Count int
	Share float64
}

// ranked orders a breakdown by count, naming its keys with name.
func ranked(counts map[string]int, name func(string) string) []rank {
	sum := total(counts)
	ranks := make([]rank, 0, len(counts))
	for key, count := range counts {
		ranks = append(ranks, rank{Name: name(key), Count: count, Share: float64(count) / float64(max(sum, 1))})
	}
	slices.SortFunc(ranks, func(a, b rank) int { return cmp.Or(b.Count-a.Count, cmp.Compare(a.Name, b.Name)) })
	return ranks
}

// verdict is a state in words with its status color.
type verdict struct {
	Text, Dot string
}

// healthOf sums up the catalog's state.
func healthOf(status catalog.Status) verdict {
	switch {
	case status.Healthy():
		return verdict{"Healthy", "bg-good"}
	case status.Loaded:
		return verdict{strconv.Itoa(len(status.Missing)) + " missing", "bg-critical"}
	case status.Error != "":
		return verdict{"Unreachable", "bg-critical"}
	}
	return verdict{"Not checked yet", "bg-warning"}
}

// count adds up the series of a part-to-whole.
func count(series []share) int {
	sum := 0
	for _, s := range series {
		sum += s.Count
	}
	return sum
}
