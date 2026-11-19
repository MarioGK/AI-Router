package dashboard

import (
	"strconv"
	"strings"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/outcome"
)

// hours is a time per task in minutes, or hours from two on.
func hours(h float64) string {
	if h < 2 {
		return strconv.Itoa(int(h*60+0.5)) + " min"
	}
	return strconv.FormatFloat(h, 'f', 1, 64) + " h"
}

// learnedLine is what thread outcomes taught about a cell, in a few words.
func learnedLine(e outcome.Estimate) string {
	return hours(e.Hours) + " · " + probability(e.MissRate) + " up"
}

// learnedNote is where a cell's time and miss rate come from, and what the
// rate does to the miss penalty.
func learnedNote(e outcome.Estimate, miss float64) string {
	var note strings.Builder
	if e.Timed == 0 {
		note.WriteString("Time per task: no settled thread timed yet, learned from the other levels and models. ")
	} else {
		note.WriteString("Time per task learned from " + counted(e.Timed, "settled thread") + ". ")
	}
	switch {
	case e.AtRisk == 0:
		note.WriteString("No settled thread with a stronger step above it yet: the fleet's miss rate stands in")
	default:
		note.WriteString(strconv.Itoa(e.Missed) + " of " + counted(e.AtRisk, "settled thread") +
			" an operator or a redecision moved up to a stronger model, " + probability(e.MissRate) + " shrunk towards the fleet")
	}
	if e.Factor != 1 {
		note.WriteString(", so a miss here costs " + multiple(miss*e.Factor) + " on top instead of " + multiple(miss))
	}
	return note.String() + "."
}

// stepNote is how learned time and miss rates entered a ladder step's
// score; empty when neither did.
func stepNote(costs *decide.Costs, model string) string {
	if costs == nil {
		return ""
	}
	var parts []string
	if value, ok := costs.Time[model]; ok {
		parts = append(parts, "task "+money(costs.Task[model])+" + "+hours(costs.Hours[model])+" of time ("+money(value)+")")
	}
	if miss, ok := costs.Miss[model]; ok {
		parts = append(parts, probability(costs.MissRate[model])+" of its threads moved up, so a miss costs "+
			multiple(miss)+" on top")
	}
	return strings.Join(parts, "; ")
}

// spent is what a queued thread took: its turns, active time and cost.
func spent(o outcome.Outcome) string {
	return counted(o.Requests, "turn") + " · " + duration(o.Active.Round(time.Second)) + " active · " + money(o.Cost)
}
