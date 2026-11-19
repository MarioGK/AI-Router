package dashboard

import (
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/costs"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/usage"
)

// dash is text, or a dash when there is none.
func dash(text string) string {
	if text == "" {
		return "—"
	}
	return text
}

// ago is a time relative to now.
func ago(t time.Time) string {
	elapsed := time.Since(t)
	switch {
	case elapsed < 10*time.Second:
		return "just now"
	case elapsed < time.Minute:
		return strconv.Itoa(int(elapsed/time.Second)) + "s ago"
	case elapsed < time.Hour:
		return strconv.Itoa(int(elapsed/time.Minute)) + "m ago"
	case elapsed < 24*time.Hour:
		return strconv.Itoa(int(elapsed/time.Hour)) + "h ago"
	case elapsed < 30*24*time.Hour:
		return strconv.Itoa(int(elapsed/(24*time.Hour))) + "d ago"
	}
	return t.Local().Format("Jan 2, 2006")
}

// stamp is an absolute time.
func stamp(t time.Time) string {
	return t.Local().Format("2006-01-02 15:04:05 MST")
}

// shortKey is the prefix a pin key is shown by.
func shortKey(key string) string {
	return key[:min(12, len(key))]
}

// duration is a latency in its natural unit.
func duration(d time.Duration) string {
	switch {
	case d < time.Millisecond:
		return "<1 ms"
	case d < time.Second:
		return strconv.Itoa(int(d/time.Millisecond)) + " ms"
	case d < time.Minute:
		return strconv.FormatFloat(d.Seconds(), 'f', 2, 64) + " s"
	}
	return d.Truncate(time.Second).String()
}

// number groups the digits of n by thousands.
func number(n int) string {
	digits := strconv.Itoa(max(n, -n))
	var grouped strings.Builder
	if n < 0 {
		grouped.WriteByte('-')
	}
	for i, digit := range digits {
		if i > 0 && (len(digits)-i)%3 == 0 {
			grouped.WriteByte(',')
		}
		grouped.WriteRune(digit)
	}
	return grouped.String()
}

// money is a US dollar amount precise enough for single requests:
// $0.0042, $0.18, $1,240.
func money(v float64) string {
	switch {
	case v == 0:
		return "$0"
	case v < 0.0001:
		return "<$0.0001"
	case v < 0.01:
		return "$" + strconv.FormatFloat(v, 'f', 4, 64)
	case v < 1000:
		return "$" + strconv.FormatFloat(v, 'f', 2, 64)
	}
	return "$" + number(int(math.Round(v)))
}

// tokenSummary is a usage's input, output and cached input, abbreviated.
func tokenSummary(t usage.Tokens) string {
	summary := compact(int(t.Input)) + " in · " + compact(int(t.Output)) + " out"
	if cached := t.CacheRead + t.CacheWrite; cached > 0 {
		summary += " · " + compact(int(cached)) + " cached"
	}
	return summary
}

// compact abbreviates large counts: 12,480 is 12.5K.
func compact(n int) string {
	switch {
	case n >= 1_000_000:
		return strconv.FormatFloat(float64(n)/1_000_000, 'f', 1, 64) + "M"
	case n >= 10_000:
		return strconv.FormatFloat(float64(n)/1_000, 'f', 1, 64) + "K"
	}
	return number(n)
}

// percent is part's share of whole.
func percent(part, whole int) string {
	if whole == 0 {
		return "0%"
	}
	return probability(float64(part) / float64(whole))
}

// probability is a fraction as a percentage, with a decimal only when it
// would otherwise round to nothing.
func probability(p float64) string {
	switch {
	case p <= 0:
		return "0%"
	case p < 0.01:
		return strconv.FormatFloat(p*100, 'f', 1, 64) + "%"
	}
	return strconv.Itoa(int(p*100+0.5)) + "%"
}

// count words a number of things: "1 request", "3 requests".
func counted(n int, thing string) string {
	if n == 1 {
		return "1 " + thing
	}
	return number(n) + " " + thing + "s"
}

// efforts is the reasoning a candidate runs at: the range of levels the
// chooser may pick, else its fixed effort, else nothing.
func efforts(c config.Candidate) string {
	switch len(c.Efforts) {
	case 0:
		return c.Reasoning
	case 1:
		return c.Efforts[0]
	}
	return c.Efforts[0] + "–" + c.Efforts[len(c.Efforts)-1]
}

// deciderState is a decider's health in words.
func deciderState(h decide.DeciderHealth) string {
	switch {
	case !h.Resting.IsZero():
		return "resting until " + h.Resting.Local().Format("15:04:05")
	case h.Failures == 1:
		return "1 failure"
	case h.Failures > 1:
		return strconv.Itoa(h.Failures) + " failures in a row"
	}
	return "answering"
}

// deciderDot is the color of a decider's health.
func deciderDot(h decide.DeciderHealth) string {
	switch {
	case !h.Resting.IsZero():
		return "bg-critical"
	case h.Failures > 0:
		return "bg-serious"
	}
	return "bg-good"
}

// label turns an identifier such as side_request into words.
func label(identifier string) string {
	return strings.ReplaceAll(identifier, "_", " ")
}

// failed reports whether a request did not get a clean upstream answer.
func failed(request pin.RequestLog) bool {
	return request.Status == 0 || request.Status >= 400 || request.Error != ""
}

// statusText is how an upstream status is shown; a request without one
// found CLIProxy unreachable or was canceled by its client.
func statusText(status int) string {
	if status == 0 {
		return "no response"
	}
	return strconv.Itoa(status)
}

// statusDot is the status color of an upstream status.
func statusDot(status int) string {
	switch {
	case status >= 200 && status < 300:
		return "bg-good"
	case status >= 300 && status < 400:
		return "bg-fg-3"
	case status >= 400 && status < 500:
		return "bg-warning"
	}
	return "bg-critical"
}

// step is one line of a decision's explanation: what happened, the glyph
// marking it and the decider's own reason, when it gave one.
type step struct {
	Glyph  string
	Text   string
	Reason string
}

// explain tells in words how a decision came about, attempt by attempt:
// "jev chose glm-5.3 (65%) but put 35% on stronger models, at least the
// 10% threshold, so it asked judge".
func explain(e pin.Event) []step {
	threshold := probability(e.Basis.Threshold)
	// decided is the model the deciders settled on, before any spill or
	// exploration.
	decided := e.Model
	switch {
	case e.Basis.Explore && e.Basis.ExploredFrom != "":
		decided = e.Basis.ExploredFrom
	case e.Basis.Spilled != "":
		decided = e.Basis.Spilled
	}
	var steps []step
	for i, attempt := range e.Attempts {
		chose := attempt.Decider + " chose " + attempt.Choice
		if p, ok := attempt.Probabilities[attempt.Choice]; ok {
			chose += " (" + probability(p) + ")"
		}
		if attempt.Confidence > 0 {
			chose += " with " + probability(attempt.Confidence) + " confidence"
		}
		above := probability(attempt.Above) + " on stronger models"
		switch {
		case len(attempt.Pair) == 2 && attempt.Outcome == decide.Failed:
			steps = append(steps, step{Glyph: "alert", Text: attempt.Decider + "'s comparison failed: " + attempt.Error})
		case len(attempt.Pair) == 2:
			cheap, strong := attempt.Pair[0], attempt.Pair[1]
			text := attempt.Decider + " compared " + cheap + " with " + strong + " and put " + probability(attempt.Above) + " on " + strong
			if attempt.Blended > 0 {
				text += " (" + probability(attempt.Blended) + " with the history)"
			}
			if e.Basis.Costs != nil {
				text += ", where moving up pays from " + probability(attempt.Cutoff)
			}
			if decided == strong {
				steps = append(steps, step{Glyph: "arrow-up", Text: text + ", so the thread went one step up to " + strong})
			} else {
				steps = append(steps, step{Glyph: "check", Text: text + ", so " + cheap + " stood"})
			}
		case attempt.Outcome == decide.Failed:
			steps = append(steps, step{Glyph: "alert", Text: attempt.Decider + " failed: " + attempt.Error})
		case attempt.Outcome == decide.Skipped:
			steps = append(steps, step{Glyph: "pause", Text: attempt.Decider + " was not asked: " + attempt.Error})
		case attempt.Outcome == decide.Doubted && e.Basis.Costs != nil && i+1 < len(e.Attempts) && len(e.Attempts[i+1].Pair) == 2:
			strong := e.Attempts[i+1].Pair[1]
			steps = append(steps, step{Glyph: "question", Text: chose + cheaper(*e.Basis.Costs, attempt.Choice, strong) +
				", so it was asked to compare " + attempt.Choice + " with " + strong, Reason: attempt.Reason})
		case attempt.Outcome == decide.Doubted:
			text := chose + " but put " + above + ", at least the " + threshold + " threshold, "
			switch {
			case i+1 < len(e.Attempts) && len(e.Attempts[i+1].Pair) == 2:
				text += "so it was asked to compare " + attempt.Choice + " with " + e.Attempts[i+1].Pair[1]
			case i+1 < len(e.Attempts):
				text += "so it asked " + e.Attempts[i+1].Decider
			default:
				text += "and nobody was left to ask"
			}
			steps = append(steps, step{Glyph: "question", Text: text, Reason: attempt.Reason})
		default:
			text := chose
			switch {
			case e.Escalated && attempt.Decider == e.Decider && e.Basis.Costs != nil:
				text += cheaper(*e.Basis.Costs, attempt.Choice, decided) + ", so the thread went up to " + decided
			case attempt.Decider == e.Decider && e.Basis.Costs != nil:
				text += "; getting the task done cost least on " + attempt.Choice + " (" +
					money(e.Basis.Costs.Done[attempt.Choice]) + ", counting the risk of a miss)"
			case e.Escalated && attempt.Decider == e.Decider:
				text += " but put " + above + ", at least the " + threshold + " threshold, so the thread went one step up to " + decided
			case i > 0 && attempt.Above+1e-9 >= e.Basis.Threshold && e.Consulted():
				text += " with " + above + "; a second opinion settles the doubt as given"
			case attempt.Above > 0:
				text += "; " + above + " stayed under the " + threshold + " threshold"
			}
			steps = append(steps, step{Glyph: "check", Text: text, Reason: attempt.Reason})
		}
	}
	if e.Basis.Prior != nil && len(e.Basis.Prior.Similar) > 0 {
		steps = append(steps, step{Glyph: "search", Text: historyNote(e)})
	}
	if e.Basis.Spilled != "" {
		steps = append(steps, step{Glyph: "swap", Text: e.Basis.Spilled + " had every slot busy, so the thread went to " + e.Model + " instead"})
	}
	if e.Basis.Explore {
		steps = append(steps, step{Glyph: "spark", Text: "Explored one step cheaper (" + probability(e.Basis.Propensity) +
			" of borderline decisions): the thread went to " + e.Model + " instead of " + decided +
			", to learn whether the cheaper step would do"})
	}
	if e.Level != "" {
		text := e.Decider + " judged that the task needs " + e.Level + " reasoning"
		if p, ok := e.Levels[e.Level]; ok {
			text += " (" + probability(p) + ")"
		}
		steps = append(steps, step{Glyph: "gauge", Text: text})
	}
	switch last := len(e.Attempts) - 1; {
	case e.Decider == config.DefaultDecider && last < 0:
		steps = append(steps, step{Glyph: "spark", Text: "There was no user text to judge, so the thread got the default model " + decided})
	case e.Decider == config.DefaultDecider:
		steps = append(steps, step{Glyph: "spark", Text: "No decider answered, so the thread got the default model " + decided})
	case e.Escalated && e.Attempts[last].Outcome == decide.Failed:
		steps = append(steps, step{Glyph: "arrow-up", Text: "Nothing settled the doubt, so " + e.Decider +
			"'s choice went one step up to " + decided})
	}
	return steps
}

// historyNote is what the labelled decisions with similar first prompts
// suggested and how much it weighed: "History: 7 similar prompts, 5 belonged
// on gpt-6.1-sol; weight 0.32".
func historyNote(e pin.Event) string {
	prior := e.Basis.Prior
	went := map[string]int{}
	for _, p := range prior.Similar {
		went[p.Model]++
	}
	model := ""
	for id, n := range went {
		if model == "" || n > went[model] || n == went[model] && id < model {
			model = id
		}
	}
	text := "History: " + counted(len(prior.Similar), "similar prompt") + ", " + number(went[model]) + " belonged on " + model +
		"; weight " + weightText(prior.Weight)
	if prior.LevelWeight > 0 {
		text += ", " + weightText(prior.LevelWeight) + " on the reasoning"
	}
	if prior.Weight == 0 && prior.LevelWeight == 0 {
		return text + ", so it changed nothing"
	}
	for _, attempt := range e.Attempts {
		if len(attempt.Pair) == 0 && (attempt.Outcome == decide.Decided || attempt.Outcome == decide.Doubted) {
			if attempt.Choice != e.Choice {
				text += ", which moved the choice from " + attempt.Choice + " to " + e.Choice
			}
			break
		}
	}
	return text
}

// multiple is a factor as "3×".
func multiple(f float64) string {
	return strconv.FormatFloat(f, 'g', 3, 64) + "×"
}

// costNote is where an expected cost per task comes from.
func costNote(e costs.Estimate) string {
	if e.Threads == 0 {
		return "No settled thread yet: learned from the other levels and models"
	}
	return counted(e.Threads, "settled thread") + ", which cost " + money(e.Observed) + " per task as they ran"
}

// cheaper is why a verdict moved from its choice to a stronger step when
// the decision weighed the cost of getting the task done.
func cheaper(costs decide.Costs, choice, strong string) string {
	return " but getting the task done looked cheaper on " + strong + " (" + money(costs.Done[strong]) + ") than on " +
		choice + " (" + money(costs.Done[choice]) + ", counting the risk of a miss)"
}

// explanation is a decision's explanation in one line, for a tooltip.
func explanation(e pin.Event) string {
	var lines []string
	for _, s := range explain(e) {
		lines = append(lines, s.Text+".")
	}
	return strings.Join(lines, " ")
}
