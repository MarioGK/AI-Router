package decide

import (
	"fmt"
	"slices"
	"strings"
)

const (
	PriorityAuto    = "auto"
	PriorityQuality = "quality"
	PrioritySpeed   = "speed"
	PriorityCost    = "cost"
)

// Preferences resolve trade-offs among suitable models without changing the
// capability ladder. An empty priority means full auto. Ranking is a soft
// preference, highest first; it never makes an unsuitable model eligible.
type Preferences struct {
	Priority string   `json:"priority,omitempty"`
	Ranking  []string `json:"ranking,omitempty"`
}

func (p Preferences) Validate() error {
	switch p.Priority {
	case "", PriorityAuto, PriorityQuality, PrioritySpeed, PriorityCost:
	default:
		return fmt.Errorf("priority must be auto, quality, speed, or cost")
	}
	if len(p.Ranking) > 128 {
		return fmt.Errorf("model ranking has more than 128 entries")
	}
	seen := make(map[string]bool, len(p.Ranking))
	for _, model := range p.Ranking {
		if strings.TrimSpace(model) == "" || model != strings.TrimSpace(model) || seen[model] {
			return fmt.Errorf("model ranking contains a blank, padded, or repeated model")
		}
		seen[model] = true
	}
	return nil
}

func (p Preferences) instructions() Instructions {
	framing := instructions
	switch p.Priority {
	case PriorityQuality:
		framing.Question = "Which offered model best fits this task when the operator prioritizes quality?"
		framing.Focus += " Operator priority: quality. Prefer stronger task fit and reduce the risk of insufficient capability; price and latency are secondary. Candidate task exclusions still apply."
	case PrioritySpeed:
		framing.Question = "Which suitable offered model can complete this task fastest?"
		framing.Focus += " Operator priority: speed. Prefer faster suitable models using the supplied latency estimates and criteria. Never assume speed from a model's name or its capability rank. Missing latency is unknown, not zero. Preserve sufficient capability and minimum sufficient reasoning."
	case PriorityCost:
		framing.Question = "Which suitable offered model can complete this task at the lowest expected cost?"
		framing.Focus += " Operator priority: cost. Prefer economical suitable models; preserve sufficient capability and minimum sufficient reasoning. Candidate task exclusions still apply."
	}
	if len(p.Ranking) > 0 {
		framing.Focus += " Operator model preference, highest first: " + strings.Join(p.Ranking, " -> ") + ". Use this order to break ties between similarly suitable models, after applying task fit and the operator priority. Unlisted models remain eligible. This preference is separate from the capability ladder."
	}
	return framing
}

func (c Candidate) description() string {
	text := c.Description
	if c.ExpectedLatencyMS > 0 {
		text += fmt.Sprintf(" Typical response latency: %d ms (operator estimate for comparable requests).", c.ExpectedLatencyMS)
	}
	if c.ExpectedCostUSD != nil {
		text += fmt.Sprintf(" Estimated cost per task: $%.6f (usage-based heuristic).", *c.ExpectedCostUSD)
	}
	return text
}

// Candidates adds cost evidence when the operator asks for a trade-off.
// Capability order and the original configuration stay unchanged.
func (s *Selection) Candidates(candidates []Candidate) []Candidate {
	if s.chain.economy == nil || s.preferences.Priority == "" || s.preferences.Priority == PriorityAuto {
		return candidates
	}
	prepared := slices.Clone(candidates)
	for i, candidate := range prepared {
		if cost, ok := s.chain.economy.Task(candidate.ID, "", candidates); ok {
			prepared[i].ExpectedCostUSD = new(cost)
		}
	}
	return prepared
}

// Prefer returns a request-local selection. Transport clients and circuit
// health remain shared; instructions and cost policy cannot leak to other
// concurrent requests. Callers validate preferences before taking a snapshot.
func (s *Selection) Prefer(preferences Preferences) *Selection {
	if preferences.Priority == PriorityAuto {
		preferences.Priority = ""
	}
	if preferences.Priority == "" && len(preferences.Ranking) == 0 && s.preferences.Priority == "" && len(s.preferences.Ranking) == 0 {
		return s
	}
	preferences.Ranking = slices.Clone(preferences.Ranking)
	view := &Selection{chain: s.chain, steps: slices.Clone(s.steps), preferences: preferences}
	framing := preferences.instructions()
	for i, step := range view.steps {
		switch decider := step.Decider.(type) {
		case *SystemOne:
			copy := *decider
			copy.framing = &framing
			view.steps[i].Decider = &copy
		case *Judge:
			copy := *decider
			copy.framing = &framing
			view.steps[i].Decider = &copy
		}
	}
	return view
}

func (s *SystemOne) instructions() Instructions {
	if s.framing != nil {
		return *s.framing
	}
	return instructions
}

func (j *Judge) instructions() Instructions {
	if j.framing != nil {
		return *j.framing
	}
	return instructions
}

func preferenceEconomy(base *Economy, priority string) *Economy {
	if base == nil || priority == PrioritySpeed {
		return nil
	}
	if priority != PriorityQuality && priority != PriorityCost {
		return base
	}
	copy := *base
	// These are explicit operating heuristics. Quality weighs an uncertain
	// capability miss four times more; cost uses one third of the normal
	// penalty. Neither permits downgrading the classifier's suitable tier.
	switch priority {
	case PriorityQuality:
		copy.Miss *= 4
	case PriorityCost:
		copy.Miss /= 3
	}
	return &copy
}
