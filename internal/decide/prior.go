package decide

// Prior is what the history of similar past prompts suggests for a new
// one: a distribution over the offered candidates and one over the
// reasoning levels, built from the labelled decisions whose first prompts
// read most like it, and how much each weighs against the deciders'
// verdict. A decision blends it in before the economy or the threshold
// judges the verdict, so cost, time and miss logic are unchanged.
type Prior struct {
	// Models and Levels are the suggested distributions; empty when no
	// labelled neighbour bears on them.
	Models map[string]float64 `json:"models,omitempty"`
	Levels map[string]float64 `json:"levels,omitempty"`
	// Neighbours and LevelNeighbours are the effective numbers of
	// neighbours behind each distribution: each counts by how similar and
	// how recent it is.
	Neighbours      float64 `json:"neighbours,omitempty"`
	LevelNeighbours float64 `json:"level_neighbours,omitempty"`
	// Fit and LevelFit are the weights the history earned when it was last
	// refitted on its own labels; Weight and LevelWeight are what this
	// prior weighs in the blend, the fit shrunk by the neighbours behind it.
	Fit         float64 `json:"fit"`
	LevelFit    float64 `json:"level_fit"`
	Weight      float64 `json:"weight"`
	LevelWeight float64 `json:"level_weight"`
	// Similar are the neighbours, most similar first.
	Similar []Precedent `json:"similar,omitempty"`
}

// Precedent is one labelled past decision a prior was built from: its
// decision event, how similar its first prompt is (1 for the closest),
// and where its outcome says the task belonged, with how much that label
// counts.
type Precedent struct {
	Event      int64   `json:"event"`
	Similarity float64 `json:"similarity"`
	Model      string  `json:"model"`
	Level      string  `json:"level,omitempty"`
	Weight     float64 `json:"weight"`
}

// Learning is the part of a policy the history sets: the weights the
// prior earned at its last refit. Decisions under the same fingerprint
// blended the same way.
type Learning struct {
	Fit      float64 `json:"fit"`
	LevelFit float64 `json:"level_fit"`
}

// learning is the prior's part of the policy; nil when it changes no
// decision.
func (p *Prior) learning() *Learning {
	if p == nil || p.Fit == 0 && p.LevelFit == 0 {
		return nil
	}
	return &Learning{Fit: p.Fit, LevelFit: p.LevelFit}
}

// Learned returns a request-local selection that blends prior into every
// verdict; nil blends nothing.
func (s *Selection) Learned(prior *Prior) *Selection {
	if prior == nil && s.prior == nil {
		return s
	}
	view := *s
	view.prior = prior
	return &view
}

// blend mixes the prior into a normalized verdict: each probability
// becomes (1-w) times the verdict's plus w times the prior's, normalized
// over the candidates, and the choice the most likely candidate, the
// decider's own on a tie. The levels blend the same way with their own
// weight. A weight of zero leaves the verdict exactly as it was.
func (p *Prior) blend(verdict Verdict, candidates []Candidate, levels []Level) Verdict {
	if p == nil {
		return verdict
	}
	if p.Weight > 0 && len(p.Models) > 0 {
		ids := make([]string, len(candidates))
		for i, candidate := range candidates {
			ids[i] = candidate.ID
		}
		verdict.Probabilities = mix(verdict.Probabilities, p.Models, ids, p.Weight)
		best := verdict.Choice
		for _, id := range ids {
			if verdict.Probabilities[id] > verdict.Probabilities[best]+1e-12 {
				best = id
			}
		}
		verdict.Choice = best
	}
	if p.LevelWeight > 0 && len(p.Levels) > 0 && verdict.Levels != nil {
		ids := make([]string, len(levels))
		for i, level := range levels {
			ids[i] = level.ID
		}
		verdict.Levels = mix(verdict.Levels, p.Levels, ids, p.LevelWeight)
	}
	return verdict
}

// pair blends the prior into a pair question's probability on the
// stronger of two steps: the prior's own odds between the two, renormalized
// to them, weighed as in blend. ok is false when the prior weighs nothing
// or has no odds between the two.
func (p *Prior) pair(above float64, cheap, strong string) (float64, bool) {
	if p == nil || p.Weight <= 0 {
		return above, false
	}
	total := p.Models[cheap] + p.Models[strong]
	if total <= 0 {
		return above, false
	}
	return (1-p.Weight)*above + p.Weight*p.Models[strong]/total, true
}

// mix is (1-w)*a + w*b over ids, normalized; a when nothing is left.
func mix(a, b map[string]float64, ids []string, w float64) map[string]float64 {
	mixed := make(map[string]float64, len(ids))
	total := 0.0
	for _, id := range ids {
		mixed[id] = (1-w)*a[id] + w*b[id]
		total += mixed[id]
	}
	if total <= 0 {
		return a
	}
	for id := range mixed {
		mixed[id] /= total
	}
	return mixed
}
