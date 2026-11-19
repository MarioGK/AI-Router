package learn

import (
	"math"
	"math/rand/v2"
	"testing"
	"time"
)

var models = []string{"glm-5.3", "gpt-6.1-sol", "claude-opus-5-5"}

var levels = []string{"low", "medium", "high"}

func TestAPriorWeighsNeighboursBySimilarityRecencyAndLabel(t *testing.T) {
	now := start.Add(40 * 24 * time.Hour)
	halfLife := 7 * 24 * time.Hour
	neighbours := []neighbour{
		{Similarity: 1, Label: Label{Decision: 1, At: now, Model: "gpt-6.1-sol", Level: "high", Weight: ReviewedWeight}},
		// A week old: it counts half.
		{Similarity: 1, Label: Label{Decision: 2, At: now.Add(-halfLife), Model: "glm-5.3", Level: "low", Weight: ReviewedWeight}},
		{Similarity: 0.5, Label: Label{Decision: 3, At: now, Model: "gpt-6.1-sol", Weight: SufficedWeight}},
		// A model no longer offered, and a level-less label, bear on
		// nothing they do not name.
		{Similarity: 1, Label: Label{Decision: 4, At: now, Model: "retired-model", Level: "medium", Weight: ReviewedWeight}},
	}
	prior := build(neighbours, models, levels, now, halfLife, 0.5, 0.25)
	mass := map[string]float64{"glm-5.3": smoothing + 0.5, "gpt-6.1-sol": smoothing + 1 + 0.15, "claude-opus-5-5": smoothing}
	total := mass["glm-5.3"] + mass["gpt-6.1-sol"] + mass["claude-opus-5-5"]
	for id, m := range mass {
		if math.Abs(prior.Models[id]-m/total) > 1e-9 {
			t.Errorf("P(%s) = %v, want %v", id, prior.Models[id], m/total)
		}
	}
	if _, ok := prior.Models["retired-model"]; ok {
		t.Error("the prior offers a model that is not offered")
	}
	if math.Abs(prior.Neighbours-2.0) > 1e-9 || math.Abs(prior.Weight-0.5*2/7) > 1e-9 {
		t.Errorf("neighbours = %v, weight %v", prior.Neighbours, prior.Weight)
	}
	// Levels: high 1, low 0.5 and medium 1 (the retired model's level).
	if math.Abs(prior.LevelNeighbours-2.5) > 1e-9 || math.Abs(prior.LevelWeight-0.25*2.5/7.5) > 1e-9 ||
		prior.Levels["high"] <= prior.Levels["low"] {
		t.Errorf("levels = %v over %v, weight %v", prior.Levels, prior.LevelNeighbours, prior.LevelWeight)
	}
	if len(prior.Similar) != 4 || prior.Similar[1].Event != 2 || prior.Similar[2].Similarity != 0.5 {
		t.Errorf("similar = %+v", prior.Similar)
	}
	if empty := build(nil, models, levels, now, halfLife, 0.5, 0.5); empty.Models != nil || empty.Weight != 0 ||
		empty.LevelWeight != 0 || empty.Fit != 0.5 {
		t.Errorf("a prior without neighbours = %+v", empty)
	}
}

// draw picks an id from a distribution.
func draw(r *rand.Rand, p map[string]float64) string {
	x := r.Float64()
	for _, id := range models {
		if x -= p[id]; x < 0 {
			return id
		}
	}
	return models[len(models)-1]
}

// random is a random distribution over the models.
func random(r *rand.Rand) map[string]float64 {
	p := map[string]float64{}
	total := 0.0
	for _, id := range models {
		p[id] = r.ExpFloat64()
		total += p[id]
	}
	for id := range p {
		p[id] /= total
	}
	return p
}

func TestTheFitKeepsTheHistoryOutWhenItIsNoise(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	var replays []replay
	for range 300 {
		// The deciders are right as often as they say; the history is
		// unrelated to the task.
		verdict := random(r)
		replays = append(replays, replay{verdict: verdict, prior: random(r), neighbours: 15, label: draw(r, verdict),
			weight: SufficedWeight})
	}
	if w := fitWeight(replays); w > 0.05 {
		t.Fatalf("fit = %v on noise, want about 0", w)
	}
}

func TestTheFitTrustsTheHistoryWhenItPredicts(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	var replays []replay
	for range 300 {
		// The history knows where tasks belong; the deciders guess.
		prior := random(r)
		replays = append(replays, replay{verdict: random(r), prior: prior, neighbours: 15, label: draw(r, prior),
			weight: ReviewedWeight})
	}
	if w := fitWeight(replays); w < 0.5 {
		t.Fatalf("fit = %v on a predictive history, want well above 0", w)
	}
	if w := fitWeight(replays[:minLabels-1]); w != 0 {
		t.Fatalf("fit = %v below %d labels, want 0", w, minLabels)
	}
}
