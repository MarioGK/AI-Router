package dashboard

import (
	"cmp"
	"path"
	"slices"
	"strconv"
)

// hue is the color of a model family on every page and in every chart.
type hue int

// The hues, in the order charts stack them: the order keeps neighbors
// apart in both themes. Slate is every model outside the families.
const (
	blue hue = iota
	orange
	emerald
	violet
	pink
	yellow
	slate
)

// families match model ids to their hue; the first match wins.
var families = []struct {
	pattern string
	hue     hue
}{
	{"deepseek*", blue},
	{"glm*flash*", yellow},
	{"glm*", orange},
	{"gpt*", emerald},
	{"claude-opus*", violet},
	{"claude-sonnet*", pink},
}

func hueOf(model string) hue {
	for _, family := range families {
		if matched, _ := path.Match(family.pattern, model); matched {
			return family.hue
		}
	}
	return slate
}

// The classes of each hue, spelled out for Tailwind.
var (
	swatches = [...]string{
		blue: "bg-model-blue", orange: "bg-model-orange", emerald: "bg-model-emerald", violet: "bg-model-violet",
		pink: "bg-model-pink", yellow: "bg-model-yellow", slate: "bg-model-slate",
	}
	tints = [...]string{
		blue: "bg-model-blue/12 ring-model-blue/35", orange: "bg-model-orange/12 ring-model-orange/35",
		emerald: "bg-model-emerald/12 ring-model-emerald/35", violet: "bg-model-violet/12 ring-model-violet/35",
		pink: "bg-model-pink/12 ring-model-pink/35", yellow: "bg-model-yellow/12 ring-model-yellow/35",
		slate: "bg-model-slate/12 ring-model-slate/35",
	}
)

// swatch is the fill of a model's marks.
func swatch(model string) string {
	return swatches[hueOf(model)]
}

// tint is the background of a model's badge.
func tint(model string) string {
	return tints[hueOf(model)]
}

// share is one series of a part-to-whole: a model, or every slate model
// folded together, since color could not tell them apart.
type share struct {
	Label  string
	Hue    hue
	Count  int
	Folded []string
}

// shares splits counts into series in stacking order, leaving out models
// without any.
func shares(counts map[string]int) []share {
	var series []share
	other := share{Hue: slate}
	for model, count := range counts {
		switch {
		case count == 0:
		case hueOf(model) == slate:
			other.Count += count
			other.Folded = append(other.Folded, model)
		default:
			series = append(series, share{Label: model, Hue: hueOf(model), Count: count})
		}
	}
	slices.SortFunc(series, func(a, b share) int { return cmp.Or(cmp.Compare(a.Hue, b.Hue), cmp.Compare(a.Label, b.Label)) })
	switch len(other.Folded) {
	case 0:
		return series
	case 1:
		other.Label = other.Folded[0]
	default:
		slices.Sort(other.Folded)
		other.Label = "other (" + strconv.Itoa(len(other.Folded)) + ")"
	}
	return append(series, other)
}

// total adds up counts.
func total(counts map[string]int) int {
	sum := 0
	for _, count := range counts {
		sum += count
	}
	return sum
}
