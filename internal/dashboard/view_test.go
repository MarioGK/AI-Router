package dashboard

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
)

func TestFormats(t *testing.T) {
	now := time.Now()
	for _, c := range []struct{ got, want string }{
		{ago(now), "just now"},
		{ago(now.Add(-42 * time.Second)), "42s ago"},
		{ago(now.Add(-5 * time.Minute)), "5m ago"},
		{ago(now.Add(-3 * time.Hour)), "3h ago"},
		{ago(now.Add(-50 * time.Hour)), "2d ago"},
		{duration(400 * time.Microsecond), "<1 ms"},
		{duration(420 * time.Millisecond), "420 ms"},
		{duration(1300 * time.Millisecond), "1.30 s"},
		{duration(125 * time.Second), "2m5s"},
		{number(1234567), "1,234,567"},
		{number(-1234), "-1,234"},
		{compact(9999), "9,999"},
		{compact(12480), "12.5K"},
		{compact(3_400_000), "3.4M"},
		{probability(0.004), "0.4%"},
		{probability(0.826), "83%"},
		{percent(1, 0), "0%"},
		{money(0.000028), "<$0.0001"},
		{money(0.0061), "$0.0061"},
		{money(3.456), "$3.46"},
		{statusText(0), "no response"},
		{shortKey(strings.Repeat("f", 64)), "ffffffffffff"},
	} {
		if c.got != c.want {
			t.Errorf("got %q, want %q", c.got, c.want)
		}
	}
}

func TestDecisionsAreExplainedInWords(t *testing.T) {
	doubted := decide.Attempt{Decider: "jev", Outcome: decide.Doubted, Choice: "glm-5.3",
		Probabilities: map[string]float64{"glm-5.3": 0.65, "gpt-6.1-sol": 0.35}, Above: 0.35}
	basis := pin.Basis{Threshold: 0.1}
	for name, c := range map[string]struct {
		event pin.Event
		want  []string
	}{
		"a sure decider": {
			pin.Event{Decider: "jev", Model: "glm-5.3", Basis: basis, Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided,
				Choice: "glm-5.3", Probabilities: map[string]float64{"glm-5.3": 0.94}, Confidence: 0.9, Above: 0.06}}},
			[]string{"jev chose glm-5.3 (94%) with 90% confidence; 6% on stronger models stayed under the 10% threshold"},
		},
		"a doubt that stood": {
			pin.Event{Decider: "jev", Model: "gpt-6.1-sol", Escalated: true, Basis: basis, Attempts: []decide.Attempt{doubted,
				{Decider: "judge", Outcome: decide.Failed, Error: "context deadline exceeded"}}},
			[]string{
				"jev chose glm-5.3 (65%) but put 35% on stronger models, at least the 10% threshold, so it asked judge",
				"judge failed: context deadline exceeded",
				"Nothing settled the doubt, so jev's choice went one step up to gpt-6.1-sol",
			},
		},
		"a second opinion": {
			pin.Event{Decider: "judge", Model: "glm-5.3", Basis: basis, Attempts: []decide.Attempt{doubted,
				{Decider: "judge", Outcome: decide.Decided, Choice: "glm-5.3", Probabilities: map[string]float64{"glm-5.3": 0.8}, Above: 0.2,
					Reason: "Routine retries."}}},
			[]string{
				"jev chose glm-5.3 (65%) but put 35% on stronger models, at least the 10% threshold, so it asked judge",
				"judge chose glm-5.3 (80%) with 20% on stronger models; a second opinion settles the doubt as given",
			},
		},
		"a pair question that moves up": {
			pin.Event{Decider: "jev", Model: "gpt-6.1-sol", Escalated: true, Basis: basis, Attempts: []decide.Attempt{doubted,
				{Decider: "jev", Outcome: decide.Decided, Pair: []string{"glm-5.3", "gpt-6.1-sol"}, Choice: "gpt-6.1-sol", Above: 0.57}}},
			[]string{
				"jev chose glm-5.3 (65%) but put 35% on stronger models, at least the 10% threshold, so it was asked to compare glm-5.3 with gpt-6.1-sol",
				"jev compared glm-5.3 with gpt-6.1-sol and put 57% on gpt-6.1-sol, so the thread went one step up to gpt-6.1-sol",
			},
		},
		"a failed pair question": {
			pin.Event{Decider: "jev", Model: "gpt-6.1-sol", Escalated: true, Basis: basis, Attempts: []decide.Attempt{doubted,
				{Decider: "jev", Outcome: decide.Failed, Pair: []string{"glm-5.3", "gpt-6.1-sol"}, Error: "status 503"}}},
			[]string{
				"jev chose glm-5.3 (65%) but put 35% on stronger models, at least the 10% threshold, so it was asked to compare glm-5.3 with gpt-6.1-sol",
				"jev's comparison failed: status 503",
				"Nothing settled the doubt, so jev's choice went one step up to gpt-6.1-sol",
			},
		},
		"a spill": {
			pin.Event{Decider: "jev", Model: "gpt-6.1-sol", Basis: pin.Basis{Threshold: 0.1, Spilled: "glm-5.3"},
				Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "glm-5.3",
					Probabilities: map[string]float64{"glm-5.3": 0.9}, Above: 0.02}}},
			[]string{
				"jev chose glm-5.3 (90%); 2% on stronger models stayed under the 10% threshold",
				"glm-5.3 had every slot busy, so the thread went to gpt-6.1-sol instead",
			},
		},
		"an exploration": {
			pin.Event{Decider: "jev", Model: "glm-5.3", Escalated: true, Basis: pin.Basis{Threshold: 0.1,
				Costs:           &decide.Costs{Done: map[string]float64{"glm-5.3": 0.05, "gpt-6.1-sol": 0.04}},
				ExploreEligible: true, Explore: true, Propensity: 0.05, ExploredFrom: "gpt-6.1-sol"},
				Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "glm-5.3",
					Probabilities: map[string]float64{"glm-5.3": 0.7, "gpt-6.1-sol": 0.3}, Above: 0.3}}},
			[]string{
				"jev chose glm-5.3 (70%) but getting the task done looked cheaper on gpt-6.1-sol ($0.04) than on glm-5.3 ($0.05, counting the risk of a miss), so the thread went up to gpt-6.1-sol",
				"Explored one step cheaper (5% of borderline decisions): the thread went to glm-5.3 instead of gpt-6.1-sol, to learn whether the cheaper step would do",
			},
		},
		"a history that moved the choice": {
			pin.Event{Decider: "jev", Model: "gpt-6.1-sol", Choice: "gpt-6.1-sol", Basis: pin.Basis{Threshold: 0.1,
				Prior: &decide.Prior{Weight: 0.32, Similar: []decide.Precedent{{Event: 1, Model: "gpt-6.1-sol"},
					{Event: 2, Model: "gpt-6.1-sol"}, {Event: 3, Model: "glm-5.3"}}}},
				Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "glm-5.3",
					Probabilities: map[string]float64{"glm-5.3": 0.55, "gpt-6.1-sol": 0.45}, Above: 0.45}}},
			[]string{
				"jev chose glm-5.3 (55%); 45% on stronger models stayed under the 10% threshold",
				"History: 3 similar prompts, 2 belonged on gpt-6.1-sol; weight 0.32, which moved the choice from glm-5.3 to gpt-6.1-sol",
			},
		},
		"a history without weight": {
			pin.Event{Decider: "jev", Model: "glm-5.3", Choice: "glm-5.3", Basis: pin.Basis{Threshold: 0.1,
				Prior: &decide.Prior{Similar: []decide.Precedent{{Event: 1, Model: "glm-5.3"}}}},
				Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Decided, Choice: "glm-5.3",
					Probabilities: map[string]float64{"glm-5.3": 1}}}},
			[]string{
				"jev chose glm-5.3 (100%)",
				"History: 1 similar prompt, 1 belonged on glm-5.3; weight 0.00, so it changed nothing",
			},
		},
		"no user text": {
			pin.Event{Decider: "default", Model: "glm-5.3", Basis: basis},
			[]string{"There was no user text to judge, so the thread got the default model glm-5.3"},
		},
		"every decider failed": {
			pin.Event{Decider: "default", Model: "glm-5.3", Basis: basis, Attempts: []decide.Attempt{{Decider: "jev", Outcome: decide.Failed, Error: "down"}}},
			[]string{"jev failed: down", "No decider answered, so the thread got the default model glm-5.3"},
		},
	} {
		var got []string
		for _, s := range explain(c.event) {
			got = append(got, s.Text)
		}
		if !slices.Equal(got, c.want) {
			t.Errorf("%s:\n%q\nwant\n%q", name, got, c.want)
		}
	}
}

func TestModelsKeepTheirFamilyHue(t *testing.T) {
	for model, want := range map[string]hue{
		"deepseek-flash": blue, "glm-5.3": orange, "glm-5.3-flash": yellow, "gpt-6.1-sol": emerald,
		"claude-opus-5-5": violet, "claude-sonnet-5": pink, "claude-sonnet-4-6": pink, "qwen-4": slate,
	} {
		if got := hueOf(model); got != want {
			t.Errorf("hue of %s = %d, want %d", model, got, want)
		}
	}
	series := shares(map[string]int{"gpt-6.1-sol": 3, "deepseek-flash": 1, "mystery-a": 2, "mystery-b": 1, "glm-5.3": 0})
	var labels []string
	for _, s := range series {
		labels = append(labels, s.Label)
	}
	if strings.Join(labels, ",") != "deepseek-flash,gpt-6.1-sol,other (2)" || series[2].Count != 3 || series[2].Hue != slate {
		t.Fatalf("shares = %+v", series)
	}
	if single := shares(map[string]int{"mystery": 4}); single[0].Label != "mystery" || single[0].Hue != slate {
		t.Fatalf("a lone unknown model = %+v", single)
	}
}

func TestTimelineLaysBucketsOutByHour(t *testing.T) {
	start := time.Date(2026, 9, 28, 0, 0, 0, 0, time.Local)
	overview := pin.Overview{Timeline: []pin.Bucket{
		{Start: start, Requests: map[string]int{"glm-5.3": 30, "gpt-6.1-sol": 7}, Decisions: map[string]int{"glm-5.3": 2}},
		{Start: start.Add(5 * time.Hour), Requests: map[string]int{"gpt-6.1-sol": 12}},
		{Start: start.Add(30 * time.Hour), Requests: map[string]int{"gpt-6.1-sol": 99}},
	}}
	chart := newTimeline(overview, start, 24)
	if len(chart.Columns) != 24 || chart.Columns[0].Requests != 37 || chart.Columns[0].Decisions != 2 || chart.Columns[5].Requests != 12 {
		t.Fatalf("columns = %+v", chart.Columns[:6])
	}
	if !slices.Equal(chart.Ticks, []int{0, 20, 40}) || chart.Columns[0].count("gpt-6.1-sol") != 7 {
		t.Fatalf("ticks = %v", chart.Ticks)
	}
	if !chart.Columns[0].labelled() || chart.Columns[1].labelled() || chart.Columns[0].span() != "00:00–01:00" {
		t.Fatal("hour labels")
	}
	for peak, want := range map[int][]int{0: {0, 1}, 3: {0, 1, 2, 3}, 4: {0, 2, 4}, 1000: {0, 500, 1000}} {
		if got := ticks(peak); !slices.Equal(got, want) {
			t.Errorf("ticks(%d) = %v, want %v", peak, got, want)
		}
	}
}

func TestTailCatchesUpAndStartsOverWhenBehind(t *testing.T) {
	source := &fakeSource{}
	tail := &tail{source: source}
	tail.load(t.Context())
	if tail.poll(t.Context()) || len(tail.rows) != 0 {
		t.Fatal("an empty log changed")
	}
	source.log(pin.RequestLog{ID: 1})
	if !tail.poll(t.Context()) || len(tail.rows) != 1 || tail.after != 1 {
		t.Fatalf("the first request did not show: %+v", tail)
	}
	for id := int64(2); id <= tailBatch+1; id++ {
		source.log(pin.RequestLog{ID: id})
	}
	if !tail.poll(t.Context()) || len(tail.rows) != tailSize || tail.rows[0].ID != tailBatch+1 {
		t.Fatalf("a tail that fell behind did not start over from the newest: %d rows, top %d", len(tail.rows), tail.rows[0].ID)
	}
	source.log(pin.RequestLog{ID: tailBatch + 2})
	tail.poll(t.Context())
	if v := tail.view(); !v.arrived(tail.rows[0]) || v.arrived(tail.rows[1]) || tail.rows[0].ID != tailBatch+2 || len(tail.rows) != tailSize {
		t.Fatalf("only the newest request arrived: fresh %d, top %d", v.Fresh, tail.rows[0].ID)
	}
	if (requestTable{Rows: tail.rows}).arrived(tail.rows[0]) {
		t.Fatal("a static table flashes its rows")
	}
}
