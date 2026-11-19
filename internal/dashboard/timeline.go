package dashboard

import (
	"fmt"
	"time"

	"github.com/a-h/templ"

	"github.com/MarioGK/AI-Router/internal/pin"
)

// timeline is the overview's requests-per-hour chart: a column per hour,
// stacked per model.
type timeline struct {
	Columns []column
	// Ticks are the gridline values; the last one tops the scale.
	Ticks  []int
	Legend []share
}

type column struct {
	Start     time.Time
	Requests  int
	Decisions int
	Segments  []share
}

// newTimeline lays the overview's buckets out in hourly columns from start.
func newTimeline(overview pin.Overview, start time.Time, hours int) timeline {
	counts := make([]map[string]int, hours)
	decisions := make([]int, hours)
	totals := map[string]int{}
	for _, bucket := range overview.Timeline {
		i := int(bucket.Start.Sub(start) / time.Hour)
		if i < 0 || i >= hours {
			continue
		}
		if counts[i] == nil {
			counts[i] = map[string]int{}
		}
		for model, n := range bucket.Requests {
			counts[i][model] += n
			totals[model] += n
		}
		decisions[i] += total(bucket.Decisions)
	}
	t := timeline{Columns: make([]column, hours), Legend: shares(totals)}
	peak := 0
	for i := range t.Columns {
		requests := total(counts[i])
		t.Columns[i] = column{
			Start: start.Add(time.Duration(i) * time.Hour), Requests: requests, Decisions: decisions[i], Segments: shares(counts[i]),
		}
		peak = max(peak, requests)
	}
	t.Ticks = ticks(peak)
	return t
}

// ticks are round gridline values from 0 up to at least peak, about three
// steps of 1, 2 or 5 times a power of ten.
func ticks(peak int) []int {
	step := 1
	for scale := 1; step*3 < peak; scale *= 10 {
		for _, factor := range []int{1, 2, 5} {
			if step = factor * scale; step*3 >= peak {
				break
			}
		}
	}
	values := []int{0}
	for value := step; values[len(values)-1] < peak; value += step {
		values = append(values, value)
	}
	if len(values) == 1 {
		values = append(values, step)
	}
	return values
}

// top is the value at the top of the scale.
func (t timeline) top() int {
	return t.Ticks[len(t.Ticks)-1]
}

// empty reports whether the period saw no requests.
func (t timeline) empty() bool {
	return len(t.Legend) == 0
}

// height places a value on the scale.
func (t timeline) height(value int) templ.SafeCSS {
	return templ.SafeCSS(fmt.Sprintf("height: %.2f%%", 100*float64(value)/float64(t.top())))
}

// bottom places a gridline on the scale.
func (t timeline) bottom(value int) templ.SafeCSS {
	return templ.SafeCSS(fmt.Sprintf("bottom: %.2f%%", 100*float64(value)/float64(t.top())))
}

// labelled reports whether a column carries an hour label on the axis.
func (c column) labelled() bool {
	return c.Start.Local().Hour()%6 == 0
}

// span is the hour a column covers.
func (c column) span() string {
	return c.Start.Local().Format("15:04") + "–" + c.Start.Add(time.Hour).Local().Format("15:04")
}

// count is a series' requests in the column.
func (c column) count(label string) int {
	for _, s := range c.Segments {
		if s.Label == label {
			return s.Count
		}
	}
	return 0
}

// grow sizes a series by its count within a stacked mark.
func grow(count int) templ.SafeCSS {
	return templ.SafeCSS(fmt.Sprintf("flex-grow: %d", count))
}

// width sizes a bar by a fraction.
func width(fraction float64) templ.SafeCSS {
	return templ.SafeCSS(fmt.Sprintf("width: %.2f%%", 100*max(0, min(1, fraction))))
}
