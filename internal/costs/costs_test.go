package costs

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/usage"
)

// log is a fake request log.
type log struct {
	usages        []pin.TaskUsage
	since, settle time.Time
	limit         int64
}

func (l *log) TaskUsages(_ context.Context, since, settled time.Time, limit int64) ([]pin.TaskUsage, error) {
	l.since, l.settle, l.limit = since, settled, limit
	return l.usages, nil
}

var prices = map[string]usage.Price{
	"deepseek-flash": {Input: 0.30, Output: 1.20, CacheRead: 0.006},
	"glm-5.3":        {Input: 1.40, Output: 4.40, CacheRead: 0.26},
}

func table(source Source, weight float64) *Table {
	return New(source, func(model string) (usage.Price, bool) {
		price, ok := prices[model]
		return price, ok
	}, func(model string) float64 {
		if model == "glm-5.3" {
			return 0.25
		}
		return 1
	}, 30*24*time.Hour, weight)
}

func near(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestAnEmptyLogPricesTheTypicalTask(t *testing.T) {
	costs := table(&log{}, 20)
	if err := costs.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if cost, ok := costs.Task("deepseek-flash", "high"); !ok || !near(cost, prices["deepseek-flash"].Cost(Typical)) {
		t.Fatalf("deepseek-flash = %v, %v", cost, ok)
	}
	if cost, _ := costs.Task("glm-5.3", ""); !near(cost, prices["glm-5.3"].Cost(Typical)*0.25) {
		t.Fatalf("glm-5.3 at its cost factor = %v", cost)
	}
	if _, ok := costs.Task("gpt-6.1-sol", "high"); ok {
		t.Fatal("a model without a price was priced")
	}
}

func TestTheLogTeachesTaskSizesAndVerbosity(t *testing.T) {
	small := usage.Tokens{Input: 10_000, Output: 2_000, CacheRead: 50_000}
	large := usage.Tokens{Input: 400_000, Output: 80_000, CacheRead: 4_000_000}
	source := &log{usages: []pin.TaskUsage{
		// DeepSeek does small tasks with the fleet's small ones; GLM takes
		// twice the tokens of the level's other threads.
		{Model: "deepseek-flash", Level: "low", Threads: 100, Tokens: times(small, 100), Cost: 1.5},
		{Model: "glm-5.3", Level: "low", Threads: 100, Tokens: times(small, 200), Cost: 3},
		{Model: "glm-5.3", Level: "xhigh", Threads: 40, Tokens: times(large, 40), Cost: 60},
	}}
	costs := table(source, 0)
	costs.now = func() time.Time { return time.Date(2026, 9, 29, 12, 0, 0, 0, time.UTC) }
	if err := costs.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !source.settle.Equal(costs.now().Add(-time.Hour)) || !source.since.Equal(costs.now().Add(-30*24*time.Hour)) ||
		source.limit != 20*Typical.Total() {
		t.Fatalf("read threads from %v, settled before %v, cut to %d tokens", source.since, source.settle, source.limit)
	}
	// Low tasks average 1.5 small ones; DeepSeek used 1, GLM 2 on them, and
	// GLM matched its level on xhigh.
	lowSize := times(small, 3).Total() / 2
	glmVerbosity := float64(200*small.Total()+40*large.Total()) / float64(100*lowSize+40*large.Total())
	deepseek, _ := costs.Task("deepseek-flash", "low")
	if want := prices["deepseek-flash"].Cost(scale(times(small, 3), 0.5*2.0/3)); math.Abs(deepseek-want) > 1e-6 {
		t.Fatalf("deepseek-flash at low = %v, want %v", deepseek, want)
	}
	glm, _ := costs.Task("glm-5.3", "xhigh")
	if want := prices["glm-5.3"].Cost(scale(large, glmVerbosity)) * 0.25; math.Abs(glm-want) > 1e-6 {
		t.Fatalf("glm-5.3 at xhigh = %v, want %v", glm, want)
	}
	if threads, learned := costs.Learned(); threads != 240 || learned.IsZero() {
		t.Fatalf("learned from %d threads at %v", threads, learned)
	}
	estimates := costs.Estimates([]string{"glm-5.3", "gpt-6.1-sol"}, []string{"low", "xhigh"})
	if len(estimates) != 2 || estimates[0].Threads != 100 || !near(estimates[0].Observed, 0.03*0.25) {
		t.Fatalf("estimates = %+v", estimates)
	}

	// An overwhelming prior weight leaves every model as verbose as the
	// fleet.
	costs.weight = 1e9
	if v := costs.verbosity("glm-5.3"); math.Abs(v-1) > 1e-6 {
		t.Fatalf("a weighted verbosity = %v, want 1", v)
	}
}

func times(t usage.Tokens, n int64) usage.Tokens {
	return usage.Tokens{Input: t.Input * n, Output: t.Output * n, CacheRead: t.CacheRead * n, CacheWrite: t.CacheWrite * n}
}
