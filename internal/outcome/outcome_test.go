package outcome

import (
	"context"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
)

var rules = Rules{Ladder: []string{"deepseek-flash", "glm-5.3-flash", "glm-5.3", "gpt-6.1-sol", "claude-opus-5-5"},
	Long: 200, Tiny: 3}

var start = time.Date(2026, 9, 1, 9, 0, 0, 0, time.UTC)

// decisions numbers the fake threads' decisions.
var decisions int64

// thread is a settled thread decided for model at level that ran requests
// for active time over an hour's span.
func thread(key, model, level string, requests int, active time.Duration, changes ...pin.Event) pin.ThreadActivity {
	decisions++
	return pin.ThreadActivity{Key: key, Decided: model, Level: level, Model: model, Decision: decisions,
		Requests: requests, First: start, Last: start.Add(time.Hour), Active: active, Cost: 0.5, Changes: changes}
}

func kinds(o Outcome) []string {
	var list []string
	for _, s := range o.Signals {
		list = append(list, s.Kind)
	}
	return list
}

func TestAnOutcomeNamesHowTheThreadWent(t *testing.T) {
	cases := []struct {
		name      string
		thread    pin.ThreadActivity
		signals   []string
		escalated bool
		moved     bool
	}{
		{"a quiet thread", thread("a", "glm-5.3", "medium", 40, 10*time.Minute), nil, false, false},
		{"an operator moved it up", thread("b", "glm-5.3-flash", "low", 40, time.Minute,
			pin.Event{Kind: pin.Override, Previous: "glm-5.3-flash", Model: "gpt-6.1-sol"}), []string{RepinUp}, true, true},
		{"an operator moved it down", thread("c", "claude-opus-5-5", "high", 40, time.Minute,
			pin.Event{Kind: pin.Override, Previous: "claude-opus-5-5", Model: "glm-5.3"}), []string{RepinDown}, false, true},
		{"the router decided it again, up", thread("d", "deepseek-flash", "low", 40, time.Minute,
			pin.Event{Kind: Redecision, Previous: "deepseek-flash", Model: "glm-5.3"}), []string{RedecisionUp}, true, true},
		{"a quota failover says nothing", thread("e", "gpt-6.1-sol", "high", 40, time.Minute,
			pin.Event{Kind: pin.Failover, Previous: "gpt-6.1-sol", Model: "claude-opus-5-5", Detail: "upstream status 429"}),
			[]string{QuotaFailover}, false, true},
		{"a server error failover up", thread("f", "glm-5.3", "medium", 40, time.Minute,
			pin.Event{Kind: pin.Failover, Previous: "glm-5.3", Model: "gpt-6.1-sol", Detail: "upstream status 529"}),
			[]string{Failover}, false, true},
		{"images and a migration", thread("g", "glm-5.3", "medium", 40, time.Minute,
			pin.Event{Kind: pin.CapabilityUpgrade, Previous: "glm-5.3", Model: "glm-5.3-flash"},
			pin.Event{Kind: pin.Migration, Previous: "glm-5.3-flash", Model: "gpt-6.1-sol", Detail: "retired"}),
			[]string{CapabilityUpgrade, Migration}, false, true},
		{"a level-only re-pin", thread("l", "glm-5.3", "low", 40, time.Minute,
			pin.Event{Kind: pin.Override, Previous: "glm-5.3", Model: "glm-5.3", Level: "high"}), nil, false, true},
		{"back up to the decision is no miss", thread("m", "glm-5.3", "low", 40, time.Minute,
			pin.Event{Kind: pin.Override, Previous: "glm-5.3", Model: "glm-5.3-flash"},
			pin.Event{Kind: pin.Override, Previous: "glm-5.3-flash", Model: "glm-5.3"}), []string{RepinDown, RepinUp}, false, true},
		{"a redecision back to the decision is no miss", thread("n", "glm-5.3", "low", 40, time.Minute,
			pin.Event{Kind: Redecision, Previous: "glm-5.3", Model: "deepseek-flash"},
			pin.Event{Kind: Redecision, Previous: "deepseek-flash", Model: "glm-5.3"}), []string{RedecisionDown, RedecisionUp}, false, true},
		{"a light marathon", thread("h", "glm-5.3-flash", "low", 201, time.Hour), []string{RanLong}, false, false},
		{"a marathon after a quota failover is no light marathon", thread("o", "deepseek-flash", "low", 500, time.Hour,
			pin.Event{Kind: pin.Failover, Previous: "deepseek-flash", Model: "gpt-6.1-sol", Detail: "upstream status 429"}),
			[]string{QuotaFailover}, false, true},
		{"a tiny thread moved off a heavy model", thread("p", "claude-opus-5-5", "low", 2, time.Minute,
			pin.Event{Kind: pin.Migration, Previous: "claude-opus-5-5", Model: "gpt-6.1-sol", Detail: "retired"}),
			[]string{Migration}, false, true},
		{"a middle marathon is no sign", thread("i", "glm-5.3", "low", 500, time.Hour), nil, false, false},
		{"a tiny heavy thread", thread("j", "claude-opus-5-5", "xhigh", 3, time.Minute), []string{StayedTiny}, false, false},
		{"a move off the ladder", thread("k", "glm-5.3", "low", 40, time.Minute,
			pin.Event{Kind: pin.Override, Previous: "glm-5.3", Model: "sonnet-retired"}), nil, false, true},
	}
	for _, c := range cases {
		o := Derive(c.thread, rules)
		if !slices.Equal(kinds(o), c.signals) || o.Escalated != c.escalated || o.Moved != c.moved {
			t.Errorf("%s: signals %v escalated %v moved %v, want %v %v %v", c.name, kinds(o), o.Escalated, o.Moved,
				c.signals, c.escalated, c.moved)
		}
	}
	o := Derive(thread("a", "glm-5.3", "medium", 40, 10*time.Minute), rules)
	if o.Span != time.Hour || o.Active != 10*time.Minute || o.Requests != 40 || o.Cost != 0.5 || !o.AtRisk {
		t.Fatalf("outcome = %+v", o)
	}
	if Derive(thread("z", "claude-opus-5-5", "high", 40, time.Minute), rules).AtRisk {
		t.Fatal("the strongest step has no stronger one to miss to")
	}
	up := Derive(thread("b", "glm-5.3-flash", "low", 40, time.Minute,
		pin.Event{Kind: pin.Override, Previous: "glm-5.3-flash", Model: "gpt-6.1-sol"}), rules)
	if text := up.Signals[0].Text; !strings.Contains(text, "glm-5.3-flash → gpt-6.1-sol") {
		t.Fatalf("signal text = %q", text)
	}
}

func TestAnExploredThreadThatMovedUpIsAMissOfTheExploredStep(t *testing.T) {
	explored := thread("x", "glm-5.3", "medium", 40, time.Minute,
		pin.Event{Kind: Redecision, Previous: "glm-5.3", Model: "gpt-6.1-sol"})
	explored.Explored, explored.Propensity = true, 0.05
	o := Derive(explored, rules)
	if !o.Explored || o.Propensity != 0.05 || o.Model != "glm-5.3" || !o.Escalated {
		t.Fatalf("explored outcome = %+v", o)
	}
}

func TestSeveralSignalsOutrankOneOfTheSameStrength(t *testing.T) {
	one := Outcome{Signals: []Signal{{Strength: 0.5}}}
	two := Outcome{Signals: []Signal{{Strength: 0.3}, {Strength: 0.5}}}
	if math.Abs(two.Strength()-0.53) > 1e-12 || one.Strength() != 0.5 || (Outcome{}).Strength() != 0 {
		t.Fatalf("strengths = %v, %v", one.Strength(), two.Strength())
	}
}

// log is a fake source of settled threads.
type log struct {
	threads       []pin.ThreadActivity
	since, settle time.Time
}

func (l *log) ThreadActivities(_ context.Context, since, settled time.Time) ([]pin.ThreadActivity, error) {
	l.since, l.settle = since, settled
	return l.threads, nil
}

func learn(t *testing.T, weight float64, threads ...pin.ThreadActivity) *Table {
	t.Helper()
	table := New(&log{threads: threads}, rules, 30*24*time.Hour, weight)
	if err := table.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	return table
}

func near(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestAnEmptyLogLearnsTheTypicalTimeAndChangesNoPenalty(t *testing.T) {
	e := learn(t, 20).Estimate("glm-5.3", "medium")
	if !near(e.Hours, Typical.Hours()) || e.Factor != 1 || e.MissRate != 0 {
		t.Fatalf("estimate = %+v", e)
	}
	if e := New(nil, rules, time.Hour, 20).Estimate("glm-5.3", ""); e.Factor != 1 {
		t.Fatalf("a table without a source = %+v", e)
	}
}

func TestTheTableReadsTheSettledThreadsOfTheWindow(t *testing.T) {
	source := &log{}
	table := New(source, rules, 30*24*time.Hour, 20)
	now := start.Add(40 * 24 * time.Hour)
	table.now = func() time.Time { return now }
	if err := table.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !source.since.Equal(now.Add(-30*24*time.Hour)) || !source.settle.Equal(now.Add(-time.Hour)) {
		t.Fatalf("read %v to %v", source.since, source.settle)
	}
}

func TestTimePerTaskShrinksTowardsTheFleet(t *testing.T) {
	var threads []pin.ThreadActivity
	// Twenty glm-5.3 threads of an hour, twenty deepseek-flash threads of
	// six minutes, all at low.
	for i := range 20 {
		threads = append(threads, thread("glm"+string(rune('a'+i)), "glm-5.3", "low", 40, time.Hour))
		threads = append(threads, thread("ds"+string(rune('a'+i)), "deepseek-flash", "low", 40, 6*time.Minute))
	}
	// A re-pinned thread's time does not count.
	threads = append(threads, thread("moved", "deepseek-flash", "low", 40, 100*time.Hour,
		pin.Event{Kind: pin.Override, Previous: "deepseek-flash", Model: "glm-5.3"}))
	table := learn(t, 20, threads...)
	slow, fast := table.Estimate("glm-5.3", "low"), table.Estimate("deepseek-flash", "low")
	if slow.Timed != 20 || fast.Timed != 20 {
		t.Fatalf("timed %d and %d threads", slow.Timed, fast.Timed)
	}
	// The level's mean is (40*0.55 + 20*fleet)/60 with the fleet at
	// (40*0.55 + 20*0.25)/60; each model's slowness shrinks towards 1.
	fleet := (40*0.55 + 20*Typical.Hours()) / 60
	level := (40*0.55 + 20*fleet) / 60
	slowness := func(used float64) float64 {
		expected := 20 * level
		return (used + expected) / (expected + expected)
	}
	if !near(slow.Hours, level*slowness(20)) || !near(fast.Hours, level*slowness(2)) {
		t.Fatalf("hours = %v and %v, want %v and %v", slow.Hours, fast.Hours, level*slowness(20), level*slowness(2))
	}
	if slow.Hours <= fast.Hours || slow.Hours >= 1 || fast.Hours <= 0.1 {
		t.Fatalf("the estimates did not shrink between the evidence and the fleet: %v, %v", slow.Hours, fast.Hours)
	}
	// A marathon counts at most twenty typical tasks.
	marathon := learn(t, 0, thread("m", "glm-5.3", "high", 40, 1000*time.Hour))
	if e := marathon.Estimate("glm-5.3", "high"); !near(e.Hours, limit.Hours()) {
		t.Fatalf("a marathon counted %v hours", e.Hours)
	}
}

func TestTheMissRateShrinksTowardsTheFleetAndScalesThePenaltyOnEvidence(t *testing.T) {
	up := func(key, model string) pin.ThreadActivity {
		return thread(key, model, "low", 40, time.Minute, pin.Event{Kind: pin.Override, Previous: model, Model: "claude-opus-5-5"})
	}
	var threads []pin.ThreadActivity
	// deepseek-flash: 5 of 10 moved up; glm-5.3: 0 of 30; gpt-6.1-sol: 1 of
	// 4, too few to move its penalty; Opus has nowhere to go and does not
	// count.
	for i := range 10 {
		key := "ds" + string(rune('a'+i))
		if i < 5 {
			threads = append(threads, up(key, "deepseek-flash"))
		} else {
			threads = append(threads, thread(key, "deepseek-flash", "low", 40, time.Minute))
		}
	}
	for i := range 30 {
		threads = append(threads, thread("glm"+string(rune('a'+i)), "glm-5.3", "low", 40, time.Minute))
	}
	threads = append(threads, up("sol-a", "gpt-6.1-sol"))
	for i := range 3 {
		threads = append(threads, thread("sol"+string(rune('b'+i)), "gpt-6.1-sol", "low", 40, time.Minute))
	}
	for i := range 10 {
		threads = append(threads, thread("opus"+string(rune('a'+i)), "claude-opus-5-5", "low", 40, time.Minute))
	}
	table := learn(t, 20, threads...)
	fleet := 6.0 / 44
	ds := table.Estimate("deepseek-flash", "low")
	wantRate := (5 + 20*fleet) / 30
	if ds.Missed != 5 || ds.AtRisk != 10 || !near(ds.MissRate, wantRate) || !near(ds.Factor, wantRate/fleet) {
		t.Fatalf("deepseek-flash = %+v, want rate %v factor %v", ds, wantRate, wantRate/fleet)
	}
	glm := table.Estimate("glm-5.3", "low")
	if wantRate := (20 * fleet) / 50; !near(glm.MissRate, wantRate) || !near(glm.Factor, wantRate/fleet) || glm.Factor >= 1 {
		t.Fatalf("glm-5.3 = %+v", glm)
	}
	if sol := table.Estimate("gpt-6.1-sol", "low"); sol.Factor != 1 || sol.AtRisk != 4 {
		t.Fatalf("gpt-6.1-sol on too little evidence = %+v", sol)
	}
	if opus := table.Estimate("claude-opus-5-5", "low"); opus.AtRisk != 0 || opus.Factor != 1 {
		t.Fatalf("the strongest step = %+v", opus)
	}
	if other := table.Estimate("deepseek-flash", "high"); other.Factor != 1 || !near(other.MissRate, fleet) {
		t.Fatalf("a cell without threads = %+v", other)
	}
	// Without weight the factor is the raw ratio, bounded.
	raw := learn(t, 0, threads...).Estimate("deepseek-flash", "low")
	if !near(raw.Factor, min(0.5/fleet, maxFactor)) {
		t.Fatalf("unshrunk factor = %v", raw.Factor)
	}
}

func TestTheQueueRanksUnreviewedSignsOfAMisroute(t *testing.T) {
	long := thread("long", "deepseek-flash", "low", 300, time.Hour)
	repinned := thread("repinned", "glm-5.3", "low", 40, time.Minute,
		pin.Event{Kind: pin.Override, Previous: "glm-5.3", Model: "claude-opus-5-5"})
	both := thread("both-long", "glm-5.3-flash", "low", 300, time.Hour,
		pin.Event{Kind: pin.Failover, Previous: "glm-5.3-flash", Model: "glm-5.3", Detail: "upstream status 500"})
	reviewed := thread("reviewed-one", "glm-5.3", "low", 40, time.Minute,
		pin.Event{Kind: pin.Override, Previous: "glm-5.3", Model: "claude-opus-5-5"})
	reviewed.Reviewed = true
	quota := thread("quota", "gpt-6.1-sol", "high", 40, time.Minute,
		pin.Event{Kind: pin.Failover, Previous: "gpt-6.1-sol", Model: "claude-opus-5-5", Detail: "upstream status 429"})
	quiet := thread("quiet", "glm-5.3", "low", 40, time.Minute)
	table := learn(t, 20, long, repinned, both, reviewed, quota, quiet)
	var keys []string
	for _, o := range table.Queue(10) {
		keys = append(keys, o.Key)
	}
	// A thread that failed over ran long elsewhere, not on its light model.
	if want := []string{"repinned", "long", "both-long"}; !slices.Equal(keys, want) {
		t.Fatalf("queue = %v, want %v", keys, want)
	}
	if len(table.Queue(1)) != 1 {
		t.Fatal("the queue ignored its limit")
	}
	table.Reviewed(repinned.Decision)
	if first := table.Queue(1)[0].Key; first != "long" {
		t.Fatalf("a reviewed decision stayed queued: %s", first)
	}
	counts := table.Counts()
	if counts.Threads != 6 || counts.Queued != 2 || counts.Signals[RepinUp] != 2 || counts.Signals[RanLong] != 1 ||
		counts.Signals[QuotaFailover] != 1 || counts.Signals[Failover] != 1 || counts.Signals[StayedTiny] != 0 {
		t.Fatalf("counts = %+v", counts)
	}
	if _, listed := counts.Signals[Migration]; !listed {
		t.Fatal("a signal no thread showed is not counted as 0")
	}
}

func TestAThreadWhoseModelChangedTeachesNoTime(t *testing.T) {
	table := learn(t, 0,
		thread("stayed", "glm-5.3", "low", 40, time.Hour),
		thread("failed-over", "glm-5.3", "low", 40, 9*time.Hour,
			pin.Event{Kind: pin.Failover, Previous: "glm-5.3", Model: "gpt-6.1-sol", Detail: "upstream status 529"}),
		thread("upgraded", "glm-5.3", "low", 40, 9*time.Hour,
			pin.Event{Kind: pin.CapabilityUpgrade, Previous: "glm-5.3", Model: "glm-5.3-flash"}),
		thread("migrated", "glm-5.3", "low", 40, 9*time.Hour,
			pin.Event{Kind: pin.Migration, Previous: "glm-5.3", Model: "gpt-6.1-sol", Detail: "retired"}))
	if e := table.Estimate("glm-5.3", "low"); e.Timed != 1 || !near(e.Hours, 1) {
		t.Fatalf("estimate = %+v, want only the thread that stayed", e)
	}
}

// slowLog holds a refresh between reading the log and publishing it.
type slowLog struct {
	log
	read, release chan struct{}
}

func (l *slowLog) ThreadActivities(ctx context.Context, since, settled time.Time) ([]pin.ThreadActivity, error) {
	threads, err := l.log.ThreadActivities(ctx, since, settled)
	l.read <- struct{}{}
	<-l.release
	return threads, err
}

func TestARefreshDoesNotUndoAReviewSavedWhileItRead(t *testing.T) {
	repinned := thread("repinned", "glm-5.3", "low", 40, time.Minute,
		pin.Event{Kind: pin.Override, Previous: "glm-5.3", Model: "claude-opus-5-5"})
	other := thread("other", "deepseek-flash", "low", 300, time.Minute)
	source := &slowLog{log: log{threads: []pin.ThreadActivity{repinned, other}}, read: make(chan struct{}), release: make(chan struct{})}
	table := New(source, rules, 30*24*time.Hour, 20)
	done := make(chan error)
	go func() { done <- table.Refresh(context.Background()) }()
	<-source.read
	// The review is saved after the log was read, before it is published.
	table.Reviewed(repinned.Decision)
	close(source.release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if queue := table.Queue(10); len(queue) != 1 || queue[0].Key != "other" {
		t.Fatalf("queue = %+v", queue)
	}
	// A later refresh reads the review from the log, and needs no keeping.
	repinned.Reviewed = true
	source.threads = []pin.ThreadActivity{repinned, other}
	go func() { <-source.read }()
	go func() { done <- table.Refresh(context.Background()) }()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if len(table.reviewed) != 0 {
		t.Fatalf("reviews kept after a refresh read them: %v", table.reviewed)
	}
}
