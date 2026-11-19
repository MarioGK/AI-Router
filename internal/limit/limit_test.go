package limit

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"
)

func TestTheGateCaps(t *testing.T) {
	g := NewGate("glm-5.3", 2, "", time.Minute)
	first, _ := g.Acquire(context.Background())
	second, _ := g.Acquire(context.Background())
	if !g.Saturated() {
		t.Fatal("a full gate is not saturated")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := g.Acquire(ctx); err == nil {
		t.Fatal("a third request got a slot")
	}
	first()
	first() // a second release frees nothing
	if stats := g.Stats(); stats.InFlight != 1 || g.Saturated() {
		t.Fatalf("stats = %+v", stats)
	}
	second()
}

func TestWaitersAreServedInArrivalOrder(t *testing.T) {
	// One slot serializes the grants, so the order they are recorded in is
	// the order they were made in.
	g := NewGate("glm-5.3", 1, "", time.Minute)
	held, _ := g.Acquire(context.Background())
	var mu sync.Mutex
	var served []int
	var wg sync.WaitGroup
	for i := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			release, err := g.Acquire(context.Background())
			if err != nil {
				t.Error(err)
				return
			}
			mu.Lock()
			served = append(served, i)
			mu.Unlock()
			release()
		}()
		for g.Stats().Waiting != i+1 {
			time.Sleep(time.Millisecond)
		}
	}
	held()
	wg.Wait()
	if !slices.Equal(served, []int{0, 1, 2, 3}) {
		t.Fatalf("served %v, want arrival order", served)
	}
	if stats := g.Stats(); stats.InFlight != 0 || stats.Waiting != 0 {
		t.Fatalf("stats = %+v", stats)
	}
}

func TestAWaiterThatGivesUpLeavesTheQueue(t *testing.T) {
	g := NewGate("glm-5.3", 1, "", time.Minute)
	release, _ := g.Acquire(context.Background())
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := g.Acquire(ctx); err == nil {
		t.Fatal("a waiter past its deadline got a slot")
	}
	release()
	if stats := g.Stats(); stats.InFlight != 0 || stats.Waiting != 0 {
		t.Fatalf("stats = %+v", stats)
	}
	if next, err := g.Acquire(context.Background()); err != nil {
		t.Fatal(err)
	} else {
		next()
	}
}

func TestThrottlingShrinksTheCapUntilItRecovers(t *testing.T) {
	now := time.Date(2026, 9, 29, 12, 0, 0, 0, time.UTC)
	g := NewGate("glm-5.3", 4, "retired-model", time.Minute)
	g.now = func() time.Time { return now }
	var releases []func()
	for range 3 {
		release, _ := g.Acquire(context.Background())
		releases = append(releases, release)
	}
	// The third request is throttled: two others are running.
	g.Throttled()
	if stats := g.Stats(); stats.Limit != 2 || stats.Throttles != 1 || !stats.Throttled.Equal(now) {
		t.Fatalf("stats = %+v", stats)
	}
	releases[2]()
	if !g.Saturated() {
		t.Fatal("two requests under a cap of two leave room")
	}
	l := New(g)
	if spill, ok := l.Spill("glm-5.3"); !ok || spill != "retired-model" {
		t.Fatalf("Spill = %q, %v", spill, ok)
	}
	now = now.Add(90 * time.Second)
	if stats := g.Stats(); stats.Limit != 3 {
		t.Fatalf("after one recovery interval the cap is %d, want 3", stats.Limit)
	}
	if g.Saturated() {
		t.Fatal("the recovered cap still saturates")
	}
	if _, ok := l.Spill("glm-5.3"); ok {
		t.Fatal("an unsaturated gate spills")
	}
	now = now.Add(10 * time.Minute)
	if stats := g.Stats(); stats.Limit != 4 {
		t.Fatalf("the cap grew to %d, want the maximum", stats.Limit)
	}
	// A throttle with nothing else running keeps one slot.
	for _, release := range releases[:2] {
		release()
	}
	only, _ := g.Acquire(context.Background())
	g.Throttled()
	if stats := g.Stats(); stats.Limit != 1 {
		t.Fatalf("stats = %+v", stats)
	}
	only()
}

func TestARecoveringCapAdmitsTheWaiters(t *testing.T) {
	var mu sync.Mutex
	now := time.Date(2026, 9, 29, 12, 0, 0, 0, time.UTC)
	g := NewGate("glm-5.3", 4, "", 10*time.Millisecond)
	g.now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return now
	}
	armed := func() bool {
		g.mu.Lock()
		defer g.mu.Unlock()
		return g.wake != nil
	}
	// A throttle leaves the cap at the one long request still running.
	long, _ := g.Acquire(context.Background())
	defer long()
	throttled, _ := g.Acquire(context.Background())
	g.Throttled()
	throttled()

	// A waiter that gives up takes the wake timer with it.
	ctx, cancel := context.WithCancel(context.Background())
	gaveUp := make(chan error)
	go func() {
		_, err := g.Acquire(ctx)
		gaveUp <- err
	}()
	for g.Stats().Waiting != 1 {
		time.Sleep(time.Millisecond)
	}
	if !armed() {
		t.Fatal("a waiter under a shrunken cap armed no wake timer")
	}
	cancel()
	if err := <-gaveUp; err == nil || armed() {
		t.Fatalf("the waiter gave up with %v and left the timer armed: %v", err, armed())
	}

	// A waiting retry is admitted as the cap grows back, though the long
	// request still runs and nothing else touches the gate.
	admitted := make(chan func())
	go func() {
		release, err := g.Acquire(context.Background())
		if err != nil {
			t.Error(err)
		}
		admitted <- release
	}()
	for g.Stats().Waiting != 1 {
		time.Sleep(time.Millisecond)
	}
	mu.Lock()
	now = now.Add(time.Minute)
	mu.Unlock()
	select {
	case release := <-admitted:
		release()
	case <-time.After(5 * time.Second):
		t.Fatal("the recovered cap left its waiter in line")
	}
	if stats := g.Stats(); stats.Limit != 4 || stats.InFlight != 1 || stats.Waiting != 0 || armed() {
		t.Fatalf("stats = %+v, timer armed %v", stats, armed())
	}
}

func TestTheLimiterKnowsOnlyItsModels(t *testing.T) {
	l := New(NewGate("glm-5.3", 2, "", time.Minute), NewGate("glm-5.3-flash", 8, "", time.Minute))
	if l.Gate("retired-model") != nil {
		t.Fatal("an uncapped model has a gate")
	}
	if _, ok := l.Spill("retired-model"); ok {
		t.Fatal("an uncapped model spills")
	}
	stats := l.Stats()
	if len(stats) != 2 || stats[0].Model != "glm-5.3" || stats[1].Max != 8 {
		t.Fatalf("stats = %+v", stats)
	}
}

func TestRecentProviderThrottleSpillsEvenWithoutLocalLoad(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	g := NewGate("glm-5.3-flash", 3, "gpt-6.1-sol", time.Minute)
	g.now = func() time.Time { return now }
	limiter := New(g)
	release, err := g.Acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	g.Throttled() // Provider is saturated by callers outside this router.
	release()
	if g.Stats().InFlight != 0 {
		t.Fatal("local request was not released")
	}
	if model, ok := limiter.Spill("glm-5.3-flash"); !ok || model != "gpt-6.1-sol" {
		t.Fatalf("new threads ignored recent throttle: %q, %v", model, ok)
	}
	// Existing pinned requests still acquire the shrunken gate normally.
	retry, err := g.Acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	retry()
	now = now.Add(time.Minute)
	if _, ok := limiter.Spill("glm-5.3-flash"); ok {
		t.Fatal("recovered provider stayed withheld from new threads")
	}
}
