// Package limit caps concurrent upstream requests per model. Requests over
// the cap wait for a slot in arrival order. The cap adapts to the upstream:
// a throttling reply shrinks it to the requests still running, and it grows
// back by one every recovery interval without one, admitting waiters as it
// grows.
package limit

import (
	"context"
	"slices"
	"sync"
	"time"
)

// Stats is a gate's state.
type Stats struct {
	Model string
	// Limit is the adaptive cap, at most Max.
	Limit    int
	Max      int
	InFlight int
	Waiting  int
	// Throttles counts throttling replies; Throttled is the latest.
	Throttles int64
	Throttled time.Time
	Spill     string
}

// Gate caps one model's concurrent requests.
type Gate struct {
	model   string
	max     int
	spill   string
	recover time.Duration
	now     func() time.Time

	mu        sync.Mutex
	limit     int
	inflight  int
	waiters   []chan struct{}
	throttles int64
	throttled time.Time
	// grown is when the cap last grew or shrank.
	grown time.Time
	// wake grows the cap at the next recovery while requests wait under a
	// shrunken cap, so they are admitted even when no running request ends.
	wake *time.Timer
}

// NewGate returns a gate of max concurrent requests for model whose cap
// grows back by one every recover after a throttle. spill, when set, is
// the model new threads go to while the gate is saturated.
func NewGate(model string, max int, spill string, recover time.Duration) *Gate {
	return &Gate{model: model, max: max, spill: spill, recover: recover, now: time.Now, limit: max}
}

// Acquire waits for a slot, in arrival order, until ctx ends. The returned
// release frees the slot; calling it again does nothing.
func (g *Gate) Acquire(ctx context.Context) (release func(), err error) {
	g.mu.Lock()
	g.dispatch()
	if g.inflight < g.limit && len(g.waiters) == 0 {
		g.inflight++
		g.mu.Unlock()
		return g.releaser(), nil
	}
	ready := make(chan struct{})
	g.waiters = append(g.waiters, ready)
	g.schedule()
	g.mu.Unlock()
	select {
	case <-ready:
		return g.releaser(), nil
	case <-ctx.Done():
		g.mu.Lock()
		defer g.mu.Unlock()
		if i := slices.Index(g.waiters, ready); i >= 0 {
			g.waiters = slices.Delete(g.waiters, i, i+1)
			g.schedule()
		} else {
			// The slot was handed over as the context ended: pass it on.
			g.inflight--
			g.dispatch()
		}
		return nil, ctx.Err()
	}
}

func (g *Gate) releaser() func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			g.mu.Lock()
			defer g.mu.Unlock()
			g.inflight--
			g.dispatch()
		})
	}
}

// dispatch grows the cap and hands free slots to the waiters, oldest
// first.
func (g *Gate) dispatch() {
	g.grow()
	for g.inflight < g.limit && len(g.waiters) > 0 {
		next := g.waiters[0]
		g.waiters = g.waiters[1:]
		g.inflight++
		close(next)
	}
	g.schedule()
}

// schedule arms the wake timer for the next recovery while requests wait
// under a cap below the maximum, and stops it once none do.
func (g *Gate) schedule() {
	waiting := g.limit < g.max && len(g.waiters) > 0
	switch {
	case waiting && g.wake == nil:
		var wake *time.Timer
		wake = time.AfterFunc(g.grown.Add(g.recover).Sub(g.now()), func() {
			g.mu.Lock()
			defer g.mu.Unlock()
			if g.wake == wake {
				g.wake = nil
				g.dispatch()
			}
		})
		g.wake = wake
	case !waiting && g.wake != nil:
		g.wake.Stop()
		g.wake = nil
	}
}

// Throttled shrinks the cap to the requests running besides the caller's,
// which still holds its slot, and at least one.
func (g *Gate) Throttled() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.limit = max(1, min(g.limit, g.inflight-1))
	g.throttles++
	g.throttled = g.now()
	g.grown = g.throttled
	g.schedule()
}

// grow raises the cap by one for every recovery interval since it last
// changed.
func (g *Gate) grow() {
	if g.limit >= g.max || g.grown.IsZero() {
		return
	}
	if steps := int(g.now().Sub(g.grown) / g.recover); steps > 0 {
		g.limit = min(g.max, g.limit+steps)
		g.grown = g.grown.Add(time.Duration(steps) * g.recover)
	}
}

// Saturated reports local load or recent upstream throttling. Other clients
// may occupy the provider's slots even when this router has none in flight;
// new threads spill for one recovery interval while existing pins keep retrying.
func (g *Gate) Saturated() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dispatch()
	return g.inflight >= g.limit || len(g.waiters) > 0 ||
		!g.throttled.IsZero() && g.now().Before(g.throttled.Add(g.recover))
}

// Stats is the gate's state.
func (g *Gate) Stats() Stats {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dispatch()
	return Stats{Model: g.model, Limit: g.limit, Max: g.max, InFlight: g.inflight, Waiting: len(g.waiters),
		Throttles: g.throttles, Throttled: g.throttled, Spill: g.spill}
}

// Limiter holds the gates of the capped models.
type Limiter struct {
	gates map[string]*Gate
	order []string
}

// New returns a limiter over the gates.
func New(gates ...*Gate) *Limiter {
	l := &Limiter{gates: map[string]*Gate{}}
	for _, g := range gates {
		l.gates[g.model] = g
		l.order = append(l.order, g.model)
	}
	return l
}

// Gate is the model's gate, nil when the model is not capped.
func (l *Limiter) Gate(model string) *Gate {
	return l.gates[model]
}

// Spill is the model a new thread decided for model goes to instead, while
// model's gate is saturated.
func (l *Limiter) Spill(model string) (string, bool) {
	g := l.gates[model]
	if g == nil || g.spill == "" || !g.Saturated() {
		return "", false
	}
	return g.spill, true
}

// Stats is every gate's state, in configuration order.
func (l *Limiter) Stats() []Stats {
	stats := make([]Stats, 0, len(l.order))
	for _, model := range l.order {
		stats = append(stats, l.gates[model].Stats())
	}
	return stats
}
