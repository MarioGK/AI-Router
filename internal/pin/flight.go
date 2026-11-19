package pin

import (
	"context"
	"sync"
)

// Flight makes concurrent first requests of one thread share one decision.
type Flight struct {
	mu     sync.Mutex
	calls  map[string]*call
	closed bool
	work   sync.WaitGroup
}

type call struct {
	done chan struct{}
	pin  *Pin
	err  error
}

// DoContext runs decide once per key at a time. Concurrent callers share
// its result, but each may stop waiting when ctx is canceled. decide keeps
// running after a caller leaves and must have its own bounded context.
func (f *Flight) DoContext(ctx context.Context, key string, decide func() (*Pin, error)) (*Pin, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	f.mu.Lock()
	if f.closed {
		f.mu.Unlock()
		return nil, context.Canceled
	}
	if f.calls == nil {
		f.calls = map[string]*call{}
	}
	c, ok := f.calls[key]
	if !ok {
		c = &call{done: make(chan struct{})}
		f.calls[key] = c
		f.work.Go(func() {
			c.pin, c.err = decide()
			f.mu.Lock()
			delete(f.calls, key)
			close(c.done)
			f.mu.Unlock()
		})
	}
	f.mu.Unlock()

	select {
	case <-ctx.Done():
		return nil, context.Cause(ctx)
	case <-c.done:
		return c.pin, c.err
	}
}

// Close prevents new calls and waits for shared work to finish. It does
// not cancel decide; the owner must cancel its context before closing.
func (f *Flight) Close() {
	f.mu.Lock()
	f.closed = true
	f.mu.Unlock()
	f.work.Wait()
}
