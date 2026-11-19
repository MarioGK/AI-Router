package proxy

import (
	"context"
	"sync/atomic"
	"time"
)

// Watchdog ends a request's upstream calls once the upstream stays silent
// for longer than its timeout: while the router waits for response headers,
// or between two chunks of a streamed body. It counts only while a call is
// out, from Reset until Pause, and every Write is a sign of life, so it can
// tap a body as it streams. All but its timer run on the request's
// goroutine.
type Watchdog struct {
	timeout time.Duration
	timer   *time.Timer
	cancel  context.CancelFunc
	fired   atomic.Bool
}

// Watch returns the context a request's upstream calls run in and the
// watchdog that cancels it, not yet counting.
func Watch(ctx context.Context, timeout time.Duration) (context.Context, *Watchdog) {
	ctx, cancel := context.WithCancel(ctx)
	return ctx, &Watchdog{timeout: timeout, cancel: cancel}
}

// Reset starts the silence over, as when an upstream call goes out.
func (w *Watchdog) Reset() {
	if w.timer == nil {
		w.timer = time.AfterFunc(w.timeout, func() {
			w.fired.Store(true)
			w.cancel()
		})
		return
	}
	w.timer.Reset(w.timeout)
}

// Pause stops counting while the request waits on the router itself, as
// for a concurrency slot.
func (w *Watchdog) Pause() {
	if w.timer != nil {
		w.timer.Stop()
	}
}

// Write starts the silence over; it never fails.
func (w *Watchdog) Write(p []byte) (int, error) {
	w.Reset()
	return len(p), nil
}

// Stop ends the watch and releases its context.
func (w *Watchdog) Stop() {
	w.Pause()
	w.cancel()
}

// Fired reports whether the upstream went silent for too long; a nil
// watchdog never fires.
func (w *Watchdog) Fired() bool {
	return w != nil && w.fired.Load()
}

// Timeout is how long the upstream may stay silent.
func (w *Watchdog) Timeout() time.Duration {
	return w.timeout
}
