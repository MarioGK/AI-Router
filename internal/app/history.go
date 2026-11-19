package app

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/stats"
	"github.com/MarioGK/AI-Router/internal/usage"
)

const (
	// historyBuffer is how many handled requests may wait for the writer.
	historyBuffer = 1024
	// maxFailure bounds a logged error summary.
	maxFailure = 160
)

// history writes the request log off the request path: one writer drains a
// bounded queue, and a request that finds the queue full is dropped and
// counted, so logging never delays or fails a response.
type history struct {
	write    func(context.Context, pin.RequestLog) error
	counters *stats.Counters
	log      *slog.Logger
	queue    chan pin.RequestLog
	stop     chan struct{}
	done     chan struct{}
	closing  sync.Once
}

func newHistory(write func(context.Context, pin.RequestLog) error, size int, counters *stats.Counters, log *slog.Logger) *history {
	h := &history{
		write: write, counters: counters, log: log, queue: make(chan pin.RequestLog, size),
		stop: make(chan struct{}), done: make(chan struct{}),
	}
	go h.run()
	return h
}

// add queues a request without ever blocking.
func (h *history) add(entry pin.RequestLog) {
	select {
	case h.queue <- entry:
	default:
		h.counters.Add("history.dropped")
	}
}

// Close writes the queued requests and stops the writer.
func (h *history) Close() {
	h.closing.Do(func() {
		close(h.stop)
		<-h.done
	})
}

func (h *history) run() {
	defer close(h.done)
	for {
		select {
		case entry := <-h.queue:
			h.save(entry)
		case <-h.stop:
			for {
				select {
				case entry := <-h.queue:
					h.save(entry)
				default:
					return
				}
			}
		}
	}
}

func (h *history) save(entry pin.RequestLog) {
	if err := h.write(context.Background(), entry); err != nil {
		h.counters.Add("history.failed")
		h.log.Warn("request log write failed", "error", err)
	}
}

// record logs a request once its response finished, with the usage the
// meter read and its list-price cost. response and meter are nil when
// CLIProxy was unreachable; err is the send or stream error.
func (a *App) record(call upstreamCall, entry pin.RequestLog, response *http.Response, meter *usage.Meter, err error) {
	entry.Latency = time.Since(entry.At)
	entry.Queue = call.queued.waited()
	if response != nil {
		entry.Status = response.StatusCode
	}
	if meter != nil {
		entry.Tokens = meter.Tokens()
		if price, ok := a.cfg.Price(entry.Model); ok {
			entry.Cost, entry.Priced = price.Cost(entry.Tokens), true
		}
	}
	entry.Error = failure(call, response, err)
	if err != nil && call.r.Context().Err() != nil && a.cut.Load() {
		entry.Error = "cut at shutdown"
	}
	a.history.add(entry)
}

// failure summarizes why a request failed, never with request text.
func failure(call upstreamCall, response *http.Response, err error) string {
	var summary string
	switch {
	case err == nil:
		return ""
	case call.r.Context().Err() != nil:
		return "canceled by the client"
	case call.watch.Fired():
		return "CLIProxy sent nothing for " + call.watch.Timeout().String()
	case errors.As(err, new(busyError)):
		return err.Error()
	case response == nil:
		summary = "CLIProxy unreachable: "
	default:
		summary = "stream interrupted: "
	}
	// The URL of a failed call repeats what the request kind says.
	if failed, ok := errors.AsType[*url.Error](err); ok {
		err = failed.Err
	}
	summary += err.Error()
	return strings.ToValidUTF8(summary[:min(len(summary), maxFailure)], "")
}

// retain prunes the request log to the configured retention and, with a
// prompt retention, thins older decisions of their prompts, now and then
// hourly.
func (a *App) retain(ctx context.Context) {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()
	for {
		deleted, err := a.store.Prune(ctx, time.Now().Add(-a.cfg.LogRetention))
		if err != nil && ctx.Err() == nil {
			a.log.Warn("request log prune failed", "error", err)
		} else if deleted > 0 {
			a.log.Debug("request log pruned", "deleted", deleted)
		}
		if a.cfg.PromptRetention > 0 {
			thinned, err := a.store.PrunePrompts(ctx, time.Now().Add(-a.cfg.PromptRetention))
			if err != nil && ctx.Err() == nil {
				a.log.Warn("prompt prune failed", "error", err)
			} else if thinned > 0 {
				a.log.Debug("decision prompts pruned", "decisions", thinned)
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}
