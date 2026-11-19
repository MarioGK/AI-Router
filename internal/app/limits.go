package app

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"slices"
	"sort"
	"time"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/limit"
	"github.com/MarioGK/AI-Router/internal/wire"
)

// busyError is a request that waited its whole queue timeout for a slot.
type busyError struct{ model string }

func (e busyError) Error() string {
	return e.model + " is busy: no slot came free within the queue timeout"
}

// retryPause is the pause before a throttled request's first retry; it
// doubles with every retry up to maxBackoff.
var retryPause = time.Second

const maxBackoff = 10 * time.Second

// limiter builds the concurrency gates, in model order.
func limiter(limits map[string]config.Limit) *limit.Limiter {
	models := make([]string, 0, len(limits))
	for model := range limits {
		models = append(models, model)
	}
	sort.Strings(models)
	gates := make([]*limit.Gate, len(models))
	for i, model := range models {
		l := limits[model]
		gates[i] = limit.NewGate(model, l.Concurrency, l.Spill, l.Recovery)
	}
	return limit.New(gates...)
}

// send forwards body for model. A capped model's request first waits for a
// slot, which the returned release frees once the response is done; a
// throttling reply shrinks the cap and is retried from the queue. Every
// effort of a model shares its gate; counting tokens generates nothing and
// never waits for a slot.
func (a *App) send(call upstreamCall, model string, body []byte) (*http.Response, func(), error) {
	model = config.Bare(model)
	gate := a.limits.Gate(model)
	if gate == nil || call.kind == wire.CountTokens {
		call.watch.Reset()
		response, err := a.forward.Send(call.ctx, call.r, body)
		return response, func() {}, err
	}
	settings := a.cfg.Limits[model]
	for attempt := 0; ; attempt++ {
		// Waiting for a slot is the router's own delay, not CLIProxy's.
		call.watch.Pause()
		queued, cancel := context.WithTimeout(call.r.Context(), settings.QueueTimeout)
		waiting := time.Now()
		release, err := gate.Acquire(queued)
		cancel()
		call.queued.add(time.Since(waiting))
		if err != nil {
			if call.r.Context().Err() == nil {
				a.counters.Add("busy." + model)
				err = busyError{model}
			}
			return nil, nil, err
		}
		call.watch.Reset()
		response, err := a.forward.Send(call.ctx, call.r, body)
		if err != nil {
			release()
			return nil, nil, err
		}
		if response.StatusCode != http.StatusTooManyRequests || attempt >= *settings.Retries {
			return response, release, nil
		}
		head, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		if !throttling(head, settings.ThrottleCodes) {
			response.Body = readCloser{io.MultiReader(bytes.NewReader(head), response.Body), response.Body}
			return response, release, nil
		}
		response.Body.Close()
		gate.Throttled()
		release()
		call.watch.Pause()
		a.counters.Add("throttled." + model)
		a.log.Info("upstream throttled; retrying from the queue", "model", model, "attempt", attempt+1,
			"limit", gate.Stats().Limit)
		select {
		case <-call.r.Context().Done():
			return nil, nil, call.r.Context().Err()
		case <-time.After(min(retryPause<<min(attempt, 8), maxBackoff)):
		}
	}
}

// throttling reports whether a 429 body carries one of the throttle codes,
// as a JSON code ("1302") or in a message ("[1302]").
func throttling(body []byte, codes []string) bool {
	return slices.ContainsFunc(codes, func(code string) bool {
		return bytes.Contains(body, []byte(`"`+code+`"`)) || bytes.Contains(body, []byte("["+code+"]"))
	})
}

// readCloser reads from one reader and closes another.
type readCloser struct {
	io.Reader
	io.Closer
}
