package route

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/thread"
	"github.com/MarioGK/AI-Router/internal/wire"
)

type controlledDecider struct {
	started chan context.Context
	release chan struct{}
	calls   atomic.Int32
}

func (d *controlledDecider) Name() string { return "controlled" }

func (d *controlledDecider) Info() decide.DeciderInfo {
	return decide.DeciderInfo{Name: d.Name(), Kind: decide.KindSystemOne, Model: "test"}
}

func (d *controlledDecider) Decide(ctx context.Context, _ string, _ []decide.Candidate, _ []decide.Level) (decide.Verdict, error) {
	d.calls.Add(1)
	d.started <- ctx
	select {
	case <-ctx.Done():
		return decide.Verdict{}, context.Cause(ctx)
	case <-d.release:
		return decide.Verdict{Choice: "chosen", Probabilities: map[string]float64{"default": 0, "chosen": 1}}, nil
	}
}

func cancellationResolver(t *testing.T, timeout time.Duration) (*Resolver, *controlledDecider, *pin.Store, func()) {
	t.Helper()
	store, err := pin.Open(filepath.Join(t.TempDir(), "router.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	d := &controlledDecider{started: make(chan context.Context, 2), release: make(chan struct{})}
	auto := config.Auto{
		DefaultModel: "default", Candidates: []config.Candidate{{ID: "default"}, {ID: "chosen"}},
		EscalationThreshold: 0.9, StateMaxChars: 1000, DecideTimeout: timeout, RecordPrompts: new(false),
	}
	chain := decide.NewChain([]decide.Step{{Decider: d, Timeout: timeout}}, auto.EscalationThreshold, decide.DoubtEscalate, nil)
	r := New(auto, "test", store, chain, served(), nil, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Cleanup(r.Close)
	release := sync.OnceFunc(func() { close(d.release) })
	t.Cleanup(release)
	return r, d, store, release
}

func awaitCancellationResult[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case result := <-ch:
		return result
	case <-time.After(2 * time.Second):
		t.Fatal("operation did not finish promptly")
		var zero T
		return zero
	}
}

func TestResolveCancellationKeepsTheSharedDecision(t *testing.T) {
	r, d, store, release := cancellationResolver(t, 30*time.Second)
	id := thread.Identity{Session: "shared-thread"}
	conversation := wire.Conversation{LatestUser: "choose a model", HasTools: true}
	ctx, cancel := context.WithCancelCause(t.Context())
	defer cancel(nil)
	first := make(chan Target, 1)
	go func() { first <- r.Resolve(ctx, wire.Messages, id, conversation) }()
	sharedCtx := awaitCancellationResult(t, d.started)
	if deadline, ok := sharedCtx.Deadline(); !ok || time.Until(deadline) > r.auto.DecideTimeout {
		t.Fatal("shared decision has no bounded deadline")
	}
	cancel(errors.New("request disconnected"))
	if target := awaitCancellationResult(t, first); target.Pinned() {
		t.Fatalf("canceled request received a pin: %+v", target)
	}
	if err := sharedCtx.Err(); err != nil {
		t.Fatalf("request cancellation stopped the shared decision: %v", err)
	}
	second := make(chan Target, 1)
	go func() { second <- r.Resolve(t.Context(), wire.Messages, id, conversation) }()
	release()
	target := awaitCancellationResult(t, second)
	if !target.Pinned() || target.Model != "chosen" || d.calls.Load() != 1 {
		t.Fatalf("remaining request = %+v; decision calls = %d", target, d.calls.Load())
	}
	detail, err := store.Thread(t.Context(), target.Key)
	if err != nil || detail == nil || len(detail.Events) != 1 || detail.Events[0].Kind != pin.Decision {
		t.Fatalf("thread decision trail = %+v, %v", detail, err)
	}
}

func TestResolverCloseCancelsSharedWork(t *testing.T) {
	r, d, store, _ := cancellationResolver(t, 30*time.Second)
	id := thread.Identity{Session: "closing-thread"}
	request := make(chan Target, 1)
	go func() {
		request <- r.Resolve(t.Context(), wire.Messages, id, wire.Conversation{LatestUser: "choose a model", HasTools: true})
	}()
	sharedCtx := awaitCancellationResult(t, d.started)
	closed := make(chan struct{})
	go func() {
		r.Close()
		close(closed)
	}()
	awaitCancellationResult(t, closed)
	if !errors.Is(context.Cause(sharedCtx), context.Canceled) {
		t.Fatalf("shared decision after Close = %v", context.Cause(sharedCtx))
	}
	if target := awaitCancellationResult(t, request); target.Pinned() {
		t.Fatalf("shutdown created a pin: %+v", target)
	}
	if p, err := store.Get(t.Context(), r.key(id.Session, "")); err != nil || p != nil {
		t.Fatalf("pin after Close = %+v, %v", p, err)
	}
	if target := r.Resolve(t.Context(), wire.Messages, thread.Identity{Session: "after-close"},
		wire.Conversation{LatestUser: "choose a model", HasTools: true}); target.Pinned() || d.calls.Load() != 1 {
		t.Fatalf("request after Close = %+v; decision calls = %d", target, d.calls.Load())
	}
}

func TestResolverChooserTimeoutPinsTheDefaultOnce(t *testing.T) {
	r, d, store, _ := cancellationResolver(t, 25*time.Millisecond)
	id := thread.Identity{Session: "timed-out-thread"}
	conversation := wire.Conversation{LatestUser: "choose a model", HasTools: true}
	request := make(chan Target, 1)
	go func() { request <- r.Resolve(t.Context(), wire.Messages, id, conversation) }()
	awaitCancellationResult(t, d.started)
	target := awaitCancellationResult(t, request)
	if !target.Pinned() || target.Model != "default" {
		t.Fatalf("request after chooser timeout = %+v", target)
	}
	second := r.Resolve(t.Context(), wire.Messages, id, conversation)
	if second.Key != target.Key || second.Model != "default" || d.calls.Load() != 1 {
		t.Fatalf("second request = %+v; decision calls = %d", second, d.calls.Load())
	}
	detail, err := store.Thread(t.Context(), target.Key)
	if err != nil || detail == nil || len(detail.Events) != 1 || detail.Events[0].Decider != config.DefaultDecider {
		t.Fatalf("default decision trail = %+v, %v", detail, err)
	}
}
