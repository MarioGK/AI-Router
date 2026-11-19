package pin

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
)

func TestFlightSharesOneDecision(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var flight Flight
		defer flight.Close()
		var calls atomic.Int32
		release := make(chan struct{})
		var wg sync.WaitGroup
		results := make([]*Pin, 50)
		for i := range results {
			wg.Go(func() {
				var err error
				results[i], err = flight.DoContext(t.Context(), "k", func() (*Pin, error) {
					calls.Add(1)
					<-release
					return &Pin{Key: "k", Model: "glm-5.3"}, nil
				})
				if err != nil {
					t.Errorf("DoContext = %v", err)
				}
			})
		}
		synctest.Wait()
		close(release)
		wg.Wait()
		if calls.Load() != 1 {
			t.Fatalf("decide ran %d times", calls.Load())
		}
		for _, p := range results {
			if p == nil || p.Model != "glm-5.3" {
				t.Fatalf("result = %+v", p)
			}
		}
	})
}

func TestFlightCancellationLeavesTheSharedDecisionRunning(t *testing.T) {
	for _, canceled := range []string{"first", "second"} {
		t.Run(canceled, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var flight Flight
				defer flight.Close()
				var calls atomic.Int32
				release := make(chan struct{})
				finish := sync.OnceFunc(func() { close(release) })
				defer finish()
				shared := &Pin{Key: "k", Model: "glm-5.3"}
				decisionErr := errors.New("shared decision error")
				decide := func() (*Pin, error) {
					calls.Add(1)
					<-release
					return shared, decisionErr
				}
				type result struct {
					pin *Pin
					err error
				}
				firstCtx, firstCancel := context.WithCancelCause(t.Context())
				defer firstCancel(nil)
				secondCtx, secondCancel := context.WithCancelCause(t.Context())
				defer secondCancel(nil)
				first, second := make(chan result, 1), make(chan result, 1)
				go func() {
					p, err := flight.DoContext(firstCtx, "k", decide)
					first <- result{p, err}
				}()
				synctest.Wait()
				go func() {
					p, err := flight.DoContext(secondCtx, "k", decide)
					second <- result{p, err}
				}()
				synctest.Wait()
				cancel, departed, remaining := firstCancel, first, second
				if canceled == "second" {
					cancel, departed, remaining = secondCancel, second, first
				}
				cause := errors.New("request disconnected")
				cancel(cause)
				synctest.Wait()
				select {
				case got := <-departed:
					if got.pin != nil || !errors.Is(got.err, cause) {
						t.Fatalf("canceled waiter = %+v, %v", got.pin, got.err)
					}
				default:
					t.Fatal("canceled waiter is still waiting for the decision")
				}
				select {
				case got := <-remaining:
					t.Fatalf("other waiter returned before the decision: %+v, %v", got.pin, got.err)
				default:
				}
				finish()
				got := <-remaining
				if got.pin != shared || !errors.Is(got.err, decisionErr) || calls.Load() != 1 {
					t.Fatalf("remaining waiter = %+v, %v; decide calls = %d", got.pin, got.err, calls.Load())
				}
			})
		})
	}
}

func TestFlightDoesNotStartWorkForACanceledCaller(t *testing.T) {
	var flight Flight
	defer flight.Close()
	ctx, cancel := context.WithCancelCause(t.Context())
	cause := errors.New("caller canceled")
	cancel(cause)
	p, err := flight.DoContext(ctx, "k", func() (*Pin, error) {
		t.Error("decide ran for a canceled caller")
		return &Pin{}, nil
	})
	if p != nil || !errors.Is(err, cause) {
		t.Fatalf("DoContext = %+v, %v", p, err)
	}
}

func TestFlightCloseWaitsForSharedWorkAndRejectsNewCalls(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var flight Flight
		release := make(chan struct{})
		finish := sync.OnceFunc(func() { close(release) })
		defer finish()
		ctx, cancel := context.WithCancel(t.Context())
		waiter := make(chan error, 1)
		go func() {
			_, err := flight.DoContext(ctx, "k", func() (*Pin, error) {
				<-release
				return &Pin{}, nil
			})
			waiter <- err
		}()
		synctest.Wait()
		cancel()
		if err := <-waiter; !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled waiter = %v", err)
		}
		closed := make(chan struct{}, 2)
		for range 2 {
			go func() {
				flight.Close()
				closed <- struct{}{}
			}()
		}
		synctest.Wait()
		select {
		case <-closed:
			t.Fatal("Close returned while a decision was still running")
		default:
		}
		for _, key := range []string{"k", "other"} {
			p, err := flight.DoContext(t.Context(), key, func() (*Pin, error) {
				t.Error("decide ran after Close")
				return &Pin{}, nil
			})
			if p != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("DoContext after Close = %+v, %v", p, err)
			}
		}
		finish()
		for range 2 {
			<-closed
		}
		flight.Close()
	})
}
