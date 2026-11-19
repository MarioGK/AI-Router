package decide

import (
	"context"
	"errors"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestCanceledChainDoesNotAskOrChangeHealth(t *testing.T) {
	decider := &fakeDecider{name: "judge", verdict: Verdict{Choice: three[0].ID}}
	chain := NewChain([]Step{{Decider: decider, Timeout: time.Second}}, 0.1, DoubtEscalate, nil).Rest(1, time.Minute)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	decision, ok := chain.Decide(ctx, "state", three)
	if ok || decider.calls != 0 || len(decision.Attempts) != 0 {
		t.Fatalf("canceled decision = %+v, %v after %d calls", decision, ok, decider.calls)
	}
	if health := chain.Health()[0]; health.Failures != 0 || health.Probing || !health.Resting.IsZero() {
		t.Fatalf("canceled request changed health: %+v", health)
	}
}

func TestExpiredStepRejectsALateSuccessfulVerdict(t *testing.T) {
	decider := &fakeDecider{name: "judge"}
	_, attempt := try(t.Context(), Step{Decider: decider, Timeout: time.Millisecond}, three, nil,
		func(ctx context.Context) (Verdict, error) {
			<-ctx.Done()
			return Verdict{Choice: three[0].ID, Model: "judge-model", Response: "late reply", UsageKnown: true}, nil
		})
	if attempt.Outcome != Failed || attempt.Error != context.DeadlineExceeded.Error() || attempt.Model != "judge-model" ||
		attempt.Response != "late reply" || !attempt.UsageKnown {
		t.Fatalf("late verdict attempt = %+v", attempt)
	}
}

func TestFailedAttemptsKeepAvailableVerdictMetadata(t *testing.T) {
	for _, test := range []struct {
		name string
		err  error
	}{
		{name: "decider error", err: errors.New("invalid answer")},
		{name: "unusable probabilities"},
	} {
		t.Run(test.name, func(t *testing.T) {
			verdict := Verdict{
				Choice: "unknown", Model: "judge-model", Confidence: 0.7, Reason: "partial explanation",
				Response: "raw answer", Digest: "request-digest", InputTokens: 10, OutputTokens: 4, UsageKnown: true,
			}
			_, attempt := try(t.Context(), Step{Decider: &fakeDecider{name: "judge"}, Timeout: time.Second}, three, nil,
				func(context.Context) (Verdict, error) { return verdict, test.err })
			if attempt.Outcome != Failed || attempt.Model != verdict.Model || attempt.Confidence != verdict.Confidence ||
				attempt.Reason != verdict.Reason || attempt.Response != verdict.Response || attempt.Digest != verdict.Digest ||
				attempt.InputTokens != verdict.InputTokens || attempt.OutputTokens != verdict.OutputTokens || !attempt.UsageKnown {
				t.Fatalf("failed attempt lost metadata: %+v", attempt)
			}
		})
	}
}

func TestPairCutoffToleratesARoundedTie(t *testing.T) {
	decider := &fakePairer{
		name: "judge",
		verdict: Verdict{Choice: three[0].ID, Probabilities: map[string]float64{
			three[0].ID: 0.8, three[1].ID: 0.2,
		}},
		pair: Verdict{Probabilities: map[string]float64{
			three[0].ID: math.Nextafter(0.3, 1), three[1].ID: 0.3,
		}},
	}
	decision, ok := NewChain([]Step{{Decider: decider, Timeout: time.Second}}, 0.1, DoubtPair, nil).
		Decide(t.Context(), "state", three)
	if !ok || decision.Model != three[1].ID || !decision.Escalated || len(decision.Attempts) != 2 {
		t.Fatalf("rounded pair tie = %+v, %v", decision, ok)
	}
}

func TestAnOlderTurnCannotReleaseANewerProbe(t *testing.T) {
	decider := &probeOwnershipDecider{
		oldStarted: make(chan struct{}), oldRelease: make(chan struct{}),
		probeStarted: make(chan struct{}), probeRelease: make(chan struct{}),
	}
	chain := NewChain([]Step{{Decider: decider, Timeout: time.Minute}}, 0.1, DoubtEscalate, nil).Rest(1, time.Second)
	var seconds atomic.Int64
	chain.now = func() time.Time { return time.Unix(seconds.Load(), 0) }
	oldDone := make(chan struct{})
	var turns sync.WaitGroup
	turns.Go(func() {
		chain.Decide(t.Context(), "old", three)
		close(oldDone)
	})
	awaitSignal(t, decider.oldStarted)
	chain.Decide(t.Context(), "trip", three)
	seconds.Store(1)
	turns.Go(func() { chain.Decide(t.Context(), "probe", three) })
	awaitSignal(t, decider.probeStarted)
	close(decider.oldRelease)
	awaitSignal(t, oldDone)
	health := chain.Health()[0]
	seconds.Store(2)
	decision, ok := chain.Decide(t.Context(), "another", three)
	close(decider.probeRelease)
	turns.Wait()
	if ok || decider.calls.Load() != 3 || !health.Probing || len(decision.Attempts) != 1 || decision.Attempts[0].Outcome != Skipped {
		t.Fatalf("probe ownership lost: health %+v, calls %d, decision %+v, %v", health, decider.calls.Load(), decision, ok)
	}
}

// probeOwnershipDecider keeps a turn from before the trip alive during the probe.
type probeOwnershipDecider struct {
	calls        atomic.Int32
	oldStarted   chan struct{}
	oldRelease   chan struct{}
	probeStarted chan struct{}
	probeRelease chan struct{}
}

func (*probeOwnershipDecider) Name() string { return "judge" }

func (*probeOwnershipDecider) Info() DeciderInfo { return DeciderInfo{Name: "judge", Kind: "fake"} }

func (d *probeOwnershipDecider) Decide(ctx context.Context, _ string, _ []Candidate, _ []Level) (Verdict, error) {
	switch d.calls.Add(1) {
	case 1:
		close(d.oldStarted)
		select {
		case <-d.oldRelease:
		case <-ctx.Done():
		}
	case 3:
		close(d.probeStarted)
		select {
		case <-d.probeRelease:
		case <-ctx.Done():
		}
	}
	return Verdict{}, errors.New("decider unavailable")
}

func awaitSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for a decider turn")
	}
}
