package proxy

import (
	"context"
	"testing"
	"time"
)

func TestTheWatchdogEndsASilentUpstreamOnly(t *testing.T) {
	ctx, watch := Watch(context.Background(), 50*time.Millisecond)
	defer watch.Stop()
	watch.Reset()
	for range 5 {
		time.Sleep(20 * time.Millisecond)
		_, _ = watch.Write([]byte("chunk"))
	}
	if ctx.Err() != nil || watch.Fired() {
		t.Fatal("the watchdog fired on a talking upstream")
	}
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("the watchdog never fired on a silent upstream")
	}
	if !watch.Fired() {
		t.Fatal("Fired is false after the watchdog canceled the call")
	}
}

func TestTheWatchdogCountsOnlyWhileACallIsOut(t *testing.T) {
	ctx, watch := Watch(context.Background(), 20*time.Millisecond)
	defer watch.Stop()
	time.Sleep(50 * time.Millisecond)
	watch.Reset()
	watch.Pause()
	time.Sleep(50 * time.Millisecond)
	if ctx.Err() != nil || watch.Fired() {
		t.Fatal("the watchdog counted while no call was out")
	}
	var none *Watchdog
	if none.Fired() {
		t.Fatal("a nil watchdog fired")
	}
}

func TestAStoppedWatchdogIsNotFired(t *testing.T) {
	ctx, watch := Watch(context.Background(), time.Hour)
	watch.Reset()
	watch.Stop()
	if ctx.Err() == nil || watch.Fired() {
		t.Fatalf("stopped watchdog: context %v, fired %v", ctx.Err(), watch.Fired())
	}
}
