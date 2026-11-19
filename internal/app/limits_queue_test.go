package app

import (
	"net/http"
	"testing"
	"time"
)

func TestTheQueueWaitIsLoggedApartFromLatency(t *testing.T) {
	h := newHarness(t, "")
	h.yaml += glmLimits(1)
	h.restart()
	readAll(t, h.claude("ses_queue", "add a flag"))
	// The thread is pinned now; hold the only slot so its next turn queues.
	release, err := h.app.limits.Gate("glm-5.3").Acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan int, 1)
	go func() {
		response := h.claude("ses_queue", "continue")
		readAll(t, response)
		done <- response.StatusCode
	}()
	const wait = 150 * time.Millisecond
	time.Sleep(wait)
	release()
	if status := <-done; status != http.StatusOK {
		t.Fatalf("status = %d", status)
	}
	requests := h.logged()
	if len(requests) != 2 {
		t.Fatalf("logged %d requests, want 2", len(requests))
	}
	first, second := requests[0], requests[1]
	if first.Queue > 50*time.Millisecond {
		t.Errorf("an uncontended request queued for %v", first.Queue)
	}
	if second.Queue < wait*8/10 {
		t.Errorf("the queued request waited %v, want about %v", second.Queue, wait)
	}
	if second.Latency < second.Queue {
		t.Errorf("latency %v does not include the queue wait %v", second.Latency, second.Queue)
	}
}
