package pin

import (
	"context"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func TestQueueWaitIsLoggedAndSummarized(t *testing.T) {
	ctx := context.Background()
	store := open(t, filepath.Join(t.TempDir(), "router.db"))
	at(store, base)
	request := func(model string, queue time.Duration) RequestLog {
		return RequestLog{At: base, Kind: "messages", Requested: "auto", Model: model, Reason: "pinned", Status: 200,
			Latency: 10 * time.Second, Queue: queue}
	}
	logRequests(t, store, request("glm-5.3", 0), request("glm-5.3", 2*time.Second), request("glm-5.3", 4*time.Second),
		request("gpt-6.1-sol", 0), RequestLog{At: base.Add(-48 * time.Hour), Kind: "messages", Model: "glm-5.3", Queue: time.Minute})
	requests, err := store.Requests(ctx, RequestFilter{Model: "glm-5.3"})
	if err != nil || len(requests) != 4 {
		t.Fatalf("Requests = %v, %v", requests, err)
	}
	var queues []time.Duration
	for _, r := range requests {
		queues = append(queues, r.Queue)
	}
	slices.Sort(queues)
	if want := []time.Duration{0, 2 * time.Second, 4 * time.Second, time.Minute}; !slices.Equal(queues, want) {
		t.Errorf("queues read back as %v, want %v", queues, want)
	}
	waits, err := store.QueueWaits(ctx, base.Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := waits["glm-5.3"], (QueueWait{Requests: 3, Queued: 2, Mean: 3 * time.Second, P95: 4 * time.Second}); got != want {
		t.Errorf("glm-5.3 waits = %+v, want %+v", got, want)
	}
	if got := waits["gpt-6.1-sol"]; got != (QueueWait{Requests: 1}) {
		t.Errorf("gpt-6.1-sol waits = %+v", got)
	}
}
