package catalog

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestAFailedRefreshStaysHealthyUntilTheListGoesStale(t *testing.T) {
	var down atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if down.Load() {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"glm-5.3"},{"id":"gpt-6.1-sol"}]}`))
	}))
	defer server.Close()
	c := New(server.URL, "k", server.Client())
	if status := c.Check([]string{"glm-5.3"}); status.Loaded || status.Healthy() || !c.Has("anything") {
		t.Fatalf("before the first refresh: %+v", status)
	}
	if err := c.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if status := c.Check([]string{"glm-5.3", "claude-opus-5-5"}); status.Healthy() || len(status.Missing) != 1 || c.Has("claude-opus-5-5") {
		t.Fatalf("a missing model = %+v", status)
	}
	down.Store(true)
	if err := c.Refresh(context.Background()); err == nil {
		t.Fatal("a failed refresh reported no error")
	}
	if status := c.Check([]string{"glm-5.3"}); !status.Healthy() || status.Error == "" || !c.Has("gpt-6.1-sol") {
		t.Fatalf("one failed refresh = %+v", status)
	}
	c.fetchedAt = c.fetchedAt.Add(-stale - time.Minute)
	if status := c.Check([]string{"glm-5.3"}); status.Healthy() || !status.Stale {
		t.Fatalf("a stale list = %+v", status)
	}
	down.Store(false)
	if err := c.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if status := c.Check([]string{"glm-5.3"}); !status.Healthy() || status.Stale || status.Error != "" {
		t.Fatalf("after recovering = %+v", status)
	}
}
