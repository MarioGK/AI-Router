package dashboard

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
)

// heldClassifier holds every trial until it is released.
type heldClassifier struct {
	*fakeClassifier
	started, release chan struct{}
}

func (c heldClassifier) Try(ctx context.Context, prompt string, images bool) (pin.Event, decide.Policy) {
	c.started <- struct{}{}
	<-c.release
	return c.fakeClassifier.Try(ctx, prompt, images)
}

func tryPrompt(d *Dashboard, cookie *http.Cookie, body string) *httptest.ResponseRecorder {
	return serve(d, http.MethodPost, "/dashboard/sse/try", body, cookie, "Datastar-Request", "true", "Content-Type", "application/json")
}

func TestTrialsAreCappedAndBounded(t *testing.T) {
	d := newDashboard(t, history())
	cookie := signIn(t, d)
	held := heldClassifier{&fakeClassifier{}, make(chan struct{}), make(chan struct{})}
	d.classifier = held

	huge := `{"prompt":"` + strings.Repeat("x", maxTrialBody) + `"}`
	if response := tryPrompt(d, cookie, huge); response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("a trial over %d bytes = %d, want 413", maxTrialBody, response.Code)
	}

	var running sync.WaitGroup
	codes := make(chan int, maxTrials)
	for range maxTrials {
		running.Go(func() { codes <- tryPrompt(d, cookie, `{"prompt":"Fix the flaky retry test"}`).Code })
		<-held.started
	}
	if response := tryPrompt(d, cookie, `{"prompt":"one more"}`); response.Code != http.StatusTooManyRequests {
		t.Errorf("trial %d while %d run = %d, want 429", maxTrials+1, maxTrials, response.Code)
	}
	close(held.release)
	running.Wait()
	close(codes)
	for code := range codes {
		if code != http.StatusOK {
			t.Errorf("a running trial = %d, want 200", code)
		}
	}
	go func() { <-held.started }()
	if response := tryPrompt(d, cookie, `{"prompt":"after the others"}`); response.Code != http.StatusOK {
		t.Errorf("a trial after the others finished = %d, want 200", response.Code)
	}
	if tried := len(held.tried); tried != maxTrials+1 {
		t.Errorf("the classifier tried %d prompts, want %d", tried, maxTrials+1)
	}
}
