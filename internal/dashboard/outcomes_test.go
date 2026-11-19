package dashboard

import (
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/learn"
)

func TestTheDecisionsTabQueuesLikelyMisroutesForReview(t *testing.T) {
	d := newDashboard(t, history())
	cookie := signIn(t, d)
	body := page(t, d, cookie, "/dashboard/decisions", http.StatusOK)
	contains(t, "decisions page", body, `id="misrouted"`, "Likely misrouted",
		"1 unreviewed decision with a sign of a misroute among 12 settled threads",
		"An operator moved it up, glm-5.3 → claude-opus-5-5.", "40 turns · 9m0s active · $0.12",
		`href="/dashboard/threads/thread-misrouted#review"`,
		"15 min · 10% up", "how often its threads moved up to a stronger model")
	if strings.Contains(body, "no account had quota left") {
		t.Error("the queue names a signal that says nothing about the decision")
	}
}

func TestTheDecisionsTabShowsWhatTheHistoryTaught(t *testing.T) {
	d := newDashboard(t, history())
	cookie := signIn(t, d)
	body := page(t, d, cookie, "/dashboard/decisions", http.StatusOK)
	contains(t, "decisions page without learning", body, `id="learning"`, "Learning from history is off")
	d.Learning(func() learn.Summary {
		return learn.Summary{Enabled: true, Labels: 42, Kinds: map[string]int{learn.Reviewed: 2, learn.Sufficed: 40},
			Fit: 0.35, LevelFit: 0.1, Replays: 42, MinLabels: 20, Refreshed: time.Now(), TookMS: 1200}
	})
	body = page(t, d, cookie, "/dashboard/decisions", http.StatusOK)
	contains(t, "decisions page", body, "42 labelled threads", "2 reviewed, 40 sufficed", "0.35", "0.10",
		"on the reasoning", "1,200 ms")
}

func TestAReviewTakesTheDecisionOffTheQueue(t *testing.T) {
	source := history()
	d := newDashboard(t, source)
	var reviewed []int64
	d.outcomes = fakeOutcomes{reviewed: &reviewed}
	cookie := signIn(t, d)
	key := source.detail.Key
	serve(d, http.MethodPost, "/dashboard/sse/threads/"+key+"/review", `{"reviewModel":"gpt-6.1-sol"}`, cookie,
		"Datastar-Request", "true", "Content-Type", "application/json")
	if !slices.Equal(reviewed, []int64{source.detail.Decision.ID}) {
		t.Fatalf("reviewed = %v", reviewed)
	}
}

func TestADecisionShowsWhatLearnedTimeAndMissRatesDidToEachStep(t *testing.T) {
	source := history()
	source.detail.Decision.Basis.Costs = &decide.Costs{
		Task:     map[string]float64{"gpt-6.1-sol": 0.25, "claude-opus-5-5": 2},
		Done:     map[string]float64{"gpt-6.1-sol": 0.9, "claude-opus-5-5": 2.1},
		Hours:    map[string]float64{"gpt-6.1-sol": 0.5, "claude-opus-5-5": 0.2},
		Time:     map[string]float64{"gpt-6.1-sol": 0.5, "claude-opus-5-5": 0.1},
		MissRate: map[string]float64{"gpt-6.1-sol": 0.3},
		Miss:     map[string]float64{"gpt-6.1-sol": 6},
	}
	d := newDashboard(t, source)
	cookie := signIn(t, d)
	body := page(t, d, cookie, "/dashboard/threads/"+source.detail.Key, http.StatusOK)
	contains(t, "thread page", body, "task $0.25 + 30 min of time ($0.50); 30% of its threads moved up, so a miss costs 6× on top",
		"task $2.00 + 12 min of time ($0.10)")
}
