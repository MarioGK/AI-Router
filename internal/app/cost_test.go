package app

import (
	"net/http"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/pin"
)

// priced turns the cost economy on over the harness's candidates.
const priced = `  cost: {enabled: true, miss: 1}
prices:
  deepseek-flash: {input: 0.30, output: 1.20, cache_read: 0.006}
  glm-5.3: {input: 1.40, output: 4.40, cache_read: 0.26}
  gpt-6.1-sol: {input: 2.00, output: 10.00, cache_read: 0.20}`

func TestTheCheapestWayToGetTheTaskDoneWins(t *testing.T) {
	// 15% on GLM would pass the 10% threshold, but a typical task costs
	// eight times as much on GLM: trying DeepSeek first is cheaper even
	// counting the risk of a miss.
	h := newHarness(t, priced)
	h.jev.answer = "deepseek-flash=0.85,glm-5.3=0.15"
	if body := readAll(t, h.claude("cc-cheap", "rename the variable")); !strings.Contains(body, `"deepseek-flash"`) {
		t.Fatalf("the cheaper step lost: %s", body)
	}
	// Half on GLM makes a miss on DeepSeek dearer than GLM itself.
	h.jev.answer = "deepseek-flash=0.5,glm-5.3=0.5"
	if body := readAll(t, h.claude("cc-doubt", "add pagination")); !strings.Contains(body, `"glm-5.3"`) {
		t.Fatalf("the likely miss stayed: %s", body)
	}
	page, err := h.app.store.Decisions(t.Context(), pin.DecisionFilter{})
	if err != nil || len(page.Decisions) != 2 {
		t.Fatalf("decisions = %+v, %v", page, err)
	}
	moved, stayed := page.Decisions[0], page.Decisions[1]
	if costs := stayed.Basis.Costs; costs == nil || stayed.Escalated || costs.Task["glm-5.3"] <= costs.Task["deepseek-flash"] ||
		costs.Done["deepseek-flash"] >= costs.Done["glm-5.3"] {
		t.Fatalf("the cheap decision's costs = %+v, escalated %v", stayed.Basis.Costs, stayed.Escalated)
	}
	if !moved.Escalated || moved.Model != "glm-5.3" || moved.Basis.Costs == nil {
		t.Fatalf("the moved decision = %+v", moved)
	}
	stats := h.do(http.MethodGet, "/stats", "")
	body := readAll(t, stats)
	if !strings.Contains(body, `"costs"`) || !strings.Contains(body, `"Model": "deepseek-flash"`) {
		t.Fatalf("stats lack the task costs: %s", body)
	}
	if !strings.Contains(body, `"outcomes"`) || !strings.Contains(body, `"repin_up": 0`) || !strings.Contains(body, `"miss_rate"`) {
		t.Fatalf("stats lack the thread outcomes: %s", body)
	}
	// Without time_value and settled threads, nothing learned enters a score.
	if costs := stayed.Basis.Costs; costs.Time != nil || costs.Miss != nil {
		t.Fatalf("an economy that learned nothing changed a score: %+v", costs)
	}
}
