package pin

import (
	"context"
	"slices"
	"time"
)

// Summary aggregates pins and the decision log for /stats.
type Summary struct {
	Pins       map[string]int `json:"pins"`
	Decisions  map[string]int `json:"decisions"`
	Levels     map[string]int `json:"levels"`
	Deciders   map[string]int `json:"deciders"`
	Escalated  int            `json:"escalated"`
	Consulted  int            `json:"consulted"`
	Failovers  map[string]int `json:"failovers"`
	Upgrades   map[string]int `json:"capability_upgrades"`
	Migrations map[string]int `json:"migrations"`
	Overrides  int            `json:"overrides"`
	// Explored counts the decisions and redecisions routed one step cheaper
	// to learn whether that step would do, of the ExploreEligible ones
	// borderline enough to be.
	Explored        int              `json:"explored"`
	ExploreEligible int              `json:"explore_eligible"`
	LatencyMS       map[string]int64 `json:"decider_latency_ms"`

	// Redecisions counts the threads decided again, from the previous model
	// to the new one.
	Redecisions map[string]int `json:"redecisions"`
}

// Summarize reports every pin by model and the log since the given time.
// Decider latency covers the decisions some decider ran for.
func (s *Store) Summarize(ctx context.Context, since time.Time) (Summary, error) {
	summary := Summary{
		Pins: map[string]int{}, Decisions: map[string]int{}, Levels: map[string]int{}, Deciders: map[string]int{},
		Failovers: map[string]int{}, Upgrades: map[string]int{}, Migrations: map[string]int{},
		Redecisions: map[string]int{}, LatencyMS: map[string]int64{},
	}
	rows, err := s.read.QueryContext(ctx, `SELECT model, COUNT(*) FROM pins GROUP BY model`)
	if err != nil {
		return summary, err
	}
	for rows.Next() {
		var model string
		var count int
		if err := rows.Scan(&model, &count); err != nil {
			rows.Close()
			return summary, err
		}
		summary.Pins[model] = count
	}
	rows.Close()

	rows, err = s.read.QueryContext(ctx,
		`SELECT kind, model, previous, decider, escalated, level,
			EXISTS (SELECT 1 FROM json_each(attempts) WHERE value->>'outcome' = 'doubted'),
			EXISTS (SELECT 1 FROM json_each(attempts) WHERE value->>'outcome' != 'skipped'), latency_ms,
			COALESCE(basis->>'explore', 0), COALESCE(basis->>'explore_eligible', 0)
		FROM events WHERE at >= ?`,
		since.UnixMilli())
	if err != nil {
		return summary, err
	}
	defer rows.Close()
	var latencies []int64
	for rows.Next() {
		var kind, model, previous, decider, level string
		var escalated, consulted, ran, explored, eligible bool
		var latency int64
		if err := rows.Scan(&kind, &model, &previous, &decider, &escalated, &level, &consulted, &ran,
			&latency, &explored, &eligible); err != nil {
			return summary, err
		}
		if explored {
			summary.Explored++
		}
		if eligible {
			summary.ExploreEligible++
		}
		switch kind {
		case Decision:
			summary.Decisions[model]++
			if level != "" {
				summary.Levels[level]++
			}
			summary.Deciders[decider]++
			if escalated {
				summary.Escalated++
			}
			if consulted {
				summary.Consulted++
			}
			// A decision no decider ran for took no decider time.
			if ran {
				latencies = append(latencies, latency)
			}
		case Failover:
			summary.Failovers[previous+"->"+model]++
		case CapabilityUpgrade:
			summary.Upgrades[previous+"->"+model]++
		case Migration:
			summary.Migrations[previous+"->"+model]++
		case Redecision:
			summary.Redecisions[previous+"->"+model]++
		case Override:
			summary.Overrides++
		}
	}
	if len(latencies) > 0 {
		slices.Sort(latencies)
		summary.LatencyMS["p50"] = latencies[len(latencies)/2]
		summary.LatencyMS["p95"] = latencies[(len(latencies)*95)/100]
		summary.LatencyMS["max"] = latencies[len(latencies)-1]
	}
	return summary, rows.Err()
}
