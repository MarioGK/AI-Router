package bench

import (
	"encoding/json/v2"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/MarioGK/AI-Router/internal/decide"
)

// GroupMetrics includes support beside every score. Reasoning and joint
// scores use all labelled trials, including errors and missing answers.
type GroupMetrics struct {
	Name                string  `json:"name"`
	Planned             int     `json:"planned"`
	Correct             int     `json:"correct"`
	Accuracy            float64 `json:"accuracy"`
	ChoiceCorrect       int     `json:"choice_correct"`
	ChoiceAccuracy      float64 `json:"choice_accuracy"`
	Labelled            int     `json:"labelled"`
	LevelCorrect        int     `json:"level_correct"`
	LevelAccuracy       float64 `json:"level_accuracy"`
	JointCorrect        int     `json:"joint_correct"`
	JointAccuracy       float64 `json:"joint_accuracy"`
	ChoiceJointCorrect  int     `json:"choice_joint_correct"`
	ChoiceJointAccuracy float64 `json:"choice_joint_accuracy"`
}

// DeciderMetrics describes trials, rather than pretending repeated trials
// are new independent tasks. Token totals cover reported counters, with
// partial usage separated; unknown usage is never a known free call.
type DeciderMetrics struct {
	Decider                 string  `json:"decider"`
	Planned                 int     `json:"planned"`
	UniqueCases             int     `json:"unique_cases"`
	Started                 int     `json:"started"`
	NotStarted              int     `json:"not_started"`
	Cancelled               int     `json:"cancelled"`
	Completed               bool    `json:"completed"`
	Errors                  int     `json:"errors"`
	Correct                 int     `json:"correct"`
	Accuracy                float64 `json:"accuracy"`
	ChoiceCorrect           int     `json:"choice_correct"`
	ChoiceAccuracy          float64 `json:"choice_accuracy"`
	Labelled                int     `json:"labelled"`
	LevelCorrect            int     `json:"level_correct"`
	LevelAccuracy           float64 `json:"level_accuracy"`
	MissingDepth            int     `json:"missing_depth"`
	JointCorrect            int     `json:"joint_correct"`
	JointAccuracy           float64 `json:"joint_accuracy"`
	ChoiceJointCorrect      int     `json:"choice_joint_correct"`
	ChoiceJointAccuracy     float64 `json:"choice_joint_accuracy"`
	Escalated               int     `json:"escalated"`
	UnderRouted             int     `json:"under_routed"`
	OverRouted              int     `json:"over_routed"`
	OtherMisses             int     `json:"other_misses"`
	LatencySamples          int     `json:"latency_samples"`
	P50MS                   float64 `json:"p50_ms"`
	P95MS                   float64 `json:"p95_ms"`
	Calls                   int     `json:"calls"`
	FailedCalls             int     `json:"failed_calls"`
	SkippedAttempts         int     `json:"skipped_attempts"`
	CallsPerStarted         float64 `json:"calls_per_started"`
	KnownUsageCalls         int     `json:"known_usage_calls"`
	UnknownUsageCalls       int     `json:"unknown_usage_calls"`
	InputTokens             int64   `json:"reported_input_tokens"`
	OutputTokens            int64   `json:"reported_output_tokens"`
	KnownInputTokens        int64   `json:"complete_usage_input_tokens"`
	KnownOutputTokens       int64   `json:"complete_usage_output_tokens"`
	PartialInputTokens      int64   `json:"partial_usage_input_tokens"`
	PartialOutputTokens     int64   `json:"partial_usage_output_tokens"`
	PairCalls               int     `json:"pair_calls"`
	FailedPairs             int     `json:"failed_pairs"`
	PairRate                float64 `json:"pair_calls_per_started"`
	PairKnownUsageCalls     int     `json:"pair_known_usage_calls"`
	PairUnknownUsageCalls   int     `json:"pair_unknown_usage_calls"`
	PairInputTokens         int64   `json:"reported_pair_input_tokens"`
	PairOutputTokens        int64   `json:"reported_pair_output_tokens"`
	PairKnownInputTokens    int64   `json:"complete_pair_usage_input_tokens"`
	PairKnownOutputTokens   int64   `json:"complete_pair_usage_output_tokens"`
	PairPartialInputTokens  int64   `json:"partial_pair_usage_input_tokens"`
	PairPartialOutputTokens int64   `json:"partial_pair_usage_output_tokens"`
	// Conditional means are over calls with known usage. Amortized means
	// include reported tokens only, with usage coverage recorded above.
	MeanReportedTokensPerKnownCall        float64        `json:"mean_reported_tokens_per_known_call"`
	MeanReportedTokensPerKnownPair        float64        `json:"mean_reported_tokens_per_known_pair"`
	AmortizedReportedTokensPerStarted     float64        `json:"amortized_reported_tokens_per_started"`
	AmortizedReportedPairTokensPerStarted float64        `json:"amortized_reported_pair_tokens_per_started"`
	RepeatedCases                         int            `json:"repeated_cases"`
	StableModelCases                      int            `json:"stable_model_cases"`
	ModelStability                        float64        `json:"model_stability"`
	JointRepeatedCases                    int            `json:"joint_repeated_cases"`
	StableJointCases                      int            `json:"stable_joint_cases"`
	JointStability                        float64        `json:"joint_stability"`
	RobustCorrectCases                    int            `json:"robust_correct_cases"`
	UniqueCaseRobustAccuracy              float64        `json:"unique_case_robust_accuracy"`
	GroupMacroAccuracy                    float64        `json:"group_macro_accuracy"`
	WeakestGroupAccuracy                  float64        `json:"weakest_group_accuracy"`
	GroupMacroJointAccuracy               float64        `json:"group_macro_joint_accuracy"`
	WeakestGroupJointAccuracy             float64        `json:"weakest_group_joint_accuracy"`
	Groups                                []GroupMetrics `json:"groups"`
	// Tiers partitions by the acceptable model set, so a case allowing
	// two models is not counted twice or assigned an arbitrary label.
	Tiers []GroupMetrics `json:"tiers"`
}

// Summary computes metrics for one named chain. Model accuracy uses every
// planned trial. Latency uses started trials, including failed ones, and
// excludes trials cancelled while waiting for a concurrency slot.
func (r Report) Summary(deciderIndex int) DeciderMetrics {
	m := DeciderMetrics{Decider: r.Deciders[deciderIndex], Planned: len(r.Cases)}
	groups, tiers := map[string]*GroupMetrics{}, map[string]*GroupMetrics{}
	unique := map[string][]int{}
	var latencies []time.Duration
	for i, c := range r.Cases {
		outcome := r.Outcomes[i][deciderIndex]
		unique[c.Name] = append(unique[c.Name], i)
		if outcome.Started {
			m.Started++
			latencies = append(latencies, outcome.Latency)
		} else {
			m.NotStarted++
		}
		if outcome.Cancelled {
			m.Cancelled++
		}
		if outcome.Err != "" {
			m.Errors++
		}
		if r.Correct(i, deciderIndex) {
			m.Correct++
		} else if outcome.Err == "" {
			switch r.routingDirection(i, outcome.Model) {
			case -1:
				m.UnderRouted++
			case 1:
				m.OverRouted++
			default:
				m.OtherMisses++
			}
		}
		if r.ChoiceCorrect(i, deciderIndex) {
			m.ChoiceCorrect++
		}
		if r.Labelled(i) {
			m.Labelled++
			if outcome.Level == "" {
				m.MissingDepth++
			}
			if r.LevelCorrect(i, deciderIndex) {
				m.LevelCorrect++
				if r.ChoiceCorrect(i, deciderIndex) {
					m.ChoiceJointCorrect++
				}
				if r.Correct(i, deciderIndex) {
					m.JointCorrect++
				}
			}
		}
		if outcome.Escalated {
			m.Escalated++
		}
		for _, attempt := range outcome.Attempts {
			m.addAttempt(attempt)
		}
		addGroup(groups, caseGroup(c), r, i, deciderIndex)
		acceptable := slices.Clone(c.Expect)
		slices.Sort(acceptable)
		addGroup(tiers, strings.Join(acceptable, "|"), r, i, deciderIndex)
	}
	m.UniqueCases = len(unique)
	m.Completed = m.Planned > 0 && m.Started == m.Planned && m.Cancelled == 0
	m.Accuracy, m.ChoiceAccuracy = share(m.Correct, m.Planned), share(m.ChoiceCorrect, m.Planned)
	m.LevelAccuracy, m.JointAccuracy = share(m.LevelCorrect, m.Labelled), share(m.JointCorrect, m.Labelled)
	m.ChoiceJointAccuracy = share(m.ChoiceJointCorrect, m.Labelled)
	m.CallsPerStarted, m.PairRate = share(m.Calls, m.Started), share(m.PairCalls, m.Started)
	m.MeanReportedTokensPerKnownCall = tokenMean(m.KnownInputTokens+m.KnownOutputTokens, m.KnownUsageCalls)
	m.MeanReportedTokensPerKnownPair = tokenMean(m.PairKnownInputTokens+m.PairKnownOutputTokens, m.PairKnownUsageCalls)
	m.AmortizedReportedTokensPerStarted = tokenMean(m.InputTokens+m.OutputTokens, m.Started)
	m.AmortizedReportedPairTokensPerStarted = tokenMean(m.PairInputTokens+m.PairOutputTokens, m.Started)
	slices.Sort(latencies)
	m.LatencySamples = len(latencies)
	m.P50MS, m.P95MS = latencyPercentileMS(latencies, 50), latencyPercentileMS(latencies, 95)
	m.Groups, m.Tiers = finishGroups(groups), finishGroups(tiers)
	if len(m.Groups) > 0 {
		m.WeakestGroupAccuracy = 1
		m.WeakestGroupJointAccuracy = 1
		labelledGroups := 0
		for _, g := range m.Groups {
			m.GroupMacroAccuracy += g.Accuracy
			m.WeakestGroupAccuracy = min(m.WeakestGroupAccuracy, g.Accuracy)
			if g.Labelled > 0 {
				labelledGroups++
				m.GroupMacroJointAccuracy += g.JointAccuracy
				m.WeakestGroupJointAccuracy = min(m.WeakestGroupJointAccuracy, g.JointAccuracy)
			}
		}
		m.GroupMacroAccuracy /= float64(len(m.Groups))
		if labelledGroups > 0 {
			m.GroupMacroJointAccuracy /= float64(labelledGroups)
		} else {
			m.WeakestGroupJointAccuracy = 0
		}
	}
	for rows := range maps.Values(unique) {
		m.addStability(r, rows, deciderIndex)
	}
	m.ModelStability = share(m.StableModelCases, m.RepeatedCases)
	m.JointStability = share(m.StableJointCases, m.JointRepeatedCases)
	m.UniqueCaseRobustAccuracy = share(m.RobustCorrectCases, m.UniqueCases)
	return m
}

func (m *DeciderMetrics) addAttempt(a decide.Attempt) {
	if a.Outcome == decide.Skipped {
		m.SkippedAttempts++
		return
	}
	m.Calls++
	if a.Outcome == decide.Failed {
		m.FailedCalls++
	}
	pair := len(a.Pair) > 0
	input, output := max(a.InputTokens, 0), max(a.OutputTokens, 0)
	m.InputTokens += input
	m.OutputTokens += output
	if pair {
		m.PairCalls++
		m.PairInputTokens += input
		m.PairOutputTokens += output
		if a.Outcome == decide.Failed {
			m.FailedPairs++
		}
	}
	if !a.UsageKnown {
		m.UnknownUsageCalls++
		m.PartialInputTokens += input
		m.PartialOutputTokens += output
		if pair {
			m.PairUnknownUsageCalls++
			m.PairPartialInputTokens += input
			m.PairPartialOutputTokens += output
		}
		return
	}
	m.KnownUsageCalls++
	m.KnownInputTokens += input
	m.KnownOutputTokens += output
	if pair {
		m.PairKnownUsageCalls++
		m.PairKnownInputTokens += input
		m.PairKnownOutputTokens += output
	}
}

func (m *DeciderMetrics) addStability(r Report, rows []int, j int) {
	first := r.Outcomes[rows[0]][j]
	modelStable, jointStable, allCorrect := true, true, true
	labelled := r.Labelled(rows[0])
	for _, i := range rows {
		outcome := r.Outcomes[i][j]
		usable := outcome.Started && !outcome.Cancelled && outcome.Err == ""
		modelStable = modelStable && usable && outcome.Model == first.Model
		jointStable = jointStable && usable && outcome.Model == first.Model && outcome.Level != "" && outcome.Level == first.Level
		allCorrect = allCorrect && usable && r.Correct(i, j)
	}
	if allCorrect {
		m.RobustCorrectCases++
	}
	if len(rows) < 2 {
		return
	}
	m.RepeatedCases++
	if modelStable {
		m.StableModelCases++
	}
	if labelled {
		m.JointRepeatedCases++
		if jointStable {
			m.StableJointCases++
		}
	}
}

// routingDirection orders ladder misses only when every acceptable model
// is offered on the same ladder. Between disjoint acceptable ranks, or
// with a specialist, the direction is ambiguous and stays unclassified.
func (r Report) routingDirection(i int, model string) int {
	if i >= len(r.Inputs) {
		return 0
	}
	candidates := r.Inputs[i].Candidates
	rank := slices.IndexFunc(candidates, func(c decide.Candidate) bool { return c.ID == model })
	if rank < 0 || candidates[rank].Specialist {
		return 0
	}
	low, high := len(candidates), -1
	for _, expected := range r.Cases[i].Expect {
		index := slices.IndexFunc(candidates, func(c decide.Candidate) bool { return c.ID == expected })
		if index < 0 || candidates[index].Specialist {
			return 0
		}
		low, high = min(low, index), max(high, index)
	}
	switch {
	case rank < low:
		return -1
	case rank > high:
		return 1
	default:
		return 0
	}
}

func caseGroup(c Case) string {
	if c.Group != "" {
		return c.Group
	}
	if group, _, found := strings.Cut(c.Name, "/"); found {
		return group
	}
	return "ungrouped"
}

func addGroup(groups map[string]*GroupMetrics, name string, r Report, i, j int) {
	g := groups[name]
	if g == nil {
		g = &GroupMetrics{Name: name}
		groups[name] = g
	}
	g.Planned++
	if r.Correct(i, j) {
		g.Correct++
	}
	if r.ChoiceCorrect(i, j) {
		g.ChoiceCorrect++
	}
	if r.Labelled(i) {
		g.Labelled++
		if r.LevelCorrect(i, j) {
			g.LevelCorrect++
			if r.ChoiceCorrect(i, j) {
				g.ChoiceJointCorrect++
			}
			if r.Correct(i, j) {
				g.JointCorrect++
			}
		}
	}
}

func finishGroups(groups map[string]*GroupMetrics) []GroupMetrics {
	result := make([]GroupMetrics, 0, len(groups))
	for _, name := range slices.Sorted(maps.Keys(groups)) {
		g := *groups[name]
		g.Accuracy = share(g.Correct, g.Planned)
		g.ChoiceAccuracy = share(g.ChoiceCorrect, g.Planned)
		g.LevelAccuracy, g.JointAccuracy = share(g.LevelCorrect, g.Labelled), share(g.JointCorrect, g.Labelled)
		g.ChoiceJointAccuracy = share(g.ChoiceJointCorrect, g.Labelled)
		result = append(result, g)
	}
	return result
}

func share(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(n) / float64(total)
}

func tokenMean(tokens int64, calls int) float64 {
	if calls == 0 {
		return 0
	}
	return float64(tokens) / float64(calls)
}

// latencyPercentileMS uses nearest rank without rounding away sub-ms work.
func latencyPercentileMS(sorted []time.Duration, p int) float64 {
	if len(sorted) == 0 {
		return 0
	}
	index := (len(sorted)*p+99)/100 - 1
	return float64(sorted[max(0, min(index, len(sorted)-1))]) / float64(time.Millisecond)
}

func (r Report) writeMetrics(w io.Writer) {
	metrics := r.summaries()
	fmt.Fprintln(w, "\nmetrics: accuracy denominators include every planned trial; joint scores use labelled trials")
	table := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(table, "decider\tstarted\tcancelled\tfinal+depth\tchoice+depth\tmissing depth\tunder\tover\tother miss\tcalls\tpairs\tknown usage\treported tokens")
	for _, m := range metrics {
		fmt.Fprintf(table, "%s\t%d/%d\t%d\t%d/%d\t%d/%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d/%d\t%d in + %d out\n",
			m.Decider, m.Started, m.Planned, m.Cancelled, m.JointCorrect, m.Labelled, m.ChoiceJointCorrect, m.Labelled, m.MissingDepth,
			m.UnderRouted, m.OverRouted, m.OtherMisses, m.Calls, m.PairCalls, m.KnownUsageCalls, m.Calls,
			m.InputTokens, m.OutputTokens)
	}
	table.Flush()
	fmt.Fprintln(w, "\ngroups and acceptable model sets:")
	groups := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(groups, "decider\tkind\tname\tfinal model\tfinal+depth\tchoice\tchoice+depth")
	for _, m := range metrics {
		for _, g := range m.Groups {
			fmt.Fprintf(groups, "%s\tgroup\t%s\t%d/%d (%.0f%%)\t%d/%d\t%d/%d (%.0f%%)\t%d/%d\n", m.Decider, g.Name, g.Correct, g.Planned, 100*g.Accuracy, g.JointCorrect, g.Labelled, g.ChoiceCorrect, g.Planned, 100*g.ChoiceAccuracy, g.ChoiceJointCorrect, g.Labelled)
		}
		for _, g := range m.Tiers {
			fmt.Fprintf(groups, "%s\ttier\t%s\t%d/%d (%.0f%%)\t%d/%d\t%d/%d (%.0f%%)\t%d/%d\n", m.Decider, g.Name, g.Correct, g.Planned, 100*g.Accuracy, g.JointCorrect, g.Labelled, g.ChoiceCorrect, g.Planned, 100*g.ChoiceAccuracy, g.ChoiceJointCorrect, g.Labelled)
		}
		fmt.Fprintf(groups, "%s\tmacro/weakest\tgroups\t%.0f%% / %.0f%%\t%.0f%% / %.0f%%\n", m.Decider, 100*m.GroupMacroAccuracy, 100*m.WeakestGroupAccuracy, 100*m.GroupMacroJointAccuracy, 100*m.WeakestGroupJointAccuracy)
	}
	groups.Flush()
	fmt.Fprintln(w, "\nreported pair tokens: conditional means use known pair usage; amortized means use started decisions")
	for _, m := range metrics {
		fmt.Fprintf(w, "%s: %d pairs (%d failed), known usage %d/%d, mean %.1f tokens/pair, amortized %.1f tokens/started decision; %d unique cases, model stability %d/%d, joint stability %d/%d\n",
			m.Decider, m.PairCalls, m.FailedPairs, m.PairKnownUsageCalls, m.PairCalls,
			m.MeanReportedTokensPerKnownPair, m.AmortizedReportedPairTokensPerStarted, m.UniqueCases,
			m.StableModelCases, m.RepeatedCases, m.StableJointCases, m.JointRepeatedCases)
	}
	fmt.Fprintln(w, "Token totals include available partial counters; unknown usage may add more. Repeats measure stability on the same tasks; they are not independent held-out cases.")
	r.writeComparisons(w)
}

func (r Report) summaries() []DeciderMetrics {
	metrics := make([]DeciderMetrics, 0, len(r.Deciders))
	for j := range r.Deciders {
		metrics = append(metrics, r.Summary(j))
	}
	return metrics
}

// WriteJSON writes a reproducible versioned report with prepared inputs,
// policies, attempts, execution settings and the same scores as text output.
func (r Report) WriteJSON(w io.Writer) error {
	comparisons, err := r.Compare(r.Reference)
	if err != nil {
		return err
	}
	type caseJSON struct {
		Name      string   `json:"name"`
		Trial     int      `json:"trial"`
		Group     string   `json:"group"`
		Expect    []string `json:"expect"`
		Reasoning []string `json:"reasoning"`
		Images    bool     `json:"images"`
		Offered   []string `json:"offered"`
		Note      string   `json:"note,omitempty"`
		Prompt    string   `json:"prompt"`
	}
	type inputJSON struct {
		State      string                       `json:"state"`
		Candidates []decide.Candidate           `json:"candidates"`
		Efforts    map[string]map[string]string `json:"efforts"`
	}
	type outcomeJSON struct {
		Model     string           `json:"model"`
		Choice    string           `json:"choice"`
		Level     string           `json:"task_depth"`
		Effort    string           `json:"applied_effort"`
		Above     float64          `json:"above"`
		Escalated bool             `json:"escalated"`
		Started   bool             `json:"started"`
		Cancelled bool             `json:"cancelled"`
		LatencyMS float64          `json:"latency_ms"`
		Err       string           `json:"error,omitempty"`
		Attempts  []decide.Attempt `json:"attempts"`
		Policy    string           `json:"policy"`
		Costs     decide.Costs     `json:"routing_costs"`
	}
	data := struct {
		SchemaVersion int                      `json:"schema_version"`
		Settings      any                      `json:"settings"`
		Cases         []caseJSON               `json:"cases"`
		Deciders      []string                 `json:"deciders"`
		Inputs        []inputJSON              `json:"inputs"`
		Outcomes      [][]outcomeJSON          `json:"outcomes"`
		Policies      map[string]decide.Policy `json:"policies"`
		Metrics       []DeciderMetrics         `json:"metrics"`
		Reference     string                   `json:"reference,omitempty"`
		Comparisons   []Comparison             `json:"comparisons,omitempty"`
	}{SchemaVersion: 2, Settings: r.Settings, Deciders: r.Deciders, Policies: r.Policies, Reference: r.Reference, Comparisons: comparisons}
	for _, c := range r.Cases {
		data.Cases = append(data.Cases, caseJSON{Name: c.Name, Trial: c.Trial, Group: caseGroup(c), Expect: c.Expect,
			Reasoning: c.Reasoning, Images: c.Images, Offered: c.Offered, Note: c.Note, Prompt: c.Prompt})
	}
	for _, input := range r.Inputs {
		data.Inputs = append(data.Inputs, inputJSON{State: input.State, Candidates: input.Candidates, Efforts: input.Efforts})
	}
	for _, row := range r.Outcomes {
		outcomes := make([]outcomeJSON, 0, len(row))
		for _, o := range row {
			outcomes = append(outcomes, outcomeJSON{Model: o.Model, Choice: o.Choice, Level: o.Level, Effort: o.Effort,
				Above: o.Above, Escalated: o.Escalated, Started: o.Started, Cancelled: o.Cancelled,
				LatencyMS: float64(o.Latency) / float64(time.Millisecond), Err: o.Err, Attempts: o.Attempts, Policy: o.Policy, Costs: o.Costs})
		}
		data.Outcomes = append(data.Outcomes, outcomes)
	}
	data.Metrics = r.summaries()
	if err := json.MarshalWrite(w, data, json.Deterministic(true)); err != nil {
		return err
	}
	_, err = io.WriteString(w, "\n")
	return err
}
