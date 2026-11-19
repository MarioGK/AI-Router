// Package bench measures routing decisions against labelled prompts: every
// decider judges every case on its own, and the report shows per-decider
// accuracy (of the model and, where a case labels it, of the reasoning
// level), latency, escalations and where the deciders disagree.
package bench

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"maps"
	"os"
	"slices"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"go.yaml.in/yaml/v3"

	"github.com/MarioGK/AI-Router/internal/decide"
)

// Case is one labelled prompt. Expect lists every acceptable model and
// Reasoning, when set, every acceptable reasoning level; Note says why.
type Case struct {
	Name      string   `yaml:"name" json:"name"`
	Group     string   `yaml:"group,omitempty" json:"group,omitempty"`
	Expect    []string `yaml:"expect,flow" json:"expect"`
	Reasoning []string `yaml:"reasoning,omitempty,flow" json:"reasoning,omitempty"`
	Images    bool     `yaml:"images,omitempty" json:"images"`
	Offered   []string `yaml:"offered,omitempty,flow" json:"offered,omitempty"`
	Note      string   `yaml:"note,omitempty" json:"note,omitempty"`
	Prompt    string   `yaml:"prompt" json:"prompt"`
	// Trial is the repetition number, starting at one after Repeat.
	Trial int `yaml:"-" json:"trial,omitzero"`
}

// Load reads a cases file.
func Load(path string) ([]Case, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file struct {
		Cases []Case `yaml:"cases"`
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&file); err != nil {
		return nil, fmt.Errorf("parse cases: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err != nil {
			return nil, fmt.Errorf("parse cases: %w", err)
		}
		return nil, fmt.Errorf("cases file must contain one YAML document")
	}
	if len(file.Cases) == 0 {
		return nil, fmt.Errorf("cases file is empty")
	}
	names := map[string]bool{}
	for _, c := range file.Cases {
		if err := validateCase(c); err != nil {
			return nil, err
		}
		if names[c.Name] {
			return nil, fmt.Errorf("case name %q is duplicated", c.Name)
		}
		names[c.Name] = true
	}
	return file.Cases, nil
}

func validateCase(c Case) error {
	if strings.TrimSpace(c.Name) == "" || len(c.Expect) == 0 || strings.TrimSpace(c.Prompt) == "" {
		return fmt.Errorf("case %q needs a name, expect and prompt", c.Name)
	}
	for _, labels := range []struct {
		name string
		ids  []string
	}{{"expect", c.Expect}, {"reasoning", c.Reasoning}, {"offered", c.Offered}} {
		seen := map[string]bool{}
		for _, id := range labels.ids {
			if strings.TrimSpace(id) == "" || strings.TrimSpace(id) != id {
				return fmt.Errorf("case %q has a blank or padded %s label", c.Name, labels.name)
			}
			if seen[id] {
				return fmt.Errorf("case %q has duplicate %s label %q", c.Name, labels.name, id)
			}
			seen[id] = true
		}
	}
	return nil
}

// Validate checks the case labels against the configured candidates and levels
// before any classifier is contacted. Expectations must be offered for the case.
func Validate(cases []Case, candidates []decide.Candidate, levels []decide.Level) error {
	models, reasoning := map[string]bool{}, map[string]bool{}
	for _, candidate := range candidates {
		models[candidate.ID] = true
	}
	for _, level := range levels {
		reasoning[level.ID] = true
	}
	if len(cases) == 0 {
		return fmt.Errorf("cases file is empty")
	}
	identities := map[string]map[int]bool{}
	for _, c := range cases {
		if err := validateCase(c); err != nil {
			return err
		}
		if c.Trial < 0 || identities[c.Name][c.Trial] {
			return fmt.Errorf("case %q has an invalid or duplicate trial %d", c.Name, c.Trial)
		}
		if identities[c.Name] == nil {
			identities[c.Name] = map[int]bool{}
		}
		identities[c.Name][c.Trial] = true
		for _, id := range append(slices.Clone(c.Expect), c.Offered...) {
			if !models[id] {
				return fmt.Errorf("case %q references unknown model %q", c.Name, id)
			}
		}
		for _, id := range c.Expect {
			if len(c.Offered) > 0 && !slices.Contains(c.Offered, id) {
				return fmt.Errorf("case %q expects model %q that is not offered", c.Name, id)
			}
		}
		for _, id := range c.Reasoning {
			if !reasoning[id] {
				return fmt.Errorf("case %q references unknown reasoning level %q", c.Name, id)
			}
		}
	}
	return nil
}

// Repeat expands every case into numbered trials without changing its identity.
func Repeat(cases []Case, repetitions int) []Case {
	var trials []Case
	for _, c := range cases {
		for trial := range repetitions {
			copy := snapshotCase(c)
			copy.Trial = trial + 1
			trials = append(trials, copy)
		}
	}
	return trials
}

func snapshotCase(c Case) Case {
	c.Expect = slices.Clone(c.Expect)
	c.Reasoning = slices.Clone(c.Reasoning)
	c.Offered = slices.Clone(c.Offered)
	return c
}

// Outcome is one decider's answer to one case: its own choice, the model
// after escalation, the probability on stronger ladder candidates and the
// reasoning level.
type Outcome struct {
	Model     string           `json:"model,omitempty"`
	Choice    string           `json:"choice,omitempty"`
	Level     string           `json:"level,omitempty"`
	Effort    string           `json:"effort,omitempty"`
	Above     float64          `json:"above"`
	Escalated bool             `json:"escalated"`
	Started   bool             `json:"started"`
	Cancelled bool             `json:"cancelled"`
	Latency   time.Duration    `json:"latency_ns"`
	Err       string           `json:"error,omitempty"`
	Attempts  []decide.Attempt `json:"attempts,omitempty"`
	Policy    string           `json:"policy,omitempty"`
	Costs     decide.Costs     `json:"costs"`
}

// Input is a case as the router would present it to the classifier, after
// stripping wrappers and restricting candidates by capabilities or availability.
type Input struct {
	State      string             `json:"state"`
	Candidates []decide.Candidate `json:"candidates"`
	// Efforts maps model and chosen reasoning level to the applied effort.
	// The empty level maps to the model's fixed or default effort.
	Efforts map[string]map[string]string `json:"efforts,omitempty"`
}

// Prepare validates and snapshots the exact inputs a benchmark will classify.
func Prepare(cases []Case, prepare func(Case) Input) ([]Input, error) {
	inputs := make([]Input, len(cases))
	for i, c := range cases {
		input, err := prepareInput(c, prepare)
		if err != nil {
			return nil, err
		}
		inputs[i] = input
	}
	return inputs, nil
}

func prepareInput(c Case, prepare func(Case) Input) (Input, error) {
	if prepare == nil {
		return Input{}, fmt.Errorf("case %q has no input preparation", c.Name)
	}
	input := prepare(c)
	input.Candidates = slices.Clone(input.Candidates)
	for i := range input.Candidates {
		input.Candidates[i].Examples = slices.Clone(input.Candidates[i].Examples)
	}
	input.Efforts = maps.Clone(input.Efforts)
	for model, efforts := range input.Efforts {
		input.Efforts[model] = maps.Clone(efforts)
	}
	if strings.TrimSpace(input.State) == "" {
		return input, fmt.Errorf("case %q has empty state after preparation", c.Name)
	}
	if len(input.Candidates) == 0 {
		return input, fmt.Errorf("case %q has no offered candidates after preparation", c.Name)
	}
	seen := map[string]bool{}
	for _, candidate := range input.Candidates {
		if strings.TrimSpace(candidate.ID) == "" || seen[candidate.ID] {
			return input, fmt.Errorf("case %q has blank or duplicate offered candidate %q", c.Name, candidate.ID)
		}
		seen[candidate.ID] = true
	}
	for _, expected := range c.Expect {
		if !seen[expected] {
			return input, fmt.Errorf("case %q expects model %q that is not offered after preparation", c.Name, expected)
		}
	}
	return input, nil
}

// Chain is a named decider chain to measure: one decider alone, or the
// router's whole configured chain.
type Chain struct {
	Name  string
	Chain Chooser
}

// Chooser accepts both a configured chain and a request-local selection.
type Chooser interface {
	Decide(context.Context, string, []decide.Candidate) (decide.Decision, bool)
	Policy([]decide.Candidate) decide.Policy
}

// Report holds every outcome, indexed by case then decider.
type Report struct {
	Cases     []Case                   `json:"cases"`
	Deciders  []string                 `json:"deciders"`
	Outcomes  [][]Outcome              `json:"outcomes"`
	Inputs    []Input                  `json:"inputs"`
	Policies  map[string]decide.Policy `json:"policies"`
	Settings  any                      `json:"settings,omitempty"`
	Reference string                   `json:"reference,omitempty"`
}

// Run asks every chain about every case, at most parallel calls at a time,
// each decision within budget.
func Run(ctx context.Context, chains []Chain, prepare func(Case) Input,
	cases []Case, parallel int, budget time.Duration) Report {
	report := Report{Cases: make([]Case, len(cases)), Outcomes: make([][]Outcome, len(cases)),
		Inputs: make([]Input, len(cases)), Policies: map[string]decide.Policy{}}
	for _, chain := range chains {
		report.Deciders = append(report.Deciders, chain.Name)
	}
	for i := range cases {
		report.Cases[i] = snapshotCase(cases[i])
		report.Outcomes[i] = make([]Outcome, len(chains))
		if err := ctx.Err(); err != nil {
			for j := range chains {
				report.Outcomes[i][j] = Outcome{Err: err.Error(), Cancelled: true}
			}
			continue
		}
		input, err := prepareInput(cases[i], prepare)
		report.Inputs[i] = input
		for j, chain := range chains {
			if err != nil {
				report.Outcomes[i][j].Err = err.Error()
				continue
			}
			if chain.Chain == nil {
				report.Outcomes[i][j].Err = "decider chain is nil"
				continue
			}
			if budget <= 0 {
				report.Outcomes[i][j].Err = "decision budget must be positive"
				continue
			}
			policy := chain.Chain.Policy(report.Inputs[i].Candidates)
			fingerprint := policy.Fingerprint()
			report.Policies[fingerprint] = policy
			report.Outcomes[i][j].Policy = fingerprint
		}
	}
	type trial struct{ caseIndex, deciderIndex int }
	jobs := make(chan trial)
	var wg sync.WaitGroup
	for range min(max(parallel, 1), len(cases)*len(chains)) {
		wg.Go(func() {
			for job := range jobs {
				i, j := job.caseIndex, job.deciderIndex
				outcome := report.Outcomes[i][j]
				if err := ctx.Err(); err != nil {
					outcome.Err, outcome.Cancelled = err.Error(), true
					report.Outcomes[i][j] = outcome
					continue
				}
				input := report.Inputs[i]
				decideCtx, cancel := context.WithTimeout(ctx, budget)
				outcome.Started = true
				decision, ok := chains[j].Chain.Decide(decideCtx, input.State, input.Candidates)
				decisionErr := decideCtx.Err()
				cancel()
				outcome.Latency, outcome.Attempts = decision.Latency, decision.Attempts
				outcome.Costs = decide.Costs{Task: maps.Clone(decision.Costs.Task), Done: maps.Clone(decision.Costs.Done)}
				switch {
				case ctx.Err() != nil:
					outcome.Err, outcome.Cancelled = ctx.Err().Error(), true
				case ok:
					outcome.Model, outcome.Choice, outcome.Escalated = decision.Model, decision.Choice, decision.Escalated
					outcome.Level = decision.Level
					outcome.Effort = input.Efforts[decision.Model][decision.Level]
					if outcome.Effort == "" {
						outcome.Effort = input.Efforts[decision.Model][""]
					}
					outcome.Above = decide.MassAbove(input.Candidates, decide.Verdict{Choice: decision.Choice, Probabilities: decision.Probabilities})
				default:
					outcome.Err = strings.Join(decision.Failures(), "; ")
					if outcome.Err == "" && decisionErr != nil {
						outcome.Err = decisionErr.Error()
					}
					if outcome.Err == "" {
						outcome.Err = "no decider returned a usable verdict"
					}
				}
				report.Outcomes[i][j] = outcome
			}
		})
	}
	for i := range cases {
		for j := range chains {
			if report.Outcomes[i][j].Err != "" {
				continue
			}
			select {
			case jobs <- trial{i, j}:
			case <-ctx.Done():
				report.Outcomes[i][j].Err = ctx.Err().Error()
				report.Outcomes[i][j].Cancelled = true
			}
		}
	}
	close(jobs)
	wg.Wait()
	return report
}

// Correct reports whether an outcome routed a case to an acceptable model.
func (r Report) Correct(caseIndex, deciderIndex int) bool {
	outcome := r.Outcomes[caseIndex][deciderIndex]
	return outcome.Err == "" && slices.Contains(r.Cases[caseIndex].Expect, outcome.Model)
}

// ChoiceCorrect reports whether the decider's own choice, before
// escalation, was acceptable.
func (r Report) ChoiceCorrect(caseIndex, deciderIndex int) bool {
	outcome := r.Outcomes[caseIndex][deciderIndex]
	return outcome.Err == "" && slices.Contains(r.Cases[caseIndex].Expect, outcome.Choice)
}

// Labelled reports whether a case labels its reasoning level.
func (r Report) Labelled(caseIndex int) bool {
	return len(r.Cases[caseIndex].Reasoning) > 0
}

// LevelCorrect reports whether an outcome chose an acceptable reasoning
// level for a case that labels one.
func (r Report) LevelCorrect(caseIndex, deciderIndex int) bool {
	outcome := r.Outcomes[caseIndex][deciderIndex]
	return outcome.Err == "" && slices.Contains(r.Cases[caseIndex].Reasoning, outcome.Level)
}

// Accuracy is the share of the cases a decider routed correctly.
func (r Report) Accuracy(deciderIndex int) float64 {
	if len(r.Cases) == 0 {
		return 0
	}
	correct := 0
	for i := range r.Cases {
		if r.Correct(i, deciderIndex) {
			correct++
		}
	}
	return float64(correct) / float64(len(r.Cases))
}

// Agree reports whether every decider routed a case to the same model.
func (r Report) Agree(caseIndex int) bool {
	outcomes := r.Outcomes[caseIndex]
	if len(outcomes) == 0 {
		return false
	}
	for _, outcome := range outcomes {
		if outcome.Err != "" || outcome.Model != outcomes[0].Model {
			return false
		}
	}
	return true
}

// Write prints the per-case table and the summary.
func (r Report) Write(w io.Writer) {
	table := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintf(table, "case\texpect\t%s\tagree\n", strings.Join(r.Deciders, "\t"))
	for i, c := range r.Cases {
		cells := make([]string, len(r.Deciders))
		for j, outcome := range r.Outcomes[i] {
			switch {
			case outcome.Cancelled:
				cells[j] = "CANCELLED"
			case outcome.Err != "":
				cells[j] = "ERROR"
			default:
				mark := "ok"
				if !r.Correct(i, j) {
					mark = "MISS"
				}
				escalated := ""
				if outcome.Escalated {
					escalated = fmt.Sprintf(" (^%s %.2f)", outcome.Choice, outcome.Above)
				}
				level := ""
				if outcome.Level != "" {
					level = " @" + outcome.Level
					if r.Labelled(i) && !r.LevelCorrect(i, j) {
						level += "!"
					}
				} else if r.Labelled(i) {
					level = " @missing!"
				}
				cells[j] = fmt.Sprintf("%s %s%s%s", mark, outcome.Model, level, escalated)
			}
		}
		agree := "yes"
		if !r.Agree(i) {
			agree = "NO"
		}
		expect := strings.Join(c.Expect, "|")
		if len(c.Reasoning) > 0 {
			expect += " @" + strings.Join(c.Reasoning, "|")
		}
		fmt.Fprintf(table, "%s\t%s\t%s\t%s\n", trialName(c), expect, strings.Join(cells, "\t"), agree)
	}
	table.Flush()

	fmt.Fprintln(w)
	summary := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(summary, "decider\tcorrect\taccuracy\tchoice correct\tlevel correct\tescalated\terrors\tp50\tp95")
	labelled := 0
	for i := range r.Cases {
		if r.Labelled(i) {
			labelled++
		}
	}
	for j, name := range r.Deciders {
		correct, choiceCorrect, levelCorrect, escalated, errors := 0, 0, 0, 0, 0
		var latencies []time.Duration
		for i := range r.Cases {
			outcome := r.Outcomes[i][j]
			if r.Correct(i, j) {
				correct++
			}
			if r.ChoiceCorrect(i, j) {
				choiceCorrect++
			}
			if r.Labelled(i) && r.LevelCorrect(i, j) {
				levelCorrect++
			}
			if outcome.Escalated {
				escalated++
			}
			if outcome.Err != "" {
				errors++
			}
			if outcome.Started {
				latencies = append(latencies, outcome.Latency)
			}
		}
		slices.Sort(latencies)
		fmt.Fprintf(summary, "%s\t%d/%d\t%.0f%%\t%d/%d\t%d/%d\t%d\t%d\t%s\t%s\n", name, correct, len(r.Cases),
			100*float64(correct)/float64(max(len(r.Cases), 1)), choiceCorrect, len(r.Cases), levelCorrect, labelled,
			escalated, errors, percentile(latencies, 50), percentile(latencies, 95))
	}
	summary.Flush()
	r.writeMetrics(w)
	agreed := 0
	for i := range r.Cases {
		if r.Agree(i) {
			agreed++
		}
	}
	fmt.Fprintf(w, "\nagreement: %d/%d cases routed to the same model by every decider\n", agreed, len(r.Cases))
	for j, name := range r.Deciders {
		for i, c := range r.Cases {
			if outcome := r.Outcomes[i][j]; outcome.Err != "" {
				fmt.Fprintf(w, "error %s/%s: %s\n", name, trialName(c), outcome.Err)
			}
		}
	}
}

func percentile(sorted []time.Duration, p int) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	rank := (len(sorted)*min(max(p, 0), 100) + 99) / 100
	return sorted[min(len(sorted)-1, max(rank-1, 0))].Round(time.Millisecond)
}

func trialName(c Case) string {
	if c.Trial > 0 {
		return fmt.Sprintf("%s#%d", c.Name, c.Trial)
	}
	return c.Name
}
