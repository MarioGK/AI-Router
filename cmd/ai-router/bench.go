package main

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"slices"
	"syscall"
	"time"

	"github.com/MarioGK/AI-Router/internal/app"
	"github.com/MarioGK/AI-Router/internal/bench"
	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/costs"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/proxy"
)

const benchTimeout = 30 * time.Minute

type benchOptions struct {
	Parallel           int                `json:"parallel"`
	Repeat             int                `json:"repeat"`
	Format             string             `json:"format"`
	Reference          string             `json:"reference,omitempty"`
	TargetDecider      string             `json:"target_decider,omitempty"`
	Score              string             `json:"score"`
	NoChain            bool               `json:"no_chain,omitzero"`
	Preferences        decide.Preferences `json:"preferences"`
	MinAccuracy        float64            `json:"min_accuracy"`
	MinGroupAccuracy   *float64           `json:"min_group_accuracy,omitzero"`
	MinJointAccuracy   *float64           `json:"min_joint_accuracy,omitzero"`
	MinChoiceAgreement *float64           `json:"min_choice_agreement,omitzero"`
	MinJointAgreement  *float64           `json:"min_joint_agreement,omitzero"`
}

func (o benchOptions) validate() error {
	if err := o.Preferences.Validate(); err != nil {
		return fmt.Errorf("invalid --priority or --ranking: %w", err)
	}
	if o.Parallel <= 0 {
		return fmt.Errorf("--parallel must be positive")
	}
	if o.Repeat <= 0 {
		return fmt.Errorf("--repeat must be positive")
	}
	if o.Format != "text" && o.Format != "json" {
		return fmt.Errorf("--format must be text or json")
	}
	if o.Score != "initial" && o.Score != "final" {
		return fmt.Errorf("--score must be initial or final")
	}
	for _, threshold := range []struct {
		name  string
		value *float64
	}{{"min-accuracy", &o.MinAccuracy}, {"min-group-accuracy", o.MinGroupAccuracy}, {"min-joint-accuracy", o.MinJointAccuracy},
		{"min-choice-agreement", o.MinChoiceAgreement}, {"min-joint-agreement", o.MinJointAgreement}} {
		if threshold.value == nil {
			continue
		}
		value := *threshold.value
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
			return fmt.Errorf("--%s must be finite and between 0 and 1", threshold.name)
		}
	}
	if o.MinChoiceAgreement != nil || o.MinJointAgreement != nil {
		if o.Reference == "" || o.TargetDecider == "" {
			return fmt.Errorf("agreement gates need both --reference and --target-decider")
		}
		if o.Reference == o.TargetDecider {
			return fmt.Errorf("agreement gates need distinct --reference and --target-decider names")
		}
	}
	return nil
}

// benchPreparer uses the same cleaned state, capabilities, candidate order
// and reasoning effort mapping as a new routing decision. Offered simulates
// availability without consulting or changing the live service.
func benchPreparer(cfg *config.Config) func(bench.Case) bench.Input {
	builder := decide.NewStateBuilder(cfg.Auto.StripTags, cfg.Auto.StateMaxChars)
	levels := append([]string{""}, cfg.Auto.Reasoning.IDs()...)
	return func(c bench.Case) bench.Input {
		input := bench.Input{State: builder.Build(c.Prompt, c.Images).Text,
			Efforts: map[string]map[string]string{}}
		for _, candidate := range cfg.Auto.Candidates {
			if cfg.Auto.IsRetired(candidate.ID) || c.Images && !candidate.Vision || len(c.Offered) > 0 && !slices.Contains(c.Offered, candidate.ID) {
				continue
			}
			input.Candidates = append(input.Candidates, candidate.Criterion())
			input.Efforts[candidate.ID] = map[string]string{}
			for _, level := range levels {
				input.Efforts[candidate.ID][level] = cfg.Auto.Effort(candidate.ID, level)
			}
		}
		return input
	}
}

// runBench validates every input before making requests, then judges every
// trial with each decider and the configured chain within the routing budget.
func runBench(cfg *config.Config, path string, options benchOptions, stdout, stderr io.Writer) int {
	if err := options.validate(); err != nil {
		fmt.Fprintf(stderr, "ai-router: %v\n", err)
		return 2
	}
	chainName := "chain(" + cfg.Auto.Doubt + ")"
	if len(cfg.Deciders) > 1 && slices.ContainsFunc(cfg.Deciders, func(d config.Decider) bool { return d.Name == chainName }) {
		fmt.Fprintf(stderr, "ai-router: decider name %q conflicts with the benchmark's configured chain\n", chainName)
		return 2
	}
	if options.Reference != "" && !slices.ContainsFunc(cfg.Deciders, func(d config.Decider) bool { return d.Name == options.Reference }) {
		fmt.Fprintf(stderr, "ai-router: --reference %q is not a configured decider\n", options.Reference)
		return 2
	}
	for _, model := range options.Preferences.Ranking {
		if cfg.Auto.Rank(model) < 0 || cfg.Auto.IsRetired(model) {
			fmt.Fprintf(stderr, "ai-router: ranked model %q is not an active configured candidate\n", model)
			return 2
		}
	}
	if options.TargetDecider != "" && !slices.ContainsFunc(cfg.Deciders, func(d config.Decider) bool { return d.Name == options.TargetDecider }) {
		fmt.Fprintf(stderr, "ai-router: --target-decider %q is not a configured decider\n", options.TargetDecider)
		return 2
	}
	if options.MinJointAgreement != nil && len(cfg.Auto.Reasoning.Levels) == 0 {
		fmt.Fprintln(stderr, "ai-router: --min-joint-agreement needs the configured reasoning chooser")
		return 2
	}
	cases, err := bench.Load(path)
	if err != nil {
		fmt.Fprintf(stderr, "ai-router: %v\n", err)
		return 1
	}
	levels := cfg.Auto.Reasoning.Criteria()
	if err := bench.Validate(cases, decideCandidates(cfg), levels); err != nil {
		fmt.Fprintf(stderr, "ai-router: %v\n", err)
		return 1
	}
	if options.MinJointAccuracy != nil && !slices.ContainsFunc(cases, func(c bench.Case) bool { return len(c.Reasoning) > 0 }) {
		fmt.Fprintln(stderr, "ai-router: --min-joint-accuracy needs cases with reasoning labels")
		return 1
	}
	tasks := costs.New(nil, cfg.Price, cfg.CostFactor, cfg.Auto.Cost.Window, *cfg.Auto.Cost.Weight)
	preferenceEvidence := decide.NewChain(nil, cfg.Auto.EscalationThreshold, cfg.Auto.Doubt, levels).
		Price(decide.Economy{Task: func(model, level string, _ []decide.Candidate) (float64, bool) {
			return tasks.Task(model, level)
		}}).Select(nil).Prefer(options.Preferences)
	prepare := benchPreparer(cfg)
	inputs, err := bench.Prepare(cases, func(c bench.Case) bench.Input {
		input := prepare(c)
		if cfg.Auto.Cost.Enabled {
			input.Candidates = preferenceEvidence.Candidates(input.Candidates)
		}
		return input
	})
	if err != nil {
		fmt.Fprintf(stderr, "ai-router: %v\n", err)
		return 1
	}
	prepared := make(map[string]bench.Input, len(cases))
	for i, c := range cases {
		prepared[c.Name] = inputs[i]
	}
	cases = bench.Repeat(cases, options.Repeat)
	signalCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(signalCtx, benchTimeout)
	defer cancel()
	steps := app.Steps(cfg, proxy.NewClient())
	// A benchmark has no settled request history. Every task estimate uses
	// the typical coding-task prior, with the effective prices and factors.
	chain := func(steps []decide.Step) *decide.Selection {
		c := decide.NewChain(steps, cfg.Auto.EscalationThreshold, cfg.Auto.Doubt, levels)
		if cfg.Auto.Cost.Enabled {
			c.Price(decide.Economy{Task: func(model, level string, _ []decide.Candidate) (float64, bool) {
				return tasks.Task(model, level)
			}, Miss: *cfg.Auto.Cost.Miss})
		}
		return c.Select(nil).Prefer(options.Preferences)
	}
	var chains []bench.Chain
	for _, step := range steps {
		chains = append(chains, bench.Chain{Name: step.Decider.Name(), Chain: chain([]decide.Step{step})})
	}
	if len(steps) > 1 && !options.NoChain {
		chains = append(chains, bench.Chain{Name: chainName, Chain: chain(steps)})
	}
	report := bench.Run(ctx, chains, func(c bench.Case) bench.Input { return prepared[c.Name] },
		cases, options.Parallel, cfg.Auto.DecideTimeout)
	report.Settings = benchSnapshot(cfg, options, tasks)
	report.Reference = options.Reference
	if options.Format == "json" {
		if err := report.WriteJSON(stdout); err != nil {
			fmt.Fprintf(stderr, "ai-router: write benchmark report: %v\n", err)
			return 1
		}
	} else {
		report.Write(stdout)
	}
	return benchExit(report, options, stderr)
}

func benchExit(report bench.Report, options benchOptions, stderr io.Writer) int {
	failed := false
	for j, name := range report.Deciders {
		summary := report.Summary(j)
		if !summary.Completed {
			fmt.Fprintf(stderr, "ai-router: %s benchmark is incomplete: %d/%d started, %d cancelled, %d errors\n",
				name, summary.Started, summary.Planned, summary.Cancelled, summary.Errors)
			failed = true
		}
	}
	j := len(report.Deciders) - 1
	if options.TargetDecider != "" {
		j = slices.Index(report.Deciders, options.TargetDecider)
		if j < 0 {
			fmt.Fprintf(stderr, "ai-router: target decider %q is not in the report\n", options.TargetDecider)
			return 1
		}
	}
	if j < 0 {
		fmt.Fprintln(stderr, "ai-router: benchmark has no deciders")
		return 1
	}
	name := report.Deciders[j]
	summary := report.Summary(j)
	accuracy, jointAccuracy := summary.Accuracy, summary.JointAccuracy
	if options.Score == "initial" {
		accuracy, jointAccuracy = summary.ChoiceAccuracy, summary.ChoiceJointAccuracy
	}
	if accuracy < options.MinAccuracy {
		if options.Score == "initial" {
			fmt.Fprintf(stderr, "ai-router: %s selected the initial model correctly for %.0f%% of the cases, below %.0f%%\n",
				name, 100*accuracy, 100*options.MinAccuracy)
		} else {
			fmt.Fprintf(stderr, "ai-router: %s routed %.0f%% of the cases correctly, below %.0f%%\n",
				name, 100*accuracy, 100*options.MinAccuracy)
		}
		failed = true
	}
	if options.MinGroupAccuracy != nil {
		threshold := *options.MinGroupAccuracy
		for _, group := range summary.Groups {
			accuracy := group.Accuracy
			if options.Score == "initial" {
				accuracy = group.ChoiceAccuracy
			}
			if accuracy < threshold {
				fmt.Fprintf(stderr, "ai-router: %s scored %.0f%% model accuracy in group %q (%s), below %.0f%%\n",
					name, 100*accuracy, group.Name, options.Score, 100*threshold)
				failed = true
			}
		}
	}
	if options.MinJointAccuracy != nil {
		threshold := *options.MinJointAccuracy
		if summary.Labelled == 0 {
			fmt.Fprintln(stderr, "ai-router: --min-joint-accuracy needs cases with reasoning labels")
			failed = true
		} else if jointAccuracy < threshold {
			fmt.Fprintf(stderr, "ai-router: %s chose the model and reasoning correctly for %.0f%% of labelled cases (%s), below %.0f%%\n",
				name, 100*jointAccuracy, options.Score, 100*threshold)
			failed = true
		}
	}
	if options.MinChoiceAgreement != nil || options.MinJointAgreement != nil {
		comparisons, err := report.Compare(options.Reference)
		if err != nil {
			fmt.Fprintf(stderr, "ai-router: %v\n", err)
			return 1
		}
		index := slices.IndexFunc(comparisons, func(c bench.Comparison) bool { return c.Decider == options.TargetDecider })
		if index < 0 {
			fmt.Fprintf(stderr, "ai-router: target decider %q has no comparison with %q\n", options.TargetDecider, options.Reference)
			return 1
		}
		comparison := comparisons[index]
		for _, gate := range []struct {
			name      string
			question  string
			threshold *float64
			score     bench.AgreementScore
		}{{"min-choice-agreement", "raw choice", options.MinChoiceAgreement, comparison.Choice},
			{"min-joint-agreement", "joint raw choice and task depth", options.MinJointAgreement, comparison.Joint}} {
			if gate.threshold == nil {
				continue
			}
			if gate.score.Compared != comparison.Planned {
				fmt.Fprintf(stderr, "ai-router: %s has %s comparison coverage %d/%d against %s; --%s needs every planned trial\n",
					options.TargetDecider, gate.question, gate.score.Compared, comparison.Planned, options.Reference, gate.name)
				failed = true
			} else if gate.score.Agreement < *gate.threshold {
				fmt.Fprintf(stderr, "ai-router: %s agreed with %s on %.0f%% of %s answers, below %.0f%%\n",
					options.TargetDecider, options.Reference, 100*gate.score.Agreement, gate.question, 100*(*gate.threshold))
				failed = true
			}
		}
	}
	if failed {
		return 1
	}
	return 0
}

// benchSettings deliberately contains only settings that affect the run;
// credentials, secret names, endpoint URLs and unrelated service settings
// never enter the benchmark document.
type benchSettings struct {
	Options         benchOptions          `json:"options"`
	RunTimeoutMS    int64                 `json:"run_timeout_ms"`
	DecideTimeoutMS int64                 `json:"decide_timeout_ms"`
	Doubt           string                `json:"doubt"`
	Threshold       float64               `json:"escalation_threshold"`
	StripTags       []string              `json:"strip_tags"`
	StateMaxChars   int                   `json:"state_max_chars"`
	Retired         []string              `json:"retired"`
	Candidates      []benchCandidate      `json:"candidates"`
	Deciders        []benchDecider        `json:"deciders"`
	Prices          map[string]benchPrice `json:"prices_usd_per_million_tokens"`
	Cost            benchCost             `json:"cost"`
}

type benchCandidate struct {
	ID         string   `json:"id"`
	Vision     bool     `json:"vision"`
	Reasoning  string   `json:"default_effort"`
	Efforts    []string `json:"accepted_efforts"`
	CostFactor float64  `json:"cost_factor"`
}

type benchDecider struct {
	Name      string `json:"name"`
	Type      string `json:"type"`
	Model     string `json:"model"`
	TimeoutMS int64  `json:"timeout_ms"`
}

type benchPrice struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cache_read"`
	CacheWrite float64 `json:"cache_write"`
}

type benchCost struct {
	Enabled        bool                `json:"enabled"`
	Miss           float64             `json:"miss"`
	WindowSeconds  float64             `json:"window_seconds"`
	Weight         float64             `json:"weight"`
	EstimateSource string              `json:"estimate_source"`
	TypicalTokens  benchTokens         `json:"typical_tokens"`
	TaskEstimates  []benchTaskEstimate `json:"task_estimates"`
}

type benchTokens struct {
	Input      int64 `json:"input"`
	Output     int64 `json:"output"`
	CacheRead  int64 `json:"cache_read"`
	CacheWrite int64 `json:"cache_write"`
}

type benchTaskEstimate struct {
	Model   string  `json:"model"`
	Level   string  `json:"level"`
	CostUSD float64 `json:"cost_usd"`
}

func benchSnapshot(cfg *config.Config, options benchOptions, tasks *costs.Table) benchSettings {
	settings := benchSettings{Options: options, RunTimeoutMS: benchTimeout.Milliseconds(),
		DecideTimeoutMS: cfg.Auto.DecideTimeout.Milliseconds(), Doubt: cfg.Auto.Doubt,
		Threshold: cfg.Auto.EscalationThreshold, StripTags: slices.Clone(cfg.Auto.StripTags),
		StateMaxChars: cfg.Auto.StateMaxChars, Retired: slices.Clone(cfg.Auto.Retired), Prices: map[string]benchPrice{},
		Cost: benchCost{Enabled: cfg.Auto.Cost.Enabled, Miss: *cfg.Auto.Cost.Miss,
			WindowSeconds: cfg.Auto.Cost.Window.Seconds(), Weight: *cfg.Auto.Cost.Weight,
			EstimateSource: "typical_task_prior", TypicalTokens: benchTokens{
				Input: costs.Typical.Input, Output: costs.Typical.Output,
				CacheRead: costs.Typical.CacheRead, CacheWrite: costs.Typical.CacheWrite}}}
	var models []string
	for _, candidate := range cfg.Auto.Candidates {
		models = append(models, candidate.ID)
		settings.Candidates = append(settings.Candidates, benchCandidate{ID: candidate.ID,
			Vision: candidate.Vision, Reasoning: candidate.Reasoning,
			Efforts: slices.Clone(candidate.Efforts), CostFactor: cfg.CostFactor(candidate.ID)})
	}
	for _, decider := range cfg.Deciders {
		settings.Deciders = append(settings.Deciders, benchDecider{Name: decider.Name,
			Type: decider.Type, Model: decider.Model, TimeoutMS: decider.Timeout.Milliseconds()})
	}
	for model, price := range cfg.Prices {
		settings.Prices[model] = benchPrice{Input: price.Input, Output: price.Output,
			CacheRead: price.CacheRead, CacheWrite: price.CacheWrite}
	}
	for _, estimate := range tasks.Estimates(models, cfg.Auto.Reasoning.IDs()) {
		settings.Cost.TaskEstimates = append(settings.Cost.TaskEstimates,
			benchTaskEstimate{Model: estimate.Model, Level: estimate.Level, CostUSD: estimate.Cost})
	}
	return settings
}
