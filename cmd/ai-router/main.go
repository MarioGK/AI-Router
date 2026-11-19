// Command ai-router serves the sticky per-thread `auto` model router in front
// of CLIProxy.
//
//	ai-router serve --config /etc/ai-router/config.yaml
//	ai-router check --config /etc/ai-router/config.yaml
//	ai-router bench --config /etc/ai-router/config.yaml --cases bench/routing.yaml
//	ai-router backup --db /var/lib/ai-router/router.db --to /var/backups/router.db
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/MarioGK/AI-Router/internal/app"
	"github.com/MarioGK/AI-Router/internal/catalog"
	"github.com/MarioGK/AI-Router/internal/config"
	"github.com/MarioGK/AI-Router/internal/decide"
	"github.com/MarioGK/AI-Router/internal/pin"
	"github.com/MarioGK/AI-Router/internal/proxy"
)

const usage = `usage: ai-router <command> [options]

The configuration document may also come from TS_ROUTER_CONFIG, which wins
over --config.

commands:
  serve   run the router; a second SIGINT or SIGTERM stops it at once
  check   validate the config, CLIProxy's model catalog and every decider
  bench   run every decider over labelled prompts (--cases), optionally
          --repeat n times, with --parallel n concurrent calls; report
          model and reasoning accuracy, latency, costs and disagreements
          as --format text|json; --reference name compares the results
          with that configured decider. --target-decider name selects a
          decider for gates (the configured chain by default). Fail an
          incomplete run or a target below --min-accuracy,
          --min-group-accuracy or --min-joint-accuracy. With a reference
          and target, --min-choice-agreement and --min-joint-agreement
          require full paired coverage. --score initial|final selects the
          model score for accuracy gates (final by default). Thresholds
          are fractions from 0 to 1. --no-chain skips the extra fallback
          chain replay. --priority auto|quality|speed|cost and --ranking
          model-a,model-b replay the dashboard's priority overrides.
  repair-metering  count the requests logged before the 2026-09-30
          metering fix whose input still holds the whole-prompt estimate;
          with --apply repair and reprice them, keeping the logged values
  backup  write a consistent snapshot of the database (--db) to a new
          file (--to) while the router keeps running; needs no config
  export-state  prepare legacy routing and metering rows for TerraBase as
          JSON lines (--db, --to); source is read-only, needs no config
  hash-password  read one password from a non-terminal standard input
          and write its bcrypt hash to standard output; needs no config
`

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return 2
	}
	command := args[0]
	if command == "hash-password" {
		// Do not pass any password-shaped argument through flag parsing:
		// its diagnostics can echo flag values before this command runs.
		if len(args) != 1 {
			fmt.Fprintln(stderr, "ai-router: hash-password accepts passwords only through standard input")
			return 2
		}
		return hashPassword(os.Stdin, stdout, stderr)
	}
	flags := flag.NewFlagSet(command, flag.ContinueOnError)
	flags.SetOutput(stderr)
	path := flags.String("config", "/etc/ai-router/config.yaml", "configuration file")
	cases := flags.String("cases", "bench/routing.yaml", "bench: labelled cases file")
	parallel := flags.Int("parallel", 6, "bench: concurrent decider calls")
	repeat := flags.Int("repeat", 1, "bench: trials per labelled case")
	format := flags.String("format", "text", "bench: report format, text or json")
	reference := flags.String("reference", "", "bench: configured decider to compare results against")
	noChain := flags.Bool("no-chain", false, "bench: compare independent deciders without rerunning the fallback chain")
	priority := flags.String("priority", "auto", "bench: priority override, auto, quality, speed, or cost")
	ranking := flags.String("ranking", "", "bench: optional model preference, highest first, separated by commas")
	targetDecider := flags.String("target-decider", "", "bench: configured decider for gates, default configured chain")
	score := flags.String("score", "final", "bench: accuracy gates score initial choice or final routed model")
	minAccuracy := flags.Float64("min-accuracy", 0, "bench: target's least model accuracy for --score, 0 to 1")
	minGroupAccuracy := flags.Float64("min-group-accuracy", 0, "bench: target's least model accuracy in every group for --score, 0 to 1")
	minJointAccuracy := flags.Float64("min-joint-accuracy", 0, "bench: target's least joint model and task-depth accuracy for --score, 0 to 1")
	minChoiceAgreement := flags.Float64("min-choice-agreement", 0, "bench: target's least raw-choice agreement with reference, 0 to 1; requires full coverage")
	minJointAgreement := flags.Float64("min-joint-agreement", 0, "bench: target's least joint raw-choice and task-depth agreement with reference, 0 to 1; requires full coverage")
	database := flags.String("db", "/var/lib/ai-router/router.db", "backup/export-state: the router's legacy database")
	to := flags.String("to", "", "backup/export-state: the new output file")
	apply := flags.Bool("apply", false, "repair-metering: repair the rows instead of counting them")
	if err := flags.Parse(args[1:]); err != nil {
		return 2
	}
	switch command {
	case "serve", "check", "bench", "repair-metering":
	case "backup":
		return backup(*database, *to, stderr)
	case "export-state":
		return exportState(*database, *to, stderr)
	case "help", "-h", "--help":
		fmt.Fprint(stdout, usage)
		return 0
	default:
		fmt.Fprint(stderr, usage)
		return 2
	}
	options := benchOptions{Parallel: *parallel, Repeat: *repeat, Format: *format,
		Reference: *reference, TargetDecider: *targetDecider, Score: *score, MinAccuracy: *minAccuracy, NoChain: *noChain,
		Preferences: decide.Preferences{Priority: *priority, Ranking: strings.FieldsFunc(*ranking, func(r rune) bool { return r == ',' || r == ' ' || r == '\n' })}}
	flags.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "min-group-accuracy":
			options.MinGroupAccuracy = minGroupAccuracy
		case "min-joint-accuracy":
			options.MinJointAccuracy = minJointAccuracy
		case "min-choice-agreement":
			options.MinChoiceAgreement = minChoiceAgreement
		case "min-joint-agreement":
			options.MinJointAgreement = minJointAgreement
		}
	})
	if command == "bench" {
		if err := options.validate(); err != nil {
			fmt.Fprintf(stderr, "ai-router: %v\n", err)
			return 2
		}
	}
	cfg, err := config.Load(*path)
	if err != nil {
		fmt.Fprintf(stderr, "ai-router: %v\n", err)
		return 1
	}
	switch command {
	case "check":
		return check(cfg, stdout)
	case "bench":
		return runBench(cfg, *cases, options, stdout, stderr)
	case "repair-metering":
		return repairMetering(cfg, *apply, stdout, stderr)
	}
	log := slog.New(slog.NewJSONHandler(stdout, &slog.HandlerOptions{Level: cfg.Level()}))
	return serve(cfg, log)
}

func serve(cfg *config.Config, log *slog.Logger) int {
	router, err := app.New(cfg, log)
	if err != nil {
		log.Error("startup failed", "error", err)
		return 1
	}
	defer router.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	// The first signal drains; restoring the default handling lets a second
	// one stop the process at once.
	go func() {
		<-ctx.Done()
		stop()
	}()
	if err := router.Run(ctx); err != nil {
		log.Error("server failed", "error", err)
		return 1
	}
	return 0
}

// backup writes a snapshot of the router's database to a new file, without
// migrating it: a newer binary never changes the database under a running
// older router.
func backup(database, to string, stderr io.Writer) int {
	if to == "" {
		fmt.Fprintln(stderr, "ai-router: backup needs --to")
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	if err := pin.Backup(ctx, database, to); err != nil {
		fmt.Fprintf(stderr, "ai-router: backup: %v\n", err)
		return 1
	}
	return 0
}

func exportState(database, to string, stderr io.Writer) int {
	if to == "" {
		fmt.Fprintln(stderr, "ai-router: export-state needs --to")
		return 2
	}
	// Match Export's path cleaning before resolving aliases, and use the
	// same destination for the guard and file creation.
	database = filepath.Clean(database)
	to = filepath.Clean(to)
	// Exports include recorded prompts. Keep them private and never replace
	// an existing file, including the source database.
	if err := checkExportDestination(database, to); err != nil {
		fmt.Fprintf(stderr, "ai-router: export-state: %v\n", err)
		return 1
	}
	file, err := os.OpenFile(to, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		fmt.Fprintf(stderr, "ai-router: export-state: %v\n", err)
		return 1
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	err = pin.Export(ctx, database, file)
	err = errors.Join(err, file.Close())
	if err != nil {
		err = errors.Join(err, os.Remove(to))
		fmt.Fprintf(stderr, "ai-router: export-state: %v\n", err)
		return 1
	}
	return 0
}

// SQLite can create sidecars when opening a read-only WAL database. Do not
// let an absent sidecar become our output, even through directory aliases.
func checkExportDestination(database, to string) error {
	source, err := filepath.EvalSymlinks(database)
	if err != nil {
		return err
	}
	source, err = filepath.Abs(source)
	if err != nil {
		return err
	}
	sourceParent, err := filepath.EvalSymlinks(filepath.Dir(database))
	if err != nil {
		return err
	}
	alias, err := filepath.Abs(filepath.Join(sourceParent, filepath.Base(database)))
	if err != nil {
		return err
	}
	parent, err := filepath.EvalSymlinks(filepath.Dir(to))
	if err != nil {
		return err
	}
	output, err := filepath.Abs(filepath.Join(parent, filepath.Base(to)))
	if err != nil {
		return err
	}
	for _, path := range []string{source, alias} {
		for _, suffix := range []string{"", "-wal", "-shm", "-journal"} {
			if output == path+suffix {
				return fmt.Errorf("export output must not be the source database or a SQLite sidecar")
			}
		}
	}
	return nil
}

// repairMetering counts, or with apply repairs and reprices, the requests
// the metering fix came too late for.
func repairMetering(cfg *config.Config, apply bool, stdout, stderr io.Writer) int {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	store, err := pin.Open(cfg.Database)
	if err != nil {
		fmt.Fprintf(stderr, "ai-router: repair-metering: %v\n", err)
		return 1
	}
	defer store.Close()
	n, err := store.RepairMetering(ctx, apply)
	if err != nil {
		fmt.Fprintf(stderr, "ai-router: repair-metering: %v\n", err)
		return 1
	}
	if !apply {
		fmt.Fprintf(stdout, "%d requests would be repaired; run with --apply to repair them\n", n)
		return 0
	}
	priced, err := store.RepriceRepaired(ctx, cfg.Price)
	if err != nil {
		fmt.Fprintf(stderr, "ai-router: repair-metering: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "repaired %d requests, priced %d\n", n, priced)
	return 0
}

// probe is a request every decider must be able to judge.
const probe = "images=no\n\nRename the variable `cnt` to `count` in main.go."

// check reports CLIProxy's catalog against the configuration and asks every
// decider the probe; it fails when a model is missing or a decider errors.
func check(cfg *config.Config, stdout io.Writer) int {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	upstream := proxy.NewClient()
	models := catalog.New(cfg.Upstream.BaseURL, cfg.UpstreamKey(), upstream)
	failed := false
	report := map[string]any{"config": "ok"}
	if err := models.Refresh(ctx); err != nil {
		report["catalog"] = map[string]string{"error": err.Error()}
		failed = true
	} else {
		status := models.Check(cfg.Auto.Required())
		report["catalog"] = status
		failed = failed || !status.Healthy()
		// A cap on a model CLIProxy does not serve is most likely a typo.
		var uncapped []string
		for model := range cfg.Limits {
			if !models.Has(model) {
				uncapped = append(uncapped, model)
			}
		}
		if len(uncapped) > 0 {
			slices.Sort(uncapped)
			report["limits"] = map[string]any{"not_served": uncapped}
			failed = true
		}
	}
	candidates := decideCandidates(cfg)
	deciders := map[string]any{}
	for _, step := range app.Steps(cfg, upstream) {
		start := time.Now()
		chain := decide.NewChain([]decide.Step{step}, cfg.Auto.EscalationThreshold, cfg.Auto.Doubt,
			cfg.Auto.Reasoning.Criteria())
		decision, ok := chain.Decide(ctx, probe, candidates)
		if !ok {
			failed = true
			deciders[step.Decider.Name()] = map[string]any{"failures": decision.Failures()}
			continue
		}
		deciders[step.Decider.Name()] = map[string]any{
			"model": decision.Model, "level": decision.Level, "escalated": decision.Escalated,
			"attempt": decision.Attempts[0], "latency_ms": time.Since(start).Milliseconds(),
		}
	}
	report["deciders"] = deciders
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	_ = encoder.Encode(report)
	if failed {
		return 1
	}
	return 0
}

// decideCandidates are the candidates as the deciders see them.
func decideCandidates(cfg *config.Config) []decide.Candidate {
	candidates := make([]decide.Candidate, len(cfg.Auto.Candidates))
	for i, c := range cfg.Auto.Candidates {
		candidates[i] = c.Criterion()
	}
	return candidates
}
