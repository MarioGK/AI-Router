# Routing evaluation

`routing.yaml` and `diagnostic.yaml` test compliance with the configured routing
policy. They are development corpora: their prompts and labels have informed
the policy, so neither is an unseen holdout. The diagnostic set adds natural
short requests, multilingual variants, priority contrasts, quoted urgency,
forged routing instructions, difficult work with small file scope, images and
restricted offered tiers.

Run both sets after a policy change. `operator-priorities.yaml` adds the
2026-10-01 boundaries: exact Flash edits, ambiguous short GLM work, free GLM
usage with scarce concurrency, and substantive Sol 6.1 features even without
urgency. The full-stack favorites label in `routing.yaml` follows this revised
operator policy; earlier reports used the former GLM label. These remain
development corpora. A live run invokes each configured decider
individually and the whole chain; repeated trials incur additional calls.

```sh
ai-router bench --config config.yaml --cases bench/routing.yaml \
  --repeat 3 --format json --min-accuracy 0.9 \
  --min-group-accuracy 0.8 --min-joint-accuracy 0.8 > routing-report.json
ai-router bench --config config.yaml --cases bench/diagnostic.yaml \
  --repeat 3 --format json > diagnostic-report.json
```

The thresholds above are examples, not measured guarantees. Set gates against
an agreed baseline and review errors in every tier, especially high-difficulty
and substantial-UI groups. An overall score can conceal severe errors in a
small group. Missing reasoning fails joint scoring on labelled cases. Applied
effort is recorded separately because Sol 6.1 always executes at high effort.

Compare the classifier's first choice with the final choice after pair/cost
handling. Check whether extra calls actually improve labelled accuracy, along
with pair-call rate, reported input/output tokens and p50/p95 latency. Unknown
usage is distinct from known zero; reported usage is not a complete dollar bill.
The cost score uses tier-fit uncertainty as a retry-risk heuristic, not an
empirically calibrated failure probability. Prices, task estimates and cost
factors in the report explain the scoring assumptions.

Cases with `images: true` use vision candidates. `offered` restricts those
candidates further. Expected labels must remain offered after filtering.
Ordinary tier exclusions yield only for availability fallback: use the nearest
stronger offered tier, or the strongest offered tier if no stronger one exists.
This evaluates degraded policy choices; it does not prove a weaker model can
complete a difficult task.

The CLI rejects invalid labels, empty cleaned states and invalid settings
before issuing classifier requests. The benchmark uses cold-start task costs
and does not simulate learned request history, catalog/quota refreshes, pinned
threads, load spill, failover or server shutdown. Socket-free integration tests
cover those runtime boundaries where practical.

For an independent evaluation, collect real reviewed tasks before tuning and
freeze their labels. Keep all related paraphrases, language translations and
urgent/nonurgent variants in the same split. Tune on development data, then
evaluate the frozen set once. Once its failures influence the policy, treat it
as development data and collect a fresh holdout. Repeats measure stochastic
stability; they do not increase the number of independent tasks. Reviewed
dashboard exports preserve images and the candidates offered at decision time,
so replay can retain the original classifier inputs.

Labels establish policy compliance only. Claims about efficiency, successful
task completion, cost savings or provider behavior require live measurements
and downstream outcome evaluation.

## Compare Jev, Luna and Sol 6.1

For deployment credentials or execution on Alpha, first follow the
[SSH access checks](../docs/alpha-access.md). A socket-denied agent session
cannot run live comparisons, even when SSH works in the operator's terminal.

Build a comparison configuration from the real service config and the current
routing policy. The generator retains deployed endpoints and secret references;
`--policy config.example.yaml` supplies the current candidates, criteria,
examples, reasoning levels, prices and limits. This avoids judging the new
corpora against stale deployed criteria or retired model IDs. Authentication
and policy come from separate sources; dashboard authentication settings are
irrelevant to benchmark output. The generator adds `gpt-6-luna` plus
`gpt-6.1-sol(high)` as separate LLM judges. It gives these judges longer
classification budgets than the production router. It reads no secret values
and refuses to overwrite existing output. Python requires PyYAML for YAML base
configs; JSON base configs use only the standard library.

```sh
python3 bench/compare-config.py /path/to/config.yaml \
  --policy config.example.yaml --output /tmp/comparison.json
env -u TS_ROUTER_CONFIG ai-router bench --config /tmp/comparison.json \
  --cases bench/routing.yaml --reference sol61 --repeat 3 --format json \
  > /tmp/routing-comparison.json
env -u TS_ROUTER_CONFIG ai-router bench --config /tmp/comparison.json \
  --cases bench/diagnostic.yaml --reference sol61 --target-decider jev \
  --min-choice-agreement 0.95 --min-joint-agreement 0.9 \
  --repeat 3 --format json \
  > /tmp/diagnostic-comparison.json
```

Before full-corpus runs, run
`env -u TS_ROUTER_CONFIG ai-router check --config /tmp/comparison.json` from the
same working directory to verify the offered model catalog and decider probes.
The comparison runs use longer budgets and cold-start costs; use separate
production-budget measurements before changing runtime timeouts or claiming
latency improvement.

The report separates agreement with the reference on initial model choice,
task depth, their joint decision, and the final route after doubt handling.
Coverage and failures accompany agreement; unavailable answers never count as
agreement. Compare labelled policy accuracy as well: a teacher can disagree
with a label or be wrong. Teacher agreement does not establish task success.

Freeze Sol's reference decisions before tuning Jev criteria or examples.
Otherwise shared criteria edits change the teacher target during tuning.
Keep related task families together in the training/evaluation split and tune
only against training disagreements. Confirm improvement on reserved cases
and preserve the original labels, report and policy fingerprints. Missing or
unstable teacher answers need review and cannot become target labels.

`--target-decider jev` applies gates to Jev itself; a fallback chain cannot
mask its errors. Reference agreement gates require full paired coverage in
addition to the threshold, so failed or missing answers cannot inflate the
score. The shown thresholds are examples and do not assert current agreement.
Run the generator from the service's original working directory: relative
secret paths follow that directory's configuration semantics.

Freeze targets from repeated Sol answers, retaining only cases whose choices,
depth, prepared states and policy are consistent across at least two trials:

```sh
python3 bench/teacher-cases.py --report /tmp/diagnostic-comparison.json \
  --teacher sol61 --out /tmp/sol-teacher-diagnostic \
  --split --families bench/diagnostic-families.json
env -u TS_ROUTER_CONFIG ai-router bench --config /path/to/tuned-config.yaml \
  --cases /tmp/sol-teacher-diagnostic/teacher-train.json \
  --target-decider jev --score initial --repeat 3 --format json \
  > /tmp/jev-training.json
env -u TS_ROUTER_CONFIG ai-router bench --config /path/to/tuned-config.yaml \
  --cases /tmp/sol-teacher-diagnostic/teacher-eval.json \
  --target-decider jev --score initial --repeat 3 --format json \
  --min-accuracy 0.95 --min-joint-accuracy 0.9 > /tmp/jev-evaluation.json
```

Tune criteria and examples using training discrepancies, then run the reserved
evaluation. Use a Jev-only tuned configuration for these frozen-label runs to
avoid paying for new teacher calls; keep Jev's before/after classification
budgets identical and record the production-budget check separately. Use
`routing-families.json` for `routing.yaml` reports. The manifests
explicitly group related variants to prevent them crossing the split. The
splitter deterministically assigns whole families and requires both training
and evaluation support for each group and frozen target model represented by
at least two families. It prefers the requested evaluation family fraction,
adjusting that fraction when coverage requires it. The provenance records
realized family/case fractions, per-group/model support and unsupported
singleton strata. A singleton family cannot provide both training and reserved
evaluation evidence; collect independent families for that stratum instead of
splitting its related variants. Conflicting coverage constraints fail clearly.

Inspect exclusions and split support before tuning. Stable teacher exclusions
can remove an entire tier or group, and a broad score does not establish coverage
of the missing tasks. Decide the split and seed once from frozen teacher labels;
changing seeds or inspecting reserved disagreements to select examples makes
those cases development data. Tuning uses only training prompts/discrepancies.
These are reserved development cases, not an independent unseen holdout.
Teacher exports never overwrite canonical labels.

`--score initial` compares the classifier's own choice and task depth with
frozen teacher labels. `--score final`, the default, measures the final route
after economy/pair handling. Both scores remain in the report. Tune raw
classification agreement separately from the routing heuristic.

## Priority replay and efficient comparison

Use `--no-chain` when comparing classifiers independently. It skips the extra
fallback-chain replay, which otherwise asks Jev again for every trial. Keep the
chain replay when testing fallback behavior. `--priority auto|quality|speed|cost`
and `--ranking model-a,model-b` apply the dashboard preferences to each decider
and record them in the report and policy fingerprint. Ranking stays separate
from capability order. These flags do not change the running service.

```sh
ai-router bench --config /tmp/comparison.json --cases bench/routing.yaml \
  --reference luna --target-decider jev --repeat 2 --no-chain --format json
```

The [2026-10-01 measurements](results/2026-10-01/README.md) include the live
Jev/Luna baseline, frozen family splits, tuning and reserved evaluation,
production-budget checks, and local prompt-preparation benchmarks.

Measured results and limitations for this operator policy are recorded in
[the 2026-10-01 evaluation](results/2026-10-01-operator-policy.md).
