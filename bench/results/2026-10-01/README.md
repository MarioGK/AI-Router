# Jev routing and performance measurements — 2026-10-01

Jev's initial model accuracy on the 72-case routing corpus rose from 95.8% to
97.9% after clarifying task boundaries. On 17 reserved prompts, grouped by task family,
initial agreement with stable, frozen Luna model targets rose from 94.1% to
98.0%. Final-route agreement on that subset stayed at 94.1%. Jev remained much
faster than Luna as a classifier. These are routing-policy measurements, not
measurements of downstream task success.

The runs used Alpha's existing CLIProxy and Jev credentials through isolated
benchmark binaries. They did not change the running service's binary, database,
settings, configuration or account priorities. The baseline used Jev 1.13.0 and
GPT-6 Luna; policies, request digests, raw replies, usage coverage and run
settings are in the archived reports. No credential values or deployment
configuration files are included.

## Full routing corpus

Each row covers the same 72 prompts, repeated twice, with six concurrent
classifier calls. Before and after use identical 60-second decider budgets
and a 120-second decision budget, the configured cost prior, and pair doubt
handling. Labels are the corpus's original policy labels. The Luna row is the
original baseline, retained rather than rejudged under tuned criteria.

| Classifier | Initial model correct | Final route correct | Depth correct | Median | p95 | Pair calls | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jev before | 138/144 (95.8%) | 139/144 (96.5%) | 140/144 (97.2%) | 294 ms | 608 ms | 11 | 0 |
| Luna baseline | 142/144 (98.6%) | 142/144 (98.6%) | 135/144 (93.8%) | 4,753 ms | 8,105 ms | 0 | 0 |
| Jev after | 141/144 (97.9%) | 140/144 (97.2%) | 139/144 (96.5%) | 292 ms | 433 ms | 3 | 0 |

Before tuning, paired Jev/Luna agreement was 136/144 (94.4%) for initial model,
130/144 (90.3%) for depth, 124/144 (86.1%) for both, and 141/144 (97.9%) for final
route. Coverage was complete. The later frozen-target evaluation below keeps
Luna's target fixed while Jev's criteria change.

The small accuracy difference on the full corpus is development evidence.
Repeats measure stability, not independent tasks, and these counts do not
establish statistical significance or a task-quality advantage over Luna.

## Frozen targets and family split

The baseline's two Luna trials yielded 68 stable model/depth targets; four
unstable cases were excluded. `routing-families.json` and the splitter's default
seed produced 51 training prompts and 17 reserved prompts, keeping related
variants together. Only training disagreements informed changes: recursive
parsers and keyset pagination require the bounded algorithm tier; unknown
lock/transaction/cancellation deadlocks require the difficult tier; known
stream cleanup and transaction-preserving migrations remain bounded; verifying
a supplied exact correction remains a tiny task. Canonical labels were kept.

| Split and run | Trials | Initial model agreement | Final-route agreement | Depth agreement | Initial model + depth |
| --- | --- | --- | --- | --- | --- |
| Training baseline | 102 | 96/102 (94.1%) | 101/102 (99.0%) | 100/102 (98.0%) | 94/102 (92.2%) |
| Training after | 153 | 152/153 (99.3%) | 153/153 (100%) | 150/153 (98.0%) | 149/153 (97.4%) |
| Reserved baseline | 34 | 32/34 (94.1%) | 32/34 (94.1%) | 26/34 (76.5%) | 26/34 (76.5%) |
| Reserved after | 51 | 50/51 (98.0%) | 48/51 (94.1%) | 42/51 (82.4%) | 41/51 (80.4%) |

The after runs used three repeats rather than two; support is shown explicitly.
Depth agreement remains weaker than model agreement. Luna sometimes chose a
higher depth than the original labels and Jev; its answers are reference
opinions, not unquestioned ground truth. The reserved prompts are development
cases from the existing corpus, not an independent, unseen production holdout.
No reserved disagreements were used to select or revise criteria.

## Production budget and local work

With the configured five-second Jev timeout and 30-second total decision budget,
a separate 72-case run had zero failures, 70/72 (97.2%) initial and final model
accuracy, 307 ms median latency and 463 ms p95. Comparison-budget results alone
were not used to justify shortening production timeouts.

Local Go benchmarks used Go 1.27.1 on the same Xeon E5-2660 v4 machine, three
runs per case. Prompt preparation now skips wrapper parsing when there are no
opening tags, reuses rune counts, and slices retained Unicode excerpts without
allocating an entire prompt as runes. Exact excerpt behavior, including malformed
UTF-8, is covered by regression tests.

| Prompt case | Median before | Median after | Speedup | Typical allocated bytes before → after |
| --- | --- | --- | --- | --- |
| Long unmatched backticks | 3.26 ms | 0.196 ms | 16.6× | 1,142,821 → 106,929 |
| Distinct backtick runs | 1.43 ms | 0.101 ms | 14.1× | 733,457 → 432 |
| Escaped quotes | 1.98 ms | 0.110 ms | 18.0× | 747,316 → 432 |
| Wrapper with tildes | 0.932 ms | 0.934 ms | approximately unchanged | 173 → 136 |

These are local prompt-preparation costs on deliberately long inputs, not
end-to-end generation latency. Routing settings now read through SQLite's
read pool instead of contending for its serialized writer. `--no-chain` avoids
replaying the fallback chain during independent classifier comparisons; frozen
teacher labels also avoid paying for fresh Luna calls during tuning.

## Evidence and replay

`summary.json` contains all run settings and aggregate metrics, including
subgroup support, stability, usage coverage and baseline frozen-target scores.
`before.json.gz`, `routing-after.json.gz`, `training.json.gz`, `reserved.json.gz`
and `production.json.gz` contain the complete versioned benchmark reports.
`teacher-train.json`, `teacher-eval.json` and `teacher-provenance.json` preserve
frozen labels, split membership and exclusions. The two `state-*.txt` files
contain every local benchmark sample. Archive checksums are in `summary.json`.

```sh
gzip -dc bench/results/2026-10-01/before.json.gz > /tmp/router-before.json
python3 bench/teacher-cases.py --report /tmp/router-before.json --teacher luna \
  --out /tmp/router-luna-frozen --split --families bench/routing-families.json
ai-router bench --config /path/to/jev-only-config.json \
  --cases bench/results/2026-10-01/teacher-eval.json --target-decider jev \
  --score initial --repeat 3 --parallel 6 --no-chain --format json
go test ./internal/decide -run '^$' -bench BenchmarkStateUnmatchedDelimiters \
  -benchmem -count 3
```

A fresh evaluation after inspecting or tuning against these saved reserved
results requires new task families. Final validation passed `make verify`
(formatting, vet and the race-tested suite), all 15 Python benchmark-tool tests,
dashboard template/CSS generation, and the static router build.
