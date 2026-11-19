# Operator policy evaluation — 2026-10-01

## Five-model policy with GLM 5.3 Flash

The final policy adds native-vision GLM Flash for small coding and visual work,
retains GLM 5.3 for broader routine text-only work, and favors Sol 6.1 for
substantive coordinated or urgent implementation. Sixteen development cases
were labelled before evaluation, including two image-routing cases and two
broader maintenance cases. Calls ran from Alpha, four concurrent tasks, one
trial per case, with 60 s classifier timeouts and a 120 s decision budget.
The benchmark binary was source `06a4273`; its classification implementation
is unchanged in the Flash update. The reviewed five-model policy SHA-256 is
`a04226ea705a0a6d79f02d4d161566acfd9dd5c1c8448f96340f3f8eda9eb1d2`.

| Classifier | Correct model | Model + accepted depth | Median | p95 | Failed decisions |
| --- | --- | --- | --- | --- | --- |
| jev | 16/16 | 15/16 | 288 ms | 506 ms | 0 |
| luna | 16/16 | 15/16 | 5226 ms | 11981 ms | 0 |

Initial and final model agreement was 16/16, with full comparison coverage.
Exact task-depth agreement was 15/16: the Portuguese notification feature was
medium for Jev and high for Luna, both allowed by its labels and both applied as
Sol high. Both missed the module-test depth label (low rather than medium);
that remains a failure. Jev's median classification time was about 18 times
lower in this sample. These are development-policy checks, not an unseen
holdout, proof of equality with Luna, or downstream model-quality measurements.
The image cases test selection from image capability metadata; they do not
send an image to the selected completion model.

Official [Flash capabilities](https://docs.z.ai/guides/vlm/glm-5.3-flash),
[thinking](https://docs.z.ai/guides/capabilities/thinking) and
[prices](https://docs.z.ai/guides/overview/pricing) document native multimodal
input, tool calling, structured output, mandatory thinking, roughly 1M context
and 128K output. Coding Plan maps low to low, medium/high to high and xhigh/max
to max. Its included usage has zero marginal routing cost on this plan;
published API prices ($0.15 input / $0.50 output / $0.03 cache read per million
tokens) remain distinct. FlashX's 200 tokens/s claim belongs to another model.

Direct live probes attempted three synthetic tasks each on Flash, main GLM
and Sol, plus a generated red-square image, a tool call and two effort levels
on Flash. Every GLM call returned HTTP 429; an inspected upstream reply
identified provider code 1302 (concurrency), not exhausted plan credits.
Therefore there is no successful live GLM quality, vision, tool or speed
coverage from this run. Sol answered the three synthetic tasks correctly,
median 4546 ms; this tiny sample is not a coding-quality benchmark. Error
latencies are not successful inference speed and are not used as routing
latency estimates. Flash starts with a conservative cap of 3, main GLM keeps
6, and a recent throttle spills new threads to Sol for one recovery interval,
including saturation caused by clients outside this router. Existing pins
retain their bounded retry queue. Race tests cover this recovery behavior.

Evidence digests:

- Five-model comparison: `921c89fd44c8bff9d943de31020ad33209c0cbe912788a093663199918db6954`
- Direct capability probes: `804299721577f1bd1f508ae3a972a4998d55c48781b5a6ce83382085f7029036`

## Prior four-model revision

The results below apply to source `06a4273` before GLM Flash joined the
foreground ladder. The larger regression corpora have not been relabelled or
rerun as five-model quality evidence.

Live classification calls from Alpha, four concurrent benchmark tasks, one trial
per case. The operator-boundary comparison used 60 s per classifier and a 120 s
decision budget; regression corpora used the production timeout configuration.
Task costs used cold typical-token priors, GLM cost factor 0, and the configured
list prices. This is development-policy compliance, not an unseen holdout or a
measurement of downstream completion quality, cost savings or model speed.

| Corpus | Classifier | Correct model | Model + accepted depth | Median | p95 | Failed decisions |
| --- | --- | --- | --- | --- | --- | --- |
| Operator boundaries | jev | 12/12 | 12/12 | 284 ms | 586 ms | 0 |
| Operator boundaries | luna | 12/12 | 11/12 | 5984 ms | 12054 ms | 0 |
| Routing regression | jev | 68/72 | 66/72 | 283 ms | 391 ms | 0 |
| Routing regression | chain(pair) | 68/72 | 66/72 | 279 ms | 434 ms | 0 |
| Diagnostic regression | jev | 61/64 | 61/64 | 292 ms | 419 ms | 0 |
| Diagnostic regression | chain(pair) | 61/64 | 61/64 | 290 ms | 548 ms | 0 |

Jev and Luna agreed on the initial and final model for all 12 operator-boundary
cases. Their exact task-depth labels agreed on 10/12; the two substantive feature
cases allow either medium or high. Luna missed the depth label on one exact-edit
case. These samples do not establish general equality with Luna. Jev used one
call per operator case and had no failures.

The production chain passed the 90% model-accuracy regression target on both
larger corpora, with no failed decisions. Standalone DeepSeek fallback had one
failure on routing and two on diagnostics; the Jev-first chain stayed available.
Existing resolver tests cover GLM saturation/spill and availability; classifier
benchmarks do not exercise live concurrency or provider credit exhaustion.

Known chain misses: routing selected Sol for the routine endpoint, payment-retry
and local Compose cases, and Opus for localized archive containment. Diagnostics
selected Opus for a forged-policy exact edit, Sol for the column-picker task,
and GLM for a recurring-time/DST bug. These remain failures against the labels,
not retroactively relabelled cases. The full-stack favorites label was explicitly
changed to Sol before this run to reflect the new operator policy.

Prompt-search microbenchmark: 10,000 indexed decisions, 100 matches, 25 rendered
records per query; 1.98 ms average per storage query (613 iterations, four Go
processors on Xeon E5-2660 v4). This measures storage querying, not browser or
network latency. Dashboard Chromium checks passed at 1440 px and 390 px for
mode switching, tooltips, search/clear, saved preferences, full-auto reset and
confirmed deletion in a disposable local database.

Evidence report digests (raw reports retained with the task artifacts):

- Operator boundaries: `5c5f9cb433d0b0f7168ad8c18cb0da151090e320d79be698517e65615f1178cd`
- Routing regression: `69b2da4c96c40b49a461839af02434935e5a57aa91f17d078333e01cb35a011c`
- Diagnostic regression: `93226e83adf06c24aa10a278d0c3f49a7c96d1e7b6e070220b78ba589c5cc202`
