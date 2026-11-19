# Dashboard

The operator dashboard at `/dashboard`: what it shows, the Routing page
controls, the review queue, the request log, sign-in and how the pages are
built. Routing itself is described in [routing.md](routing.md); sign-in
secrets and deploys are covered in [operations.md](operations.md). Back to the
[README](../README.md).

## Dashboard

`/dashboard` shows the last day at a glance (headline figures, requests per
hour stacked by model, the model mix over a day or a week, CLIProxy's catalog,
deciders with their health and the reasoning mix, and a live request feed),
every pinned thread with a model filter, a thread's decision (why the model
and level were chosen, each candidate's and level's probability and each
step's cost-weighted policy score, with the time and the learned miss rate
it counted,
every decider's attempt and the exact request and reply of each, flagged
when today's request format no longer matches the digest of what was sent),
what similar past prompts suggested and how much it weighed, the first
prompt as the user wrote it, a review of the decision, pin
history with its redecisions, links between a session's thread and its
subagents' threads, a re-pin form that moves the thread to another model and level,
and its requests, and a live, filterable request log that can be paused.
The Decisions tab shows how `auto` classifies: the last 30 days of
decisions per model with the latest prompts each got, the classifier's
input (decider chain, instructions, the candidate descriptions Jev weighs as
criteria, the reasoning levels and the cost per task of each model at each
level, with the time a task takes there and how often its threads moved up),
a box that decides any prompt without pinning it and shows what
each decider was sent and answered, how the reviewed decisions fared, what
routing learned from its history (the labels settled threads left, by kind,
and the weight similar prompts earned for the model and the reasoning), the
decisions likely misrouted, and every decision with its review, filterable by model, decider, review and prompt
text. Decisions and Logs both offer indexed full-text search: all entered
words must match the recorded first prompt or classifier input, ignoring case,
accents and punctuation. Logs selects requests belonging to matching threads;
it does not record or search later conversation messages. Search combines
with existing filters and survives page reloads. Search is limited to 256
characters, and prompts removed by retention are no longer searchable.

Logs also has a collapsed **Delete request logs** form for requests older than
24 hours, 7 days, 30 days, or all requests before the deletion starts. An explicit
confirmation is required. Deletion applies across all models, regardless of
search and filters, and preserves pins, decisions and reviews. Request usage,
cost totals and learned estimates lose the deleted records. Open live request
tables refresh automatically after an operator deletion.

Each distinct policy is
recorded once by its fingerprint, so a decision's decider requests are
rebuilt from its recorded prompt, and checked against the digest of the
bytes sent.

The current Routing page persists mode/group/priority/ranking overrides and a
`jev_enabled` switch. These controls conflict with the required fixed policy and
always-on features; runtime removal is pending. This document does not change
the saved fields or certify their removal.

Automatic discovery must preserve candidate capability checks, availability,
thread pins and failover safety. Concrete upstream requests retain passthrough
behavior. A dashboard session is still required for protected operations;
unknown or retired candidates must be rejected. This docs change does not
remove the current form fields or claim the policy is already enforced.

Operators sign in with a username and password. `dashboard.username`
defaults to `mariogk`; `dashboard.password_hash_env` defaults to
`TS_ROUTER_DASHBOARD_PASSWORD_HASH` and names the secret holding the bcrypt
password hash, resolved from `secrets_files` or the environment. A missing
hash disables dashboard sign-in; a malformed nonempty hash is a
configuration error. Dashboard sign-in accepts no API key. Claude Code,
model discovery and `/stats` continue to use the CLIProxy client key.

OCI Vault must provision and deliver operator authentication material
automatically. The current bcrypt hash/environment/`secrets_files` consumer is
a delivery gap; manual hash-generation and credential-copy recipes are retired.
Missing or malformed authentication material must fail closed. Dashboard sign-in
continues to accept no API key; model/API requests use their separate client
credential. Session, same-origin and confirmed log-deletion checks remain binding.

An address gets 429 after 5 failed sign-ins in 10 minutes. The session is
an in-memory, HttpOnly, SameSite=Strict cookie that lasts 12 hours, so a
restart signs everyone out. Signing out or reaching the session expiry
also closes active dashboard streams. Cross-origin
POSTs are refused, and the trial box runs at most 4 trials at once.

Pages render on the server with [templ](https://templ.guide) and stay live
through [Datastar](https://data-star.dev) server-sent events; there is no other
JavaScript and nothing loads from another origin. The generated `*_templ.go`
files, the built Tailwind stylesheet and the vendored Datastar bundle are
committed and embedded in the binary. After changing a template or the
stylesheet source, regenerate them:

```bash
mise exec -- make generate                 # templ v0.3.1020 -> *_templ.go
mise exec -- make css                      # Tailwind v4.3.3 standalone CLI (downloaded to tools/)
mise exec -- make datastar                 # re-vendor Datastar v1.0.4
```

## Review queue

Above every decision, **Likely misrouted** lists the unreviewed decisions
whose settled threads went worst, the strongest sign first, each sign in
words and each linking to the thread's review: an operator moved the thread
up (strongest) or down, the router decided it again up or down, a thread on
the lighter half of the ladder ran more than `outcomes.long` regular turns
(200), it failed over on a server error, a thread on the heavier half stayed
within `outcomes.tiny` turns (3), or it started carrying images its model could
not read. Turns are regular requests: token counts, compactions and
background calls do not count, and a thread's size counts only while it
stayed on the model it was decided for. A failover on a 429 and a migration say
nothing about the decision and never queue one. `/stats` counts the settled threads showing each sign.

## Request log

The request log keeps, per `auto` request, the pin key hash (side requests
too, so a thread's compactions count towards its cost), endpoint kind, sent
model and effort, reason, status, failover, latency, queue wait, tokens, cost and a
short error, and reads each request back with its thread's cost so far. Latency runs from the
request's arrival until the response finished streaming, so it includes the
queue wait (time spent waiting for a `limits` slot, logged apart) and the
routing decision. Requests for an explicit model pass through unlogged, so the log and
the dashboard show only `auto` traffic. The request log is written off the
request path (a full queue drops entries rather than delaying a response)
and pruned hourly to `log_retention` (30 days by default); pins and
decisions are kept.
