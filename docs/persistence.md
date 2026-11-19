# TerraBase persistence preparation (TS-511)

Status: migration preparation only. Production routing still uses the legacy
SQLite store. Runtime persistence and production cutover wait for COS's
TS-589 `RELEASED` digest and the released JWT-capable ts-go SDK.

## Target contract

Follow the TerraBase TS-589 route contract.
The router owns one database in a ts-auth project. Services call TerraBase
only through ts-go, using a ts-auth client credential exchanged for a
short-lived project JWT. No `tb_` credential, custom HTTP adapter, management
service credential, or organization-scoped route is an alternative.

The SDK renews tokens. Normal routing needs `documents:read` and
`documents:write`; database provisioning belongs to an operator with
`databases:write`. Paths use the project UUID and database **name**:

```text
/v1/projects/{project}/databases/{database}/collections/{collection}/documents/{id}
/v1/projects/{project}/databases/{database}/transactions
```

Keep one collection per entity and one document per record:

| Collection | Natural key | State preserved |
| --- | --- | --- |
| `pins` | Thread key | Model, decided model/level, failover, capability, parent and activity |
| `events` | Decimal event ID | Decision/change history, attempts, prompts and policy basis |
| `requests` | Decimal request ID | Tokens, nullable cost, latency/queue wait and metering repair originals |
| `policies` | Fingerprint | Recorded routing policy |
| `reviews` | Decimal event ID | Operator judgement and note |
| `routing_settings` | `1` | Runtime mode, model group and preferences |
| `thread_image_requirements` | Thread key | Images observed, including fixed-mode threads without a pin |

Use the SDK's `DocumentIDFor(collection, key)` to map natural keys to UUIDs.
Keep numeric IDs in document bodies: UUID ordering cannot replace event or
request order. Preserve nullable costs separately from a priced zero.
FTS tables are derived from event prompts and are not application records.

## Prepare an export

```sh
ai-router export-state --db /var/lib/ai-router/router.db --to /tmp/router-state.ndjson
```

The command needs no configuration or service credentials. It opens an
existing database read-only, requires this build's schema version, and
streams every application row from one transaction. It never migrates the
source. Older or newer schemas fail unchanged; use the matching released
router build for an older source before preparing a migration separately.

The output is a new file with mode `0600`; existing paths and the source's
SQLite sidecar paths (`-wal`, `-shm`, `-journal`) are rejected, including aliases.
It contains recorded prompts and decider replies: keep it private. On an
export/close failure the command removes its incomplete output.

Each JSON line is `{"collection":"pins","key":"…","document":{…}}`.
Document fields retain the SQLite column names. Integers and millisecond
timestamps stay exact, JSON-text columns become structured JSON, and flag
columns become booleans. The final line is a `manifest` containing format
version `1`, the source schema version, and counts for all seven collections,
including empty ones. An importer must require that final line, verify the
counts, and reject trailing records or unsupported versions. The artifact is
temporary migration input, not a production persistence backend.

## Finish after TS-589 is released

1. Pin the released ts-go SDK and implement the existing store operations
   against the document collections. Pin creation/change and its event must
   commit atomically; use version conditions and reconcile uncertain commits
   by stable operation identity before retrying.
2. Persist the metering lifecycle in TerraBase before forwarding upstream.
   Finalize tokens/cost when streaming ends and expose incomplete records.
   The current memory queue can drop writes and cannot be the sole record.
   TerraBase database usage counts reads/writes/storage; model token costs
   remain router-owned request documents.
3. Verify restart durability, concurrent pin changes, failed/uncertain writes,
   prompt search and aggregation, JWT renewal, and project/scope rejection
   through the released SDK. Production serve must have no SQLite fallback.
4. Drain and stop router writes for a final export. Provision the dedicated
   database through public SDKs, import with deterministic IDs, check counts
   and references, and initialize event/request sequences above imported IDs.
   A preliminary live export is consistent but omits later updates/deletions.
5. Cut over only after the release digest and production verification. Keep
   the source snapshot for recovery; do not delete it or re-enable SQLite
   automatically when TerraBase is unavailable.
