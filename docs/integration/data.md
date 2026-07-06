# Data integration — Sparrow Data

**Sparrow Data** is the sibling that ingests, indexes, and enables retrieval
over inference results. It does not link the engine directly; it consumes two
things the engine produces: **inference-log records** (for provenance +
searchable results) and **image embeddings** (for similarity retrieval). This
page is the contract for both.

The engine emits these; it never interprets or stores them itself. Storage,
indexing, deduplication, and retrieval are Sparrow Data's responsibility.

## Inference-log records

### How you get them

Set `?store=true` on any inference HTTP endpoint (see [`web.md`](web.md)). After
a successful inference, the engine emits one `InferenceLogRecord` to the
configured **sink**. The default sink (`StderrJsonLinesSink`) writes one JSON
object per line to stderr, under a stderr lock (JSON Lines / ndjson). A
deployment can swap in another sink implementation.

- `store=true, halt_on_store_failure=false` → the request returns 200 even if
  the sink errors (the error is warn-logged). Inference is never blocked by a
  storage problem.
- `store=true, halt_on_store_failure=true` → a sink error fails the request
  (500).

**Idempotency is your job.** The engine may emit the same logical inference more
than once (retries, replays); Sparrow Data's storage layer must dedupe (e.g. on
`request_id` + `media_hash`). The engine does not dedupe.

### Record schema (`SCHEMA_VERSION = "1.0"`)

Source of truth: `sparrow-engine/sparrow-engine-types/src/inference_log.rs`.

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `String` | currently `"1.0"` |
| `request_id` | `String` | correlation id |
| `timestamp_utc` | `String` | ISO-8601 UTC |
| `media_hash` | `String` | content hash of the input media |
| `model_id` | `String` | which model produced the result |
| `model_version` | `String?` | optional |
| `device` | `String` | e.g. `cpu`, `cuda:0` |
| `inference_ms` | `f64` | inference wall time |
| `result` | JSON value | the inference payload (detections / classes / embedding / …) |
| `provenance` | object? | optional training-provenance record (round-tripped from the model manifest) |
| `drift_metrics` | object? | optional per-request stateless drift metrics (Tier-1/2) |

### Schema versioning rule

- **Additive** optional-field changes keep `"1.0"` — your ingester must ignore
  unknown fields, not reject them.
- A **rename, type change, or semantic shift** bumps to `"2.0"` and is
  coordinated with a corresponding Sparrow Data ingester change. Gate your
  ingester on `schema_version` major.

### Drift split (context)

Per-request, stateless drift metrics (Tier-1/2) are computed in-engine and ride
along in `drift_metrics`. Stateful drift (reference distributions, per-camera
CUSUM, alarm paths — Tier-3) is **not** in the engine; that lives in the
`sparrow-ops` sibling. Sparrow Data ingests the Tier-1/2 numbers as data.

## Image embeddings (similarity retrieval)

The engine emits embeddings only; the vector index + nearest-neighbor search
live in Sparrow Data. Get embeddings via `POST /v1/embed` / `/v1/embed/batch`
(HTTP), `spe embed` (CLI), `Engine.embed(...)` (Python), or
`sparrow_engine_embed` (FFI).

`EmbedResult` fields (source: `sparrow-engine-types/src/types.rs`):

| Field | Meaning |
|---|---|
| `embedding` | the float vector |
| `dim` | vector dimensionality |
| `normalized` | whether the vector is L2-normalized |
| `metric` | intended similarity metric (e.g. cosine) |
| `model_id` | which encoder produced it |
| `embedding_version` | **index-compatibility key** — see below |
| `model_hash` | ONNX `sha256`, load-verified against the manifest |
| `image_width`, `image_height` | source image dims |
| `processing_time_ms` | embed wall time |

### Index-compatibility contract

`embedding_version` + `model_hash` are the versioned contract for a retrieval
index. Two embeddings are only comparable if their `embedding_version` (and
effectively `model_id` / `model_hash`) match. When onboarding a new encoder or
re-exporting one, the `embedding_version` changes and Sparrow Data must treat
the old and new vectors as **separate index spaces** (do not mix them in one
nearest-neighbor space). `model_hash` is verified at model load against the
manifest's `onnx_sha256`, so a silently-swapped model is caught.

## Sink implementation notes (for a custom sink)

The sink trait is `InferenceLogSink` with a **synchronous** `fn emit(...)`
(`sparrow-engine-server/src/sink.rs`). The default is `StderrJsonLinesSink`. A
future async or network sink must not block the async runtime — it should
internally `tokio::task::spawn_blocking` (or the trait upgrades to `async fn`).
