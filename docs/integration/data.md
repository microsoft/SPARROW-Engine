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
than once (retries, replays); Sparrow Data's storage layer must dedupe on
`media_hash` + `model_id` — the canonical UNIQUE key (see `inference_log.rs`).
Do not key on `request_id`: it is a per-request UUID (v4) that changes on every
retry, so it would defeat dedup. The engine does not dedupe.

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
(HTTP), `spe embed` (CLI), `sparrow_engine.embed(...)` (Python), or
`sparrow_engine_embed` (FFI).

**Python facade (`sparrow_engine`).** `sparrow_engine.embed()` and
`embed_with_meta()` are order-preserving and fail-closed: results come back in
caller order, and if any requested file fails they raise
`EmbedPartialFailureError` (partial batch) or `EmbedAllFailedError` (all files)
rather than silently returning a shortened list. For safe positional matching
against the inputs, use the aligned facade variants (`sparrow_engine.embed_aligned*`):
they return exactly one slot per input in caller order, `None` for each failed
file, and raise `EmbedAllFailedError` only when every input fails. The `_with_meta`
form yields `list[Optional[EmbedResult]]`; the bare form yields a per-input list of
`Optional[np.ndarray]` (a list, not a stacked matrix — `None` slots cannot stack).

**Native `PyEngine` (direct batch integration).**
`PyEngine.embed_aligned(paths, model)` returns exactly one slot per input, in
input order: an `EmbedResult` on success and `None` when that file could not be
embedded. If every input fails, it raises `EmbedAllFailedError`.

`PyEngine.embed(paths, model)` keeps its successful return type
(`list[EmbedResult]`) but raises `EmbedPartialFailureError` when only part of a
batch succeeds. It never returns a shortened list that callers could zip onto
the original paths. This fail-closed behavior prevents a corrupt image from
shifting every later vector onto the wrong media record.

`EmbedResult` fields — the Rust core struct (`sparrow-engine-types/src/types.rs`),
the engine's internal representation. This struct is not `Serialize`, so it is not
the on-the-wire shape; the representation you receive depends on the surface (see
"Representation by surface" below).

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

**Representation by surface.** The field names above are the Rust core struct.
Three surfaces differ from it:

- **Python object** — returned by `sparrow_engine.embed_with_meta(...)` and
  `PyEngine.embed` / `embed_aligned`; typed in `_core.pyi`. The vector is exposed
  as attribute **`vector`** (a NumPy array), not `embedding`, and the object also
  carries **`embed_schema_version`** (currently `"1.0"`). All other fields match by
  name.
- **HTTP `/v1/embed[/batch]` response** — the per-item vector field is `embedding`,
  image dims are `image_size: [width, height]`, and each response carries
  `embed_schema_version`.
- **Inference-log `result` payload** (`store=true`;
  `sparrow-engine-server/src/handlers/embed.rs` `embedding_log_payload`) —
  deliberately **omits the raw vector** and **includes** `embed_schema_version`
  plus the identity fields. A stored record is provenance, not the vector itself;
  fetch vectors from the embed surfaces above.

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
