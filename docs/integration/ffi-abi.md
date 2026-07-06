# FFI / C ABI reference

This is the cross-cutting reference for the C ABI exposed by the
`libsparrow_engine` cdylib. [`local.md`](local.md) covers how a native app
(e.g. Sparrow Studio Local) uses it; this page is the export inventory and the
ABI rules.

Source of truth: `sparrow-engine/sparrow-engine-cpu/src/ffi.rs` +
`sparrow-engine/sparrow-engine-cpu/exports.def` (the GPU flavor mirrors it
exactly). Both flavors export the **same 37 symbols**; a G5 acceptance gate
(`tests/integration_ffi_symbols.rs`) asserts the count and the CPU/GPU parity.

## Design principles

- **Opaque handles.** The engine handle is `pub type SparrowEngine = c_void;` —
  callers hold an opaque pointer, never a concrete struct. Result structs use
  the `*WithOwner` boxed pattern (a boxed owner holds the `Vec`/`CString`
  backing store; the `#[repr(C)]` header exposes raw pointers into it).
- **Allocator discipline: "X allocates / X frees / free exactly once."** Every
  function that returns a heap result has a matching `_free`. Calling the wrong
  `_free`, freeing twice, or freeing on the wrong thread is undefined behavior.
  All `_free` functions are null-safe.
- **`catch_unwind` in every export.** A Rust panic never crosses the ABI; it is
  caught and converted to an error return + a thread-local last-error string.
- **Errno-style errors.** On failure a function returns a null/negative
  sentinel; call `sparrow_engine_last_error` (thread-local) for the message.
- **ABI evolution via `_v2` suffix.** Structs carry **no reserved fields**; when
  a struct or signature must change incompatibly, a new `_v2` symbol is added
  alongside the old one (see `sparrow_engine_detect_audio` /
  `sparrow_engine_detect_audio_v2`). Old symbols are retained for ABI stability.

## Exported symbols (37)

### Engine lifecycle + diagnostics (4)
`sparrow_engine_engine_new`, `sparrow_engine_engine_free`,
`sparrow_engine_version`, `sparrow_engine_last_error`

### Model + pipeline management (7)
`sparrow_engine_load_model`, `sparrow_engine_load_model_by_id`,
`sparrow_engine_unload_model`, `sparrow_engine_load_pipeline`,
`sparrow_engine_load_pipeline_by_id`, `sparrow_engine_unload_pipeline`,
`sparrow_engine_list_models`

### Inference (9)
`sparrow_engine_detect`, `sparrow_engine_detect_raw`,
`sparrow_engine_detect_batch`, `sparrow_engine_classify`,
`sparrow_engine_embed`, `sparrow_engine_run_pipeline`,
`sparrow_engine_detect_audio`, `sparrow_engine_detect_audio_v2`,
`sparrow_engine_detect_audio_streaming`

### Result deallocators (9) — call exactly one, matching the producer
`sparrow_engine_detections_free`, `sparrow_engine_classify_result_free`,
`sparrow_engine_embedding_free`, `sparrow_engine_pipeline_result_free`,
`sparrow_engine_audio_result_free`, `sparrow_engine_audio_result_v2_free`,
`sparrow_engine_hash_result_free`, `sparrow_engine_verify_result_free`,
`sparrow_engine_free_string`

### Utility + introspection (8)
`sparrow_engine_health`, `sparrow_engine_hash_file`,
`sparrow_engine_day_night`, `sparrow_engine_image_brightness`,
`sparrow_engine_verify_model`, `sparrow_engine_engine_verify_model`,
`sparrow_engine_engine_model_info`, `sparrow_engine_engine_list_models_extended`

## Allocator pairing (quick reference)

| Producer | Matching free |
|---|---|
| `sparrow_engine_detect` / `_detect_raw` / `_detect_batch` | `sparrow_engine_detections_free` |
| `sparrow_engine_classify` | `sparrow_engine_classify_result_free` |
| `sparrow_engine_embed` | `sparrow_engine_embedding_free` |
| `sparrow_engine_run_pipeline` | `sparrow_engine_pipeline_result_free` |
| `sparrow_engine_detect_audio` | `sparrow_engine_audio_result_free` |
| `sparrow_engine_detect_audio_v2` / `_streaming` | `sparrow_engine_audio_result_v2_free` |
| `sparrow_engine_hash_file` | `sparrow_engine_hash_result_free` |
| `sparrow_engine_verify_model` / `_engine_verify_model` | `sparrow_engine_verify_result_free` |
| functions returning a `char*` | `sparrow_engine_free_string` |
| `sparrow_engine_engine_new` | `sparrow_engine_engine_free` |

## Stability

The 37-symbol set + signatures are a **stable contract**, evolved only by
adding `_v2` symbols. The G5 gate blocks any accidental drift in the export set.
Removing or re-signing an existing symbol is a breaking change and would be
tagged in the repo `CHANGELOG.md`.
