# Model layout, manifests, and the catalog

> **Status: stub.** This page will be expanded with the full manifest TOML
> schema reference. For now it points at the authoritative sources.

Sparrow Engine is **model-agnostic**: a model is onboarded by writing a TOML
manifest next to its ONNX file. The engine reads the manifest to drive all
pre- and post-processing; it never hard-codes model behavior.

## On-disk layout

Models live under a model directory (env var `SPARROW_ENGINE_MODEL_DIR`, e.g.
`/models` in the Docker image). Each model is a subdirectory:

```
$SPARROW_ENGINE_MODEL_DIR/
  <model-id>/
    manifest.toml       # how to pre/post-process + run this model
    model.onnx          # the ONNX graph (NMS in-graph for detectors)
    labels.txt          # (if applicable)
```

The user-supplied `?model=<id>` (HTTP) / `--model <id>` (CLI) is validated
(rejects `..`, `/`, `\`, absolute paths) before being joined to the model
directory.

## Invariants (see the design docs for rationale)

- **ONNX** for all models (vision + audio).
- **NCHW** layout mandatory (ORT CUDA EP has NHWC + dynamic-shape bugs).
- **Normalized bbox `[0,1]`** at all public API boundaries.
- **NMS in the ONNX graph, never in the engine** — validated at load time.
- Manifests are **TOML** (not YAML).

## Catalog + download

The model zoo is published to Zenodo (immutable, DOI-versioned). The single
source of truth for the zoo is `scripts/catalog.toml`; `scripts/download_models.sh`
reads it to fetch + checksum-verify models. See
[`../model-zoo-catalogue.md`](../model-zoo-catalogue.md) for the published model
list and licenses.

Until this page is filled in, the authoritative manifest schema is the
`ModelManifest` type in `sparrow-engine/sparrow-engine-types/src/manifest.rs`,
and example manifests ship next to every catalog model.
