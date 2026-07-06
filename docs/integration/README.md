# Consuming Sparrow Engine

Sparrow Engine is an ONNX inference engine for camera-trap wildlife detection,
classification, and image embedding. It is **engine-only**: it loads models and
runs inference. Annotation, data versioning, model training, retrieval, and
deployment orchestration live in separate repositories that *consume* this
engine through one of the surfaces below.

This directory is the single entry point for anyone integrating against Sparrow
Engine. Pick the surface that matches how your code talks to the engine, then
read the linked page.

## Which surface do I use?

| Consumer | Talks to the engine via | Page | Ship vehicle |
|---|---|---|---|
| Server / web app (e.g. Sparrow Studio Web) | HTTP JSON over the network | [`web.md`](web.md) | Docker image |
| Native desktop app (e.g. Sparrow Studio Local) | C ABI (cdylib) + P/Invoke | [`local.md`](local.md) | `libsparrow_engine.so` / `.dll` / `.dylib` |
| Python program / notebook | `import sparrow_engine` (PyO3) | [`python.md`](python.md) | PyPI wheel |
| Shell script / batch job | `spe` command-line binary | [`cli.md`](cli.md) | CLI binary |
| Data / retrieval sibling (Sparrow Data) | inference-log records + embeddings | [`data.md`](data.md) | (any of the above) |

Cross-cutting references:

- [`ffi-abi.md`](ffi-abi.md) — the C ABI export inventory + ABI-versioning rule (used by `local.md`).
- [`model-layout.md`](model-layout.md) — manifest TOML + on-disk model layout + catalog download.

For end-user install and usage (as opposed to *integration*), see the top-level
[`../user-manual.md`](../user-manual.md) and [`../../installer/`](../../installer).

## CPU and GPU flavors

Every surface ships in two mutually-exclusive flavors. They expose the **same
API**; they differ only in the execution backend and their runtime dependency.

| | CPU flavor | GPU flavor |
|---|---|---|
| Docker image | `docker.io/zhongqimiao/sparrow-engine-server` | `docker.io/zhongqimiao/sparrow-engine-server-gpu` |
| Python wheel | `sparrow-engine` (dep: `onnxruntime`) | `sparrow-engine-gpu` (dep: `onnxruntime-gpu`) |
| Python import | `import sparrow_engine` | `import sparrow_engine` (same) |
| CLI binary | `spe` | `spe-gpu` |
| cdylib file name | `libsparrow_engine.so` | `libsparrow_engine.so` (same) |
| Extra runtime dep | none beyond ONNX Runtime | NVIDIA driver + CUDA + **cuDNN ≥ 9.10** |

The two cdylibs and the two wheels use the **same file/import name on purpose**
so a consumer can swap flavors without changing its load path. They are never
co-located in the same directory. See [`local.md`](local.md) and
[`../user-manual.md`](../user-manual.md) for the packaging rules.

> **cuDNN note (GPU only):** the ONNX Runtime CUDA execution provider needs
> cuDNN ≥ 9.10. cuDNN 9.8 has a Conv-engine asymmetric-padding bug on `sm_89`
> that breaks some models. The GPU Docker image is built on a
> `cudnn-runtime` CUDA base to satisfy this; native GPU consumers must provide
> it themselves.

## Stability and versioning

Three surfaces are treated as stable contracts; the internal Rust APIs are not.

| Surface | Stability | Where the contract lives |
|---|---|---|
| HTTP API endpoints | stable | [`web.md`](web.md) |
| C ABI (FFI exports) | stable, `_v2`-evolved | [`ffi-abi.md`](ffi-abi.md) |
| Manifest TOML schema | stable, additive | [`model-layout.md`](model-layout.md) |
| Inference-log record schema | versioned (`SCHEMA_VERSION`) | [`data.md`](data.md) |
| Internal Rust crate APIs | **unstable** | n/a (do not depend on these) |

Breaking changes to a stable surface are called out in the repo `CHANGELOG.md`
with a consumer-impact tag. See the release-prep versioning policy for the full
deprecation-window rule.
