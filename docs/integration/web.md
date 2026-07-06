# HTTP API integration (Sparrow Studio Web + server consumers)

Sparrow Engine's server (`sparrow-engine-server`) exposes a JSON HTTP API under
the `/v1/` prefix. This is how **Sparrow Studio Web** and any other
network/server consumer talk to the engine. Ship vehicle: the Docker image.

Source of truth for the routes: `sparrow-engine/sparrow-engine-server/src/router.rs`.

## Endpoints

### Inference

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/detect` | single-image detection |
| POST | `/v1/detect/batch` | batch detection |
| POST | `/v1/classify` | single-image classification |
| POST | `/v1/embed` | image embedding (one image) |
| POST | `/v1/embed/batch` | batch embedding (all-or-nothing) |
| POST | `/v1/pipeline` | detect → classify pipeline |
| POST | `/v1/audio/detect` | audio detection |

### Model + pipeline lifecycle

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/catalog` | list catalog (downloadable models) |
| GET | `/v1/models` | list loaded models |
| POST | `/v1/models/load` | load a model by id |
| DELETE | `/v1/models/{id}` | unload a model |
| POST | `/v1/models/{id}/trt-warmup` | build the TensorRT engine for a model (GPU) |
| GET | `/v1/pipelines` | list pipelines |
| POST | `/v1/pipelines` | create a pipeline |
| POST | `/v1/pipelines/load` | load a pipeline |
| DELETE | `/v1/pipelines/{id}` | delete a pipeline |

### Health

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | health check (touches the workload) |
| GET | `/healthz` | liveness |

## Bounding boxes and coordinates

Detection bounding boxes are **normalized to `[0,1]`** at the API boundary
(not pixel coordinates). This is a hard invariant across every surface.

## Inference-log storage (`?store=`)

Every inference endpoint accepts two per-request query parameters, both
default-false:

| Param | Effect |
|---|---|
| `store=true` | after a **successful** inference, emit an `InferenceLogRecord` to the configured sink |
| `halt_on_store_failure=true` | if the sink errors, fail the request |

Behavior matrix:

| `store` | `halt_on_store_failure` | Sink OK | Sink errors |
|---|---|---|---|
| false | — | 200, no record | 200, no record |
| true | false | 200 + record | **200** (sink error is warn-logged, request still succeeds) |
| true | true | 200 + record | **500** |

The record is emitted **after** the inference result is produced, so a storage
problem never corrupts the inference response unless you opt in with
`halt_on_store_failure=true`. Idempotency of stored records is the storage
layer's job (the Sparrow Data sibling), not the engine's. For the record schema
and how a data consumer ingests these, see [`data.md`](data.md).

## TensorRT warm-up (GPU)

On the GPU flavor, TensorRT builds a per-model engine on first use, which is
slow (minutes). `POST /v1/models/{id}/trt-warmup` triggers that build
explicitly (returns `202` and you poll `/v1/catalog` for the model's TRT state)
so the first real request isn't stuck behind a cold build. Manifests select the
mode via `[inference.trt].mode = off | on_demand | always`. Not applicable to
the CPU flavor (returns an error).

## Docker image

The server ships as a Docker image, published to Docker Hub:

| Flavor | Image |
|---|---|
| CPU | `docker.io/zhongqimiao/sparrow-engine-server` |
| GPU | `docker.io/zhongqimiao/sparrow-engine-server-gpu` |

Tags: `:vX.Y.Z`, `:latest`, `:sparrow-combined`.

```bash
docker pull docker.io/zhongqimiao/sparrow-engine-server:latest        # CPU
docker pull docker.io/zhongqimiao/sparrow-engine-server-gpu:latest    # GPU
```

Runtime configuration (environment variables):

| Env var | Default | Purpose |
|---|---|---|
| `SPARROW_ENGINE_MODEL_DIR` | `/models` | where per-model dirs live (mount a volume here) |
| `SPARROW_ENGINE_BIND_ADDR` | `0.0.0.0:8080` | listen address |
| `SPARROW_ENGINE_LOG_FORMAT` | `json` | `json` (machine-parseable) or pretty |

The image does **not** bake in models — mount your model directory at
`SPARROW_ENGINE_MODEL_DIR`. See [`model-layout.md`](model-layout.md) for the
on-disk layout and catalog download. The image `HEALTHCHECK` touches the actual
workload, so it maps directly to a Kubernetes readiness probe.

GPU image only: the host must provide the NVIDIA driver + CUDA + **cuDNN ≥ 9.10**
(the image base is a `cudnn-runtime` CUDA image; use the NVIDIA Container
Toolkit for GPU passthrough).

## Stability

The HTTP endpoint set and request/response shapes are a **stable contract**.
Additive changes are backward-compatible; breaking changes are tagged in the
repo `CHANGELOG.md` with a consumer-impact note.
