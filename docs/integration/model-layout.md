# Model layout, manifests, and the catalog

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
- **NMS has two load-validated lanes**: `yolo_e2e`/`yolo_nms` detectors carry
  NMS in the ONNX graph; declared raw-head detectors use the shared engine-side
  `megadet_v5a` (or `retinanet_soft_nms`) postprocessor. Both emit normalized
  `[0, 1]` boxes at the public boundary.
- Manifests are **TOML** (not YAML).

## Catalog + download

The model zoo is published to Zenodo (immutable, DOI-versioned). The single
source of truth for the zoo is `scripts/catalog.toml`; `scripts/download_models.sh`
reads it to fetch + checksum-verify models. See
[`../model-zoo-catalogue.md`](../model-zoo-catalogue.md) for the published model
list and licenses.

## Manifest schema (TOML)

Each `manifest.toml` is a single `[model]` table with nested sub-tables. A
detector manifest, abbreviated from `sparrow-engine/tools/examples/megadetector-v6.toml`:

```toml
schema_version = "1.0"

[model]
id           = "megadetector-v6"
type         = "detection"           # detection | classification | audio | image_encoder | ...
format       = "onnx"
file         = "megadetector_v6.onnx"
sha256       = "…"                    # verified at load
input_format = "NCHW"                 # NCHW is mandatory
input_shape  = [-1, 3, 640, 640]
labels_file  = "labels.txt"

[model.preprocessing]
type        = "image_letterbox"
resize      = [640, 640]
scale       = 255.0
color_space = "RGB"

[model.postprocessing]
type                   = "yolo_nms"   # in-graph NMS lane; raw-head models declare megadet_v5a / retinanet_soft_nms
default_conf_threshold = 0.2
iou_threshold          = 0.45
max_detections         = 300

[model.labels]
"0" = "animal"
"1" = "person"
"2" = "vehicle"

# Optional, round-tripped but never interpreted by the engine:
[model.provenance]                    # producer_name / producer_version / training_* ids
# [model.drift_reference]             # reference distribution for Tier-1/2 drift metrics
```

The **authoritative, always-current** schema is the `ModelManifest` type in
`sparrow-engine/sparrow-engine-types/src/manifest.rs`; runnable example
manifests live in `sparrow-engine/tools/examples/` and next to every catalog
model. The top-level [`../user-manual.md`](../user-manual.md) §10 carries the
field-by-field walkthrough.

### Detector-to-classifier crop contract

An image classifier may opt into a pipeline crop convention:

```toml
[crop]
window = "truncate_extent" # round_clamp (default) | truncate_extent
expand_pixels = 0
batch_size = 4
```

`round_clamp` preserves the original pipeline behavior: round normalized edges,
clip to the source image, and reject crops smaller than 2 pixels per axis.
`truncate_extent` uses source-pixel detector geometry and the DeepForest /
rasterio rule: truncate the origin, truncate `max(1, extent)`, apply optional
integer context, then clip to the source image. A detector that cannot supply
source-pixel geometry returns an explicit per-detection
`crop_coords_unavailable` failure rather than approximating the crop.

The section belongs to the classifier manifest, so named pipeline aliases and
ad-hoc detector/classifier pairs use identical crop and batch behavior.
