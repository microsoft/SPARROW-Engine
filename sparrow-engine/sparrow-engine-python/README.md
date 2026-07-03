# sparrow-engine (Python)

Camera-trap ML inference engine — Python API. `sparrow-engine` loads ONNX
models and runs detection, classification, and audio inference.

This package ships the **Python API only** (`import sparrow_engine`). The
command-line binaries (`spe` / `spe-gpu`) and the HTTP server are distributed
separately (Homebrew, the system installer, or the GitHub Release tarball) —
`pip install` does not place them on your `PATH`.

## Install

```bash
pip install sparrow-engine        # CPU build (depends on onnxruntime)
pip install sparrow-engine-gpu    # GPU/CUDA build (depends on onnxruntime-gpu)
```

Both distributions import as `sparrow_engine` and are drop-in replacements for
each other. Install exactly one per environment; pip refuses to install both
into the same environment. Neither wheel bundles ONNX Runtime or CUDA — those
come from the runtime dependency (`onnxruntime` / `onnxruntime-gpu`).

## Usage

```python
import sparrow_engine

# Models are read from ~/.sparrow-engine/models (override with
# SPARROW_ENGINE_MODEL_DIR). Device defaults to "auto".
print(sparrow_engine.list_models())

# Object detection — returns list[DetectResult], one per input image.
results = sparrow_engine.detect("photo.jpg", model="MDV6-yolov10-c")
for det in results[0].detections:
    # bbox coordinates are normalized to [0, 1].
    print(det.label, det.confidence, det.bbox.x_min, det.bbox.y_min)

# Classification — result.top1 is the highest-confidence class (or None).
clf = sparrow_engine.classify("crop.jpg", model="Deepfaune-Europe")
print(clf[0].top1)

# Detect-then-classify pipeline (ad-hoc; no TOML required).
pipe = sparrow_engine.pipeline("photo.jpg", detector="MDV6-yolov10-c",
                               classifier="Deepfaune-Europe")

# Audio detection (WAV input).
audio = sparrow_engine.detect_audio("recording.wav", model="md-audiobirds-v1")
```

`init(device=..., model_dir=...)` is optional — the engine auto-initializes on
the first inference call. `detect` / `classify` / `detect_audio` / `pipeline`
each accept a file path, a directory, or a list of paths, and take an optional
`progress_callback(index, total, filename)`.

## Documentation

See the user manual (`docs/user-manual.md`) in the sparrow-engine repository for
the full model catalog, device selection, and the CLI / server surfaces.
