# Python integration

Sparrow Engine ships PyO3 bindings as a wheel. The CPU wheel is `sparrow-engine`
(depends on `onnxruntime`); the GPU wheel is `sparrow-engine-gpu` (depends on
`onnxruntime-gpu`). **Both import as `sparrow_engine`** — a program does not
change its import when switching flavors.

```bash
pip install sparrow-engine          # CPU
pip install sparrow-engine-gpu      # GPU (also install a CUDA/cuDNN-capable onnxruntime-gpu)
```

```python
import sparrow_engine
# Engine is a process-global singleton — construct it once.
```

## Public functions

`import sparrow_engine` exposes the functions below (see `__all__`). The `.pyi`
stub carries exact signatures and return types.

| Group | Functions |
|-------|-----------|
| Lifecycle | `init(device="auto", model_dir=None)` |
| Inference | `detect`, `classify`, `detect_audio`, `pipeline` |
| Embeddings | `embed`, `embed_with_meta`, `embed_aligned`, `embed_aligned_with_meta` |
| Introspection | `list_models`, `list_models_extended`, `model_info`, `active_device` |
| Utilities | `hash_file`, `day_night`, `verify_model`, `summarize` |
| Visualization / export | `visualize`, `visualize_audio`, `export` |

Model selection mirrors the CLI:

- `detect` / `detect_audio` — `model` is optional; when omitted it resolves the
  same default the CLI applies (`MDV6-yolov10-e` for `detect`,
  `md-audiobirds-v1` for `detect_audio`).
- `classify` / `embed` (and the `embed_*` family) — `model` is **required**.
- `pipeline` — `detector` and `classifier` are both **required**.

Inputs are flexible: a path, a directory, a glob (embedding family), or a list
of any of these. Add `recursive=True` to walk subdirectories. The batch
functions accept `progress_callback(index, total, filename)` — `index` is
0-based, called once per file after its inference attempt resolves; raising from
the callback aborts the batch.

```python
import sparrow_engine

# detect returns list[DetectResult], one per input image, in input order
results = sparrow_engine.detect("trail_cam/", model="MDV6-yolov10-e",
                                threshold=0.2, recursive=True)

# DetectResult does not carry the source path — pair it externally
items = list(zip(["trail_cam/a.jpg"], results))
sparrow_engine.visualize(items, output_dir="out/", show_labels=True)  # -> list[bytes]
sparrow_engine.export(items, format="csv", output="out/dets.csv")     # -> str

# Bare-array embeddings (float32); embed_with_meta() keeps identity metadata
vecs = sparrow_engine.embed("crops/", model="bioclip-2-v1")           # np.ndarray
```

Key points for integrators:

- The `Engine` is a **process singleton**. A second construction returns an
  error. Under Python multiprocessing you must use the `spawn` start method,
  **not `fork`** (the singleton guard leaks across `fork`).
- The GIL is released during inference.
- Errors surface as normal Python exceptions (never a bare panic).
- A hand-written `.pyi` type stub ships with the wheel for IDE autocomplete.

The shipped `python/sparrow_engine/_core.pyi` stub is the authoritative type
reference; [`cli.md`](cli.md) lists the equivalent CLI surface, and the
top-level [`../user-manual.md`](../user-manual.md) §6 carries worked examples.
The facade is defined in
`sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py`.
