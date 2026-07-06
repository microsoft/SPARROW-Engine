# Python integration

> **Status: stub.** This page will be expanded with the full Python API
> reference. For now it points at the authoritative sources.

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

Key points for integrators:

- The `Engine` is a **process singleton**. A second construction returns an
  error. Under Python multiprocessing you must use the `spawn` start method,
  **not `fork`** (the singleton guard leaks across `fork`).
- The GIL is released during inference.
- Errors surface as normal Python exceptions (never a bare panic).
- A hand-written `.pyi` type stub ships with the wheel for IDE autocomplete.

The Python API mirrors the CLI command set (same functions, same conventions).
Until this page is filled in, read the shipped `.pyi` stub and
[`cli.md`](cli.md) for the available operations, and the top-level
[`../user-manual.md`](../user-manual.md) for usage examples.
