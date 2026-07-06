# Native (FFI) integration — Sparrow Studio Local

**Sparrow Studio Local** (and any native desktop app) consumes Sparrow Engine
as a dynamic library through its C ABI. This page covers the packaging and
call-lifecycle rules; the exhaustive symbol list and ABI-evolution policy live
in [`ffi-abi.md`](ffi-abi.md).

## The library

| Flavor | File name |
|---|---|
| CPU | `libsparrow_engine.so` / `sparrow_engine.dll` / `libsparrow_engine.dylib` |
| GPU | **same names** |

Both flavors ship with the **same library name** on purpose, so a consumer's
`[DllImport("sparrow_engine")]` (or `dlopen` path) does not change when swapping
CPU↔GPU. The two are never placed in the same directory — deploy exactly one
per install.

The library dynamically loads ONNX Runtime at runtime (it does not statically
link or bundle it). Provide an ONNX Runtime shared library on the load path:
`onnxruntime` for the CPU flavor, `onnxruntime-gpu` for the GPU flavor. The GPU
flavor additionally needs the NVIDIA driver + CUDA + **cuDNN ≥ 9.10**
co-deployed (the CUDA EP fails without cuDNN). See
[`../user-manual.md`](../user-manual.md) for platform-specific packaging.

## Call lifecycle

```
engine = sparrow_engine_engine_new(device, model_dir)   // once per process
  sparrow_engine_load_model(engine, ...)                // or _load_model_by_id
    result = sparrow_engine_detect(engine, ...)         // or classify/embed/...
    // read result fields …
    sparrow_engine_detections_free(result)              // matching _free, exactly once
  sparrow_engine_unload_model(engine, ...)
sparrow_engine_engine_free(engine)
```

Every result has a **matching `_free`** (see the pairing table in
[`ffi-abi.md`](ffi-abi.md)). Free exactly once; the `_free` functions are
null-safe.

## Hard rules

- **The engine is a process-global singleton.** ONNX Runtime is process-global,
  so a second `sparrow_engine_engine_new` returns an error. Construct one engine
  and share it.
- **`fork()` hazard.** The singleton guard is an `AtomicBool` that leaks to
  `fork()`'d children. If the host embeds a runtime that forks, the child must
  not re-init the engine. (In Python this is why multiprocessing must use
  `spawn`, not `fork` — see [`python.md`](python.md).)
- **Errors are errno-style.** On failure a call returns a null/negative
  sentinel; retrieve the message with `sparrow_engine_last_error` (thread-local).
- **No panics cross the ABI.** Every export wraps its body in `catch_unwind`.
- **Handle/result thread affinity.** Use and free a result on a sane thread
  discipline; do not free a handle produced on another thread from an unrelated
  thread. (The mobile flavor is strictly thread-affine — handles must be used
  and freed on the creating thread.)

## P/Invoke sketch (C#)

```csharp
[DllImport("sparrow_engine")]
static extern IntPtr sparrow_engine_engine_new(string device, string modelDir);

[DllImport("sparrow_engine")]
static extern void sparrow_engine_engine_free(IntPtr engine);

[DllImport("sparrow_engine")]
static extern IntPtr sparrow_engine_last_error();
// … declare detect/classify/embed + their _free counterparts per ffi-abi.md
```

Marshal the opaque `IntPtr` engine handle straight through; never interpret it.
Match each result-returning call with its `_free`. Reference smoke tests live
at `sparrow-engine/tests/cs_pinvoke_smoke/` and
`sparrow-engine/tests/csharp_ffi_test/`.

## Stability

The C ABI is a **stable contract**, evolved only by adding `_v2` symbols
(never by re-signing an existing one). A G5 acceptance gate enforces the
37-symbol set and CPU/GPU parity. See [`ffi-abi.md`](ffi-abi.md).
