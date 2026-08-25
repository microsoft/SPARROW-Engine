# CLI integration

Sparrow Engine ships a command-line binary. The CPU build is `spe`; the GPU
build is `spe-gpu`. They expose the **same subcommands**; only the execution
backend differs.

```bash
spe --help          # CPU
spe-gpu --help      # GPU
```

## Global flags

Apply to every subcommand:

| Flag | Default | Purpose |
|------|---------|---------|
| `--device {auto,cpu,cuda:N}` | `auto` | Compute device. Flavor-strict: `spe` stays on CPU, `spe-gpu` stays on CUDA. |
| `--model-dir <dir>` | `$SPARROW_ENGINE_MODEL_DIR` or `~/.sparrow-engine/models` | Base directory of model manifests. |
| `--quiet` | off | Suppress the progress bar (also auto-suppressed when stderr is not a TTY). |
| `--trt-warm-up <ids\|all>` | off | Offline pre-bake of TensorRT engines before the subcommand runs. |

## Subcommands

| Subcommand | Purpose | Model selection |
|------------|---------|-----------------|
| `detect <inputs…>` | Object detection | `--model` optional (default `MDV6-yolov10-e`) |
| `classify <inputs…>` | Single-label classification | `--model` **required** |
| `embed <inputs…>` | Image embeddings from an encoder | `--model` **required** |
| `detect-audio <inputs…>` | Sliding-window audio detection | `--model` optional (default `md-audiobirds-v1`) |
| `pipeline <inputs…>` | Detect → classify | `--detector` + `--classifier` both **required** |
| `models list` | List loaded models | — |
| `models info <id>` | Show one model's manifest fields | — |
| `models verify [<id>] [--write]` | Re-hash ONNX vs. manifest checksums | — |
| `models trt-state <id>` | Show TensorRT warm-up state | — |
| `device` | Print the active compute device | — |
| `init` | Initialize the engine (no inference) | — |
| `hash <file>` | SHA-256 of a file | — |
| `day-night <image>` | Classify an image as day or night | — |

`inputs` accept files, directories (add `--recursive` to walk subtrees), and,
for `embed`, glob patterns. The batch subcommands (`detect`, `classify`,
`embed`, `detect-audio`, `pipeline`) share `--print`, `--format {json,csv}`, and
`--recursive`; the image subcommands also share `--visualize --output-dir <dir>`
(with `--show-labels`), and `detect`/`pipeline` add `--export-format
{megadet,coco,csv}` + `--export-output`.

```bash
# Detection with a threshold, printing one JSON object per file
spe detect trail_cam/*.jpg --model MDV6-yolov10-e --threshold 0.2 --print

# Detect → classify, exporting MegaDetector-format JSON
spe pipeline img.jpg --detector MDV6-yolov10-e --classifier SpeciesNet-Crop \
  --export-format megadet --export-output out.json

# Audio detection, merged ranges by default (--raw-segments for per-window rows)
spe detect-audio recordings/*.wav --model md-audiobirds-v1 --threshold 0.9
```

Key points for integrators (e.g. batch jobs, shell pipelines, CI):

- The CLI and the Python package expose the **same function set** with the same
  conventions (a project rule — Local and Web must not diverge).
- Output is machine-parseable (JSON / ndjson) where a batch consumer needs it.
- Exit codes are non-zero on failure; errors go to stderr as structured logs.
- The `spe` and `spe-gpu` binaries are never co-located; pick the flavor that
  matches the host.

`spe --help` and `spe <subcommand> --help` are the authoritative, always-current
flag reference; the top-level [`../user-manual.md`](../user-manual.md) §5 carries
worked examples. The command surface is defined in
`sparrow-engine/sparrow-engine-cli/src/main.rs`.
