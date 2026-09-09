# Sparrow Engine

A Rust ML inference engine for camera-trap and bioacoustic data.
Drop-in for MegaDetector v6, DeepFaune, HerdNet, OWL-T, SpeciesNet, and
MD_AudioBirds_V1; model-agnostic via TOML manifests.

## Quickstart

### Easiest: Homebrew (macOS arm64 / brew-Linux x86_64)

```bash
brew tap microsoft/sparrow-engine
brew install sparrow-engine            # CPU; works on macOS arm64 + brew-Linux x86_64
brew install sparrow-engine-gpu        # GPU; brew-Linux x86_64 + NVIDIA only

spe device                              # {"device":"cpu"}  or  {"device":"cuda:0"}

# One-time: download a model from the Zenodo bundle (brew doesn't ship models)
mkdir -p ~/.sparrow-engine/models && cd ~/.sparrow-engine/models
curl -fLO https://zenodo.org/records/22018132/files/camera_trap__detector__MDV6-yolov10-e.zip
unzip -q camera_trap__detector__MDV6-yolov10-e.zip && rm camera_trap__detector__MDV6-yolov10-e.zip
cd -

spe detect /path/to/photos --model MDV6-yolov10-e --recursive --export-format megadet --export-output detections.json
```

Both formulas can coexist (separate binaries `spe` + `spe-gpu`; shared model cache at `~/.sparrow-engine/models/`). The current public model zoo remains v0.29.0 with 75 archives. This branch stages an **unpublished 80-entry v0.30.0 candidate**: 47 currently downloadable runtime packages, 28 link-only metadata entries, and 5 admitted packages that remain marked as release candidates until the new Zenodo version exists. The example above pulls MegaDetector v6 (general camera-trap detection); see the [Model zoo](#model-zoo) section below for the full candidate catalog. See `docs/user-manual.md §2.4` for the other install paths.

#### GPU host prerequisites

The `sparrow-engine-gpu` formula ships ~256 MB of `libonnxruntime` + ORT CUDA provider sidecars, but it does **NOT** bundle NVIDIA's runtime libraries (NVIDIA's license forbids redistribution). The host must provide:

| Library | Apt package (Ubuntu/Debian) | pip wheel (no root) | Why |
|---|---|---|---|
| NVIDIA driver ≥550.x | `nvidia-driver-550` (or newer) | — (kernel module; host-only) | GPU access |
| CUDA runtime 12.6 | `nvidia-cuda-toolkit` brings it | `nvidia-cuda-runtime-cu12` | `libcudart.so.12` |
| **cuDNN ≥9.10** (9.8 has Conv bug on sm_89) | `nvidia-cudnn` | `nvidia-cudnn-cu12` | `libcudnn.so.9` — convolutions |
| cuBLAS | bundled with CUDA toolkit | `nvidia-cublas-cu12` | matrix multiplications |
| cuRAND | bundled with CUDA toolkit | `nvidia-curand-cu12` | rand sampling (some models) |
| cuFFT | bundled with CUDA toolkit | `nvidia-cufft-cu12` | audio FFT (MD_AudioBirds_V1) |
| nvJPEG | bundled with CUDA toolkit | `nvidia-nvjpeg-cu12` | GPU JPEG decode |

After installing the libraries (system or pip), the brew-installed `spe-gpu` wrapper auto-discovers them from common host locations — no `LD_LIBRARY_PATH` setup needed for production users. Search order (first hit wins):

1. `SPARROW_ENGINE_CUDA_LIB_DIR` (user override; honored as-is)
2. `~/.sparrow-engine/cuda-sidecars/lib/python*/site-packages/nvidia/*/lib` (the convention if you used pip sidecars)
3. `/usr/lib/python3/dist-packages/torch/lib` (Lambda Stack / system PyTorch — cuDNN comes bundled)
4. `/usr/local/cuda/lib64` (NVIDIA CUDA toolkit)
5. `/usr/lib/x86_64-linux-gnu` (Ubuntu apt nvidia-cudnn)

Full table + remediation appears in `brew info sparrow-engine-gpu`. Quick all-pip install (no root) for a fresh host:

```bash
uv venv ~/.sparrow-engine/cuda-sidecars --python 3.11
~/.sparrow-engine/cuda-sidecars/bin/pip install \
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 \
    nvidia-cufft-cu12 nvidia-nvjpeg-cu12 nvidia-cuda-runtime-cu12
```

Verify with `spe-gpu device` — `{"device":"cuda:0"}` means good, any dlopen error in the output names the missing library.

### Alternative install paths

If brew isn't right for your environment (server distro without brew-Linux, Windows, etc.), the install wrapper handles probe-and-install for Linux / macOS / Windows:

```bash
# Linux / macOS — clone the repo and run from its root
bash installer/sparrow-engine-install.sh
```

```powershell
# Windows PowerShell — clone the repo and run from its root
installer\sparrow-engine-install.ps1
```

The wrapper probes hardware once, picks the right CPU or GPU build, and
installs the matching CLI binary plus the Python wheel into `~/.sparrow-engine/`.
Pass `--flavor cpu` or `--flavor gpu` to skip the probe. Pass `--docker`
to install the HTTP-server image instead.

System prerequisites for GPU: NVIDIA driver ≥550.x, CUDA 12.6 runtime,
and **cuDNN ≥9.10** (cuDNN 9.8 has a Conv-engine bug on sm_89).

### Python package only (PyPI)

If you only want the Python wheel — no CLI, no Docker image — install
straight from PyPI. Both wheels target CPython ≥ 3.11 (`cp311-abi3`), so
make sure your venv runs Python 3.11 or newer.

**With `uv` (recommended)**:

```bash
uv venv --python 3.11
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# CPU
uv pip install sparrow-engine

# GPU (Linux x86_64 only; requires CUDA 12.6 runtime on the host)
uv pip install sparrow-engine-gpu
```

`uv venv` does not ship `pip` inside the venv by default, so use `uv pip
install` (uv's pip-compatible wrapper) instead of bare `pip install`.
Calling `pip install …` after `source activate` falls back to the system
pip, which usually targets the wrong Python version and fails with
`No matching distribution found`.

**With stdlib `venv`**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# CPU
pip install sparrow-engine

# GPU (Linux x86_64 only; requires CUDA 12.6 runtime on the host)
pip install sparrow-engine-gpu
```

Both wheels import as `sparrow_engine`. Never install both into the same
environment. Check the installed version with
`python -c "import sparrow_engine; print(sparrow_engine.__version__)"`.
See [§6 of the user manual](docs/user-manual.md#6-python-package--sparrow-engine)
for the full API surface and GPU sidecar options.

### Docker image (server deployments)

Sparrow Engine ships as a self-contained HTTP server in two Docker flavors. Both expose the same HTTP API on port 8080 (`/v1/detect`, `/v1/classify`, `/v1/audio/detect`, `/v1/health`, `/healthz`, and more). See [§7 of the user manual](docs/user-manual.md#7-http-api-server--sparrow-engine-server) for the full route list.

| Image | Size | GPU |
|---|---|---|
| `zhongqimiao/sparrow-engine-server:latest` | ~170 MB | CPU only |
| `zhongqimiao/sparrow-engine-server-gpu:latest` | ~3.7 GB | CUDA 12 + cuDNN bundled; requires NVIDIA Container Toolkit on the host |

Two install paths. **Option A** is the simplest; **Option B** builds from source for the absolute-latest-source case.

**Option A — `docker pull` from Docker Hub** (RP-35, 2026-06-05; published on every prod tag via `release.yml`):

```bash
# CPU image (~61 MB compressed, ~170 MB extracted)
docker pull zhongqimiao/sparrow-engine-server:latest
docker pull zhongqimiao/sparrow-engine-server:v0.1.28        # version pin (recommended for prod)

# GPU image (~2.2 GB compressed, ~3.7 GB extracted)
docker pull zhongqimiao/sparrow-engine-server-gpu:latest
docker pull zhongqimiao/sparrow-engine-server-gpu:v0.1.28
```

Public repos (anonymous pull, no Docker Hub login required):

- https://hub.docker.com/r/zhongqimiao/sparrow-engine-server
- https://hub.docker.com/r/zhongqimiao/sparrow-engine-server-gpu

Heads-up: anonymous Docker Hub pulls are rate-limited (100 pulls / 6 hr / source IP). For CI behind shared NAT, `docker login` with a free Docker Hub account lifts the limit to 200/6 hr.

**Option B — build from source** (~10 min the first time; cached layers on subsequent builds; always reflects the current source tree):

```bash
git clone https://github.com/microsoft/SPARROW-Engine.git
cd SPARROW-Engine/sparrow-engine
docker build -f docker/Dockerfile.cpu -t sparrow-engine-server:sparrow-combined .
docker build -f docker/Dockerfile.gpu -t sparrow-engine-server-gpu:sparrow-combined .  # GPU
```

**Run the server** (after any of the three options). The container expects models mounted read-only at `/models`:

```bash
# CPU (Option A pull)
docker run -d --rm --name sparrow-engine -p 8080:8080 \
  -v $HOME/.sparrow-engine/models:/models:ro \
  -e SPARROW_ENGINE_DEVICE=cpu \
  zhongqimiao/sparrow-engine-server:latest

# GPU (requires NVIDIA Container Toolkit on the host)
docker run -d --rm --name sparrow-engine-gpu -p 8080:8080 --gpus all \
  -v $HOME/.sparrow-engine/models:/models:ro \
  -e SPARROW_ENGINE_DEVICE=cuda:0 \
  zhongqimiao/sparrow-engine-server-gpu:latest

# Verify
curl -fsS http://localhost:8080/healthz            # liveness → {"alive":true}
curl -fsS http://localhost:8080/v1/health | jq     # readiness + catalog size
curl -fsS http://localhost:8080/v1/catalog | jq    # discovered models
```

**Or use the bundled `docker-compose.yml`** (resource limits, healthcheck, log rotation, read-only filesystem all pre-configured):

```bash
cd SPARROW-Engine/sparrow-engine/docker
docker compose --profile cpu up -d        # CPU
docker compose --profile gpu up -d        # GPU
docker compose --profile cpu logs -f      # tail logs
docker compose --profile cpu down         # stop
```

The Compose file mounts `${SPARROW_ENGINE_MODEL_DIR:-./models}` read-only into the container; set the env var or place models at `sparrow-engine/docker/models/` before bringing the stack up. Models can also be downloaded via the [Model zoo](#model-zoo) section below.

For full HTTP API documentation, request shapes, response schemas, and operator-grade env-var reference: [§7 of the user manual](docs/user-manual.md#7-http-api-server--sparrow-engine-server).

---

### Edge / ARM — the mobile flavor (`spe-mobile`)

A third flavor, **`sparrow-engine-mobile`**, targets ARM edge devices (Raspberry Pi; Android via the cdylib). It swaps ONNX Runtime for a **TensorFlow Lite / LiteRT** backend and ships as a cross-compiled `aarch64` cdylib (`libsparrow_engine.so`) plus the `spe-mobile` CLI — **no Homebrew formula and no Python wheel**; mobile consumers call the cdylib over native FFI (ctypes / JNI / Swift). Like the CPU/GPU flavors it is a **generic, manifest-driven engine** (`engine_new` → `load_pipeline_by_id` → `run_pipeline`); the orca two-stage detector→ecotype cascade ships as a manifest-described `pipeline.toml`, not hardcoded C.

```bash
# cross-build the CLI (use --features ffi for the cdylib instead)
cross build -p sparrow-engine-mobile --features cli --release --target aarch64-unknown-linux-gnu

# run a config-described cascade over WAVs
# (model catalog = {model_dir}/{id}/manifest.toml + {pipeline}/pipeline.toml)
spe-mobile detect-audio --model-dir /path/to/model_catalog --pipeline orca-cascade --threads 4 recording.wav
```

Validated on a 512 MB Raspberry Pi Zero 2W: both fp16 Orca `.tflite` models use ~282 MB resident memory (~297 MB observed peak), ≤ 2 s/segment (4-thread XNNPACK). The mobile catalog includes five TFLite model artifacts — `MDV6-yolov10-c-tflite`, two Orca detector variants, and two Orca ecotype variants — plus the `orca-cascade` pipeline descriptor. The validated mobile inference paths today are the Orca cascade and `MDV6-yolov10-c-tflite` image detection. Mobile image classification remains unavailable until a TFLite classifier is onboarded (tracked as RP-42-FU-1). Full details + flag reference: §5.7 of the [user manual](docs/user-manual.md).

---

> 📖 **[Read the full user manual →](docs/user-manual.md)**
>
> One document covering install, CLI (`spe`), Python wheel (`import sparrow_engine`), HTTP API server, HTTP SDK, native DLL (C ABI), TOML model manifests, the Phase 4 inference-log / drift / provenance surface, cold-start + lazy load, gotchas + edge cases, performance characteristics, and Sparrow Studio integration.

---

## Model zoo

Sparrow Engine doesn't ship model artifacts in the repo. They live in a public Zenodo record so the repo stays small and operators can pull just the models they need.

**Zenodo DOI**: [10.5281/zenodo.22018132](https://doi.org/10.5281/zenodo.22018132) (v0.29.0) — concept DOI [10.5281/zenodo.20348978](https://doi.org/10.5281/zenodo.20348978) always resolves to the latest version.

Download the 38 default hosted desktop ONNX models into
`~/.sparrow-engine/models`, the default directory used by `spe`,
`sparrow-engine-server`, and the Python wheel, by running the downloader
without arguments. The current published hosted set also holds 5 mobile TFLite models, 1 cascade descriptor, 0 recording-level ensemble packages, and 3 opt-in ONNX models. Together with
28 link-only entries and 5 release-candidate entries, they form this branch's
complete **80-entry** catalog. The recording-level ensemble and second
cascade are among the five candidates and cannot be downloaded from v0.29.0.
Non-default published entries are fetched when named explicitly or with
`--all`. Link-only entries remain discoverable but must be obtained from
their original source:

```bash
bash scripts/download_models.sh
```

Or just specific models:

```bash
bash scripts/download_models.sh MDV6-yolov10-e AI4G-Amazon-V2
bash scripts/download_models.sh --list          # list available model IDs
bash scripts/download_models.sh --dest /custom/path
```

Point Sparrow Engine at the directory (only needed if you used `--dest`; the default location is auto-detected):

```bash
# Default path (auto-detected — env var only needed if you want to be explicit):
export SPARROW_ENGINE_MODEL_DIR=$(realpath ~/.sparrow-engine/models)
# Custom path (required if you used `--dest /opt/sparrow-models`):
export SPARROW_ENGINE_MODEL_DIR=/opt/sparrow-models
spe models list                                 # confirms catalog discovery
spe detect --model MDV6-yolov10-e --print image.jpg
```

The downloader verifies MD5 per hosted package (against the Zenodo record API),
is idempotent (skip-if-present unless `--force`), and unpacks runtime
descriptors (`manifest.toml`, `pipeline.toml`, or `ensemble.toml`) plus their
declared assets. Link-only entries fail with the original source URL; aliases,
`--force`, `--no-verify`, and record overrides cannot bypass that routing.

### Unpublished 80-entry release-candidate catalogue

The index below is generated from
[`sparrow-engine/scripts/catalog.toml`](sparrow-engine/scripts/catalog.toml),
the source of truth used by `download_models.sh`. The
[full catalogue](docs/model-zoo-catalogue.md) adds geography, behavior,
developer/owner details, and source citations.

The five rows whose catalog status is `candidate` are admitted locally but are
not present in the linked v0.29.0 Zenodo record. The downloader rejects them
before network access until a verified v0.30.0 record is published and the
rows are activated.

<!-- BEGIN GENERATED MODEL ZOO INDEX -->

| Area | Models |
|---|---:|
| Camera Trap | 48 |
| Acoustics | 13 |
| Overhead | 8 |
| Marine Imagery | 6 |
| General | 5 |
| **Total** | **80** |

Formats: **72 ONNX**, **5 TFLite**, **2 cascade descriptor**, **1 recording-level ensemble**.

Counts include metadata-only entries. Hosted and hosted-restricted packages contain runtime artifacts; link-only and pending-rights entries do not. Each model retains its recorded licence terms.

#### Camera Trap — Detectors (21)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `MDV5a` | MDV5a | MegaDetector | onnx · v5a | UNVERIFIED | Link-only | unverified |
| `MDV6-yolov10-c` | MDV6-yolov10-c | MegaDetector | onnx · v6 | AGPL-3.0 | Hosted | allowed |
| `MDV6-yolov10-c-tflite` | MDV6-yolov10-c-tflite | MegaDetector | tflite-fp16 · v6 | AGPL-3.0 | Hosted | allowed |
| `MDV6-yolov10-e` | MDV6-yolov10-e | MegaDetector | onnx · v6 | AGPL-3.0 | Hosted | allowed |
| `deepfaune-yolo8s` | deepfaune-yolo8s | DeepFaune | onnx | CC-BY-SA-4.0 | Hosted-restricted | allowed |
| `european_mammals` | MD European Mammals | MegaDetector | onnx | UNVERIFIED | Link-only | unverified |
| `north_american_mammals` | MD North American Mammals | MegaDetector | onnx | UNVERIFIED | Link-only | unverified |
| `sub_saharan` | MD Sub-Saharan Mammals | MegaDetector | onnx | UNVERIFIED | Link-only | unverified |
| `MDV5b` | MDV5b | MegaDetector | onnx · v5b | UNVERIFIED | Link-only | unverified |
| `MD1000-redwood` | MD1000-redwood | MegaDetector | onnx · v1000-redwood | UNVERIFIED | Link-only | unverified |
| `MD1000-spruce` | MD1000-spruce | MegaDetector | onnx · v1000-spruce | UNVERIFIED | Link-only | unverified |
| `MD1000-larch` | MD1000-larch | MegaDetector | onnx · v1000-larch | UNVERIFIED | Link-only | unverified |
| `MD1000-cedar` | MD1000-cedar | MegaDetector | onnx · v1000-cedar | UNVERIFIED | Link-only | unverified |
| `MD1000-sorrel` | MD1000-sorrel | MegaDetector | onnx · v1000-sorrel | UNVERIFIED | Link-only | unverified |
| `MDV6-yolov9-c` | MDV6-yolov9-c | MegaDetector | onnx · v6 | AGPL-3.0 | Hosted | allowed |
| `MDV6-yolov9-e` | MDV6-yolov9-e | MegaDetector | onnx · v6 | AGPL-3.0 | Hosted | allowed |
| `MDV6-rtdetr-c` | MDV6-rtdetr-c | MegaDetector | onnx · v6 | AGPL-3.0 | Hosted | allowed |
| `MDV6-mit-yolov9-c` | MDV6-mit-yolov9-c | MegaDetector | onnx · v6 | MIT | Hosted | allowed |
| `MDV6-mit-yolov9-e` | MDV6-mit-yolov9-e | MegaDetector | onnx · v6 | MIT | Hosted | allowed |
| `MDV6-apa-rtdetr-c` | MDV6-apa-rtdetr-c | MegaDetector | onnx · v6 | Apache-2.0 | Hosted | allowed |
| `MDV6-apa-rtdetr-e` | MDV6-apa-rtdetr-e | MegaDetector | onnx · v6 | Apache-2.0 | Hosted | allowed |

#### Camera Trap — Classifiers (27)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `AI4G-Amazon-V2` | AI4G-Amazon-V2 | AI4G | onnx | MIT | Hosted | allowed |
| `AI4G-Serengeti` | AI4G-Serengeti | AI4G | onnx | MIT | Hosted | allowed |
| `Deepfaune-Europe` | Deepfaune-Europe | DeepFaune | onnx | CC-BY-SA-4.0 | Hosted-restricted | allowed |
| `Deepfaune-New-England` | Deepfaune-New-England | DeepFaune | onnx | CC-BY-NC-SA-4.0 | Hosted-restricted | non-commercial |
| `SpeciesNet-Crop` | SpeciesNet-Crop | SpeciesNet | onnx | UNVERIFIED | Link-only | unverified |
| `southwest-usa-v3` | southwest-usa-v3-SDZWA | AddaxAI | onnx | UNVERIFIED | Link-only | unverified |
| `peruvian-andes` | peruvian-andes-SDZWA | AddaxAI | onnx | UNVERIFIED | Link-only | unverified |
| `sub-saharan-drylands` | sub-saharan-drylands-Addax | AddaxAI | onnx | CC-BY-NC-SA-4.0 | Link-only | non-commercial |
| `manas-panthera` | manas-panthera | AddaxAI | onnx | CC-BY-NC-SA-4.0 | Hosted-restricted | non-commercial |
| `gifu-japan` | gifu-japan-GifuUniversity | AddaxAI | onnx | UNVERIFIED | Link-only | unverified |
| `hawaii-puaa` | hawaii-puaa-Addax | AddaxAI, SpeciesNet | onnx | CC-BY-NC-4.0 | Link-only | non-commercial |
| `central-india` | central-india-Addax | AddaxAI, SpeciesNet | onnx | CC-BY-NC-4.0 | Link-only | non-commercial |
| `top-end-savanna` | top-end-savanna-Addax | AddaxAI, SpeciesNet | onnx | CC-BY-NC-4.0 | Link-only | non-commercial |
| `parks-victoria` | parks-victoria-Addax | AddaxAI, SpeciesNet | onnx | Apache-2.0 | Hosted | allowed |
| `sw-borderlands` | sw-borderlands-Addax | AddaxAI, SpeciesNet | onnx | UNVERIFIED | Link-only | unverified |
| `ahdrift` | ahdrift-OSU-ColumbusZoo-Addax | AddaxAI, SpeciesNet | onnx | UNVERIFIED | Link-only | unverified |
| `deep-forest-vision` | deep-forest-vision-MNHN-OFVI | AddaxAI | onnx | CC-BY-NC-SA-4.0 | Hosted-restricted | non-commercial |
| `awc135` | awc135-AWC | AddaxAI | onnx | CC-BY-NC-SA-4.0 | Hosted-restricted | non-commercial |
| `namibian` | namibian-Addax | AddaxAI | onnx | CC-BY-NC-SA-4.0 | Link-only | non-commercial |
| `iran` | iran-Addax | AddaxAI | onnx | CC-BY-NC-SA-4.0 | Link-only | non-commercial |
| `nz-invasives` | nz-invasives-Addax | AddaxAI | onnx | CC-BY-NC-SA-4.0 | Link-only | non-commercial |
| `queensland` | queensland-WildObs | AddaxAI, SpeciesNet | onnx | CC-BY-4.0 | Hosted | allowed |
| `nz-species` | nz-species-wekaResearch | AddaxAI | onnx | UNVERIFIED | Link-only | non-commercial |
| `tropicam-ai` | tropicam-ai-MNCN-CSIC | AddaxAI | onnx | CONFLICTING | Link-only | non-commercial |
| `terai-nepal` | terai-nepal | AddaxAI, MEWC | onnx | MIT | Hosted | allowed |
| `tasmanian-vertebrates` | tasmanian-vertebrates-MEWC | AddaxAI, MEWC | onnx | CC-BY-NC-4.0 | Hosted-restricted | non-commercial |
| `peruvian-amazon-sdzwa` | peruvian-amazon-SDZWA | AddaxAI | onnx | UNVERIFIED | Link-only | unverified |

#### Acoustics — Detectors (5)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `md-audiobirds-v1` | md-audiobirds-v1 | — | onnx | MIT | Hosted | allowed |
| `orca-detector-dclde2026-v5` | orca-detector-dclde2026-v5 | DCLDE-orca | onnx · v5 | MIT | Hosted | allowed |
| `orca-detector-v5-fp16-tflite` | orca-detector-v5-fp16-tflite | DCLDE-orca | tflite-fp16 · v5 | MIT | Hosted | allowed |
| `orca-detector-v5-int8-tflite` | orca-detector-v5-int8-tflite | DCLDE-orca | tflite-int8 · v5 | MIT | Hosted | allowed |
| `batdetect2-uk-v2` | BatDetect2 v2 UK Bat Call Detector | BatDetect2 | onnx · 2.0.0b3 | CC-BY-NC-4.0 | Hosted-restricted | non-commercial |

#### Acoustics — Classifiers (7)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `orca-ecotype-dclde2026-v1` | orca-ecotype-dclde2026-v1 | DCLDE-orca | onnx · v1 | MIT | Hosted | allowed |
| `orca-ecotype-melinput-fp16-tflite` | orca-ecotype-melinput-fp16-tflite | DCLDE-orca | tflite-fp16 | MIT | Hosted | allowed |
| `orca-ecotype-melinput-int8-tflite` | orca-ecotype-melinput-int8-tflite | DCLDE-orca | tflite-int8 | MIT | Hosted | allowed |
| `perch-v2` | perch-v2 | — | onnx | Apache-2.0 | Hosted | allowed |
| `buzzdetect` | BuzzDetect Acoustic Event Classifier | BuzzDetect, YAMNet | onnx | MIT AND Apache-2.0 | Hosted | allowed |
| `perch-v2-fp16` | perch-v2-fp16 | — | onnx-fp16 | Apache-2.0 | Hosted | allowed |
| `hawkears-v2` | HawkEars 2.2 Bird Audio Ensemble | HawkEars | ensemble · 2.2.0 | MIT | Hosted | allowed |

#### Acoustics — Cascade (1)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `orca-cascade` | orca-cascade | DCLDE-orca | cascade | MIT | Hosted | allowed |

#### Overhead — Detectors (6)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `HerdNet_General_Dataset_2022` | HerdNet\_General\_Dataset\_2022 | — | onnx | CC-BY-NC-SA-4.0 | Hosted-restricted | non-commercial |
| `OWL` | OWL | — | onnx | CC-BY-NC-SA-4.0 | Hosted-restricted | non-commercial |
| `imageomics-mmla` | Imageomics MMLA Aerial Wildlife Detector | Imageomics, YOLO11 | onnx | MIT | Hosted | allowed |
| `deepforest-tree` | DeepForest Tree-Crown Detector | DeepForest, RetinaNet | onnx | MIT | Hosted | allowed |
| `deepforest-bird` | DeepForest Aerial Bird Detector | DeepForest, RetinaNet | onnx | MIT | Hosted | allowed |
| `ducknet` | DuckNet Waterfowl Detector | DuckNet, RetinaNet | onnx | UNVERIFIED | Link-only | non-commercial |

#### Marine Imagery — Detectors (6)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `fathomnet-mbari-315k` | FathomNet MBARI 315k Detector | FathomNet, YOLOv8 | onnx | CC-BY-4.0 | Hosted | allowed |
| `fathomnet-vme` | FathomNet VME Detector | FathomNet, YOLOv8 | onnx | CC-BY-4.0 | Hosted | allowed |
| `fathomnet-trash` | FathomNet Trash Detector | FathomNet, YOLOv8 | onnx | CC-BY-4.0 | Hosted | allowed |
| `fathomnet-megafish-yolov5s-640` | FathomNet MegaFishDetector YOLOv5s 640 | FathomNet, MegaFishDetector, YOLOv5 | onnx · v0 | MIT | Hosted | allowed |
| `fathomnet-megafish-yolov5m-1280` | FathomNet MegaFishDetector YOLOv5m 1280 | FathomNet, MegaFishDetector, YOLOv5 | onnx · v0 | MIT | Hosted | allowed |
| `fathomnet-megafish-yolov5l-640` | FathomNet MegaFishDetector YOLOv5l 640 | FathomNet, MegaFishDetector, YOLOv5 | onnx · v0 | MIT | Hosted | allowed |

#### General — Encoders (4)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `bioclip-2` | bioclip-2 | BioCLIP | onnx · v2 | MIT | Hosted | allowed |
| `bioclip-2-fp16` | bioclip-2-fp16 | BioCLIP | onnx-fp16 · v2 | MIT | Hosted | allowed |
| `bioclip-25` | BioCLIP 2.5 Huge | BioCLIP | onnx-fp16 · 1.0.0 | MIT | Hosted | allowed |
| `dinov3-vitl16` | dinov3-vitl16 | DINOv3 | onnx · vitl16-lvd1689m | DINOv3 License | Link-only | unverified |

#### General — Classifiers (1)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `plantclef-dinov2` | PlantCLEF 2024 DINOv2 Plant Classifier | PlantCLEF, DINOv2 | onnx · plantclef2024-full-finetune-ema | CC-BY-4.0 | Hosted | allowed |

#### Overhead — Classifiers (1)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `deepforest-neon-species` | DeepForest NEON Tree-Species Classifier | DeepForest | onnx · cropmodel-tree-species-3efe2a25 | MIT | Hosted | allowed |

#### Overhead — Cascade (1)

| Model ID | Display name | Family | Format | Licence | Hosting | Commercial use |
|---|---|---|---|---|---|---|
| `deepforest-tree-species` | DeepForest Tree-Species Pipeline | DeepForest | cascade | MIT | Hosted | allowed |

<!-- END GENERATED MODEL ZOO INDEX -->

Every model remains subject to its own original licence and usage
conditions. NonCommercial, ShareAlike, academic-use, custom, and other
restrictions continue to apply to users of the corresponding model.
---

## Architecture

Sparrow Engine is engine-only: it loads ONNX models and runs inference.
Annotation, training, data versioning, model registry, drift detection,
and deployment orchestration live in sibling repos.

Core invariants:

- ONNX for all models (vision + audio)
- NCHW layout mandatory
- Normalized bbox `[0,1]` at all public API boundaries
- TOML manifests (one per model)
- NMS in the ONNX graph, never in the Sparrow Engine
- `Engine` is a singleton (ORT is process-global)

## License

See [`LICENSE`](LICENSE).

---

## Internal development

This is the **public** sparrow-engine repo. It carries the shipping code, the install wrapper, models, and one user-facing manual.

Dev/AI artifacts — design rounds, research notes, audit-fix / doc-fix / `/implement` skill rounds, inquisitor reports, scope ledgers, prompt logs, agent instructions, plan / changelog / lessons / ideas — live in the **internal dev companion** repo (`zhmiao/sparrow-engine-dev`), NOT here. See that repo's `docs/design/architecture.md § Internal dev companion convention` for the full rule.
