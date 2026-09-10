#!/usr/bin/env bash
# Phase 3.8 Phase C Wave 4a (2026-05-06): build the sparrow-engine / sparrow-engine-gpu
# Python wheels from this single source tree.
#
# Usage:
#   SPARROW_ENGINE_FLAVOR=cpu  ./build.sh   # sparrow-engine wheel (default; pulls onnxruntime)
#   SPARROW_ENGINE_FLAVOR=gpu  ./build.sh   # sparrow-engine-gpu wheel (pulls onnxruntime-gpu)
#   SPARROW_ENGINE_FLAVOR=both ./build.sh   # both wheels (default if SPARROW_ENGINE_FLAVOR is unset)
#
# Output: ../target/wheels/sparrow_engine-*.whl  +/-  ../target/wheels/sparrow_engine_gpu-*.whl
#
# Both flavors build with this directory's unchanged CPU `pyproject.toml`;
# Cargo features select the native implementation. Output-owned wheel staging
# adds the flavor module and, for GPU, renames the distribution and runtime
# dependency and adds an advisory `Provides-Dist: sparrow-engine` to METADATA.
# Repacking regenerates RECORD before the GPU auditwheel gate. That field is
# advisory in pip >=22 (no Conflicts-Dist is emitted), so pip MAY still install
# both `sparrow-engine` and `sparrow-engine-gpu` into one environment — keep the
# flavors in separate environments (operator discipline), not a mechanical block.
#
# References: `docs/design/phase3.8/phase_c/implementation_plan.md`
# §2.3 + §4 W4a + §9 item 4.
#
# Linux GPU wheel — release-quality manylinux build: on Linux, a release-quality
# GPU wheel must be compiled in the release-locked Rocky 8 / glibc 2.28 container
# via `scripts/build_gpu_wheel_manylinux.sh` (matches release.yml). Running this
# script's GPU path directly on a newer-glibc host (e.g. Ubuntu 22.04, glibc
# 2.35) compiles a `linux_x86_64` wheel and then correctly HARD-FAILS the
# `auditwheel repair --plat manylinux_2_28_x86_64` step below — a raw host build
# is NOT a valid portable-wheel path. `scripts/build_all_flavors.sh` dispatches
# the Linux GPU wheel to that container helper automatically.

set -euo pipefail
cd "$(dirname "$0")"

: "${SPARROW_ENGINE_FLAVOR:=both}"

# OS detection. Linux is the canonical wheel target; Windows is supported
# for the GPU wheel (Phase J / 2026-05-26 — `pip install sparrow-engine-gpu`
# on Windows). The Linux-specific maturin `--compatibility linux` flag and
# the `auditwheel repair` post-build step are skipped on Windows; the
# nvidia-* runtime deps are gated with PEP 508 `sys_platform == 'linux'`
# markers because those packages publish Linux-only wheels.
IS_WINDOWS=0
case "${OSTYPE:-$(uname -s)}" in
    msys*|cygwin*|win32*|MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
esac

# Use uv-managed maturin if available; falls back to PATH lookup.
MATURIN="${MATURIN:-maturin}"

cleanup_packaging_dir() {
    local path="$1" wheels_dir="$2" resolved
    if [[ -z "$path" || -L "$path" ]]; then
        echo "[build.sh] ERROR: cleanup refuses empty or symlinked scratch: $path" >&2
        return 2
    fi
    [[ -e "$path" ]] || return 0
    resolved="$(cd "$path" && pwd -P)" || return 2
    if [[ "${resolved%/*}" != "$wheels_dir" || "${resolved##*/}" != .wheel-packaging.* ]]; then
        echo "[build.sh] ERROR: cleanup refuses scratch outside $wheels_dir: $resolved" >&2
        return 2
    fi
    rm -r -- "$resolved"
}

single_wheel() (
    shopt -s nullglob
    local wheels=( "$1/$2-"*.whl )
    if [[ "${#wheels[@]}" -ne 1 || ! -f "${wheels[0]}" ]]; then
        echo "[build.sh] ERROR: expected exactly one new $2 wheel in $1 (found ${#wheels[@]})" >&2
        exit 1
    fi
    printf '%s\n' "${wheels[0]}"
)

repack_wheel() {
    local wheel="$1" scratch="$2" flavor="$3" unpacked
    mkdir -p "$scratch/unpacked" "$scratch/packed"
    uv run --no-project --with wheel python -m wheel unpack "$wheel" -d "$scratch/unpacked"
    unpacked="$(uv run --no-project --with wheel python - "$scratch/unpacked" "$flavor" <<'PY_FLAVOR'
from pathlib import Path
import re
import sys

roots = list(Path(sys.argv[1]).iterdir())
flavor = sys.argv[2]
if len(roots) != 1 or not roots[0].is_dir() or flavor not in ("cpu", "gpu"):
    raise SystemExit("[build.sh] ERROR: invalid unpacked wheel or flavor")
root = roots[0]
package = root / "sparrow_engine"
infos = list(root.glob("sparrow_engine-*.dist-info"))
if not package.is_dir() or len(infos) != 1:
    raise SystemExit("[build.sh] ERROR: expected sparrow_engine package and one CPU-template dist-info")

if flavor == "gpu":
    info = infos[0]
    meta = info / "METADATA"
    headers, separator, body = meta.read_text(encoding="utf-8").partition("\n\n")
    if not separator or re.search(r"(?m)^(Provides-Dist|Conflicts-Dist):", headers):
        raise SystemExit("[build.sh] ERROR: unexpected CPU-template metadata")
    headers, names = re.subn(r"(?m)^Name: sparrow-engine$", "Name: sparrow-engine-gpu", headers)
    headers, runtimes = re.subn(
        r"(?m)^Requires-Dist: onnxruntime(?=[\[ (<>=!~;]|$)",
        "Requires-Dist: onnxruntime-gpu",
        headers,
    )
    if names != 1 or runtimes != 1:
        raise SystemExit("[build.sh] ERROR: expected one CPU distribution name and onnxruntime dependency")
    headers = headers.replace("(sparrow-engine CPU pipeline)", "(sparrow-engine GPU pipeline)")
    headers += '\nProvides-Dist: sparrow-engine'
    for dependency in ("nvidia-cudnn-cu12>=9,<10", "nvidia-cublas-cu12", "nvidia-curand-cu12", "nvidia-cufft-cu12"):
        headers += f'\nRequires-Dist: {dependency}; sys_platform == "linux"'
    meta.write_text(headers + separator + body, encoding="utf-8")
    stem = info.name.removesuffix(".dist-info")
    gpu_stem = stem.replace("sparrow_engine-", "sparrow_engine_gpu-", 1)
    info.rename(root / f"{gpu_stem}.dist-info")
    data = root / f"{stem}.data"
    if data.exists():
        data.rename(root / f"{gpu_stem}.data")

(package / "_flavor.py").write_text(
    f'# Generated by build.sh; do not edit.\nFLAVOR = "{flavor}"\n',
    encoding="utf-8",
)
print(root)
PY_FLAVOR
)"
    # wheel pack regenerates every RECORD hash after flavor/metadata injection.
    uv run --no-project --with wheel python -m wheel pack "$unpacked" -d "$scratch/packed"
}

report_console_scripts() {
    uv run --no-project --with wheel python - "$1" <<'PY_CONSOLE'
from pathlib import Path
import sys
from zipfile import ZipFile

with ZipFile(sys.argv[1]) as wheel:
    entries = [name for name in wheel.namelist() if name.endswith(".dist-info/entry_points.txt")]
    if len(entries) > 1:
        raise SystemExit("[build.sh] ERROR: multiple console-script metadata entries")
    print(f"[build.sh] console scripts in {Path(sys.argv[1]).name}:")
    if entries:
        for line in wheel.read(entries[0]).decode("utf-8").splitlines():
            print(f"    {line}")
    else:
        print("    (no [project.scripts] block - entry_points.txt absent)")
PY_CONSOLE
}

build_wheel() (
    local flavor="$1" wheels_dir scratch wheel dist="sparrow_engine"
    wheels_dir="$(cd .. && pwd -P)/target/wheels"
    mkdir -p "$wheels_dir"
    wheels_dir="$(cd "$wheels_dir" && pwd -P)"
    scratch="$(mktemp -d "$wheels_dir/.wheel-packaging.XXXXXX")"
    cleanup_packaging() {
        local rc=$?
        trap - EXIT
        if ! cleanup_packaging_dir "$scratch" "$wheels_dir"; then
            echo "[build.sh] ERROR: cleanup FAILED for $scratch" >&2
            if [[ "$rc" -eq 0 ]]; then rc=3; fi
        fi
        exit "$rc"
    }
    trap cleanup_packaging EXIT
    mkdir -p "$scratch/raw"

    local compat_args=()
    if [[ "$flavor" == cpu ]]; then
        # ORT is supplied by the wheel's runtime dependency, not bundled.
        compat_args+=(--auditwheel skip)
    else
        dist="sparrow_engine_gpu"
        if [[ "$IS_WINDOWS" -eq 0 ]]; then
            compat_args+=(--compatibility linux)
        fi
    fi
    "$MATURIN" build --release \
        --out "$scratch/raw" \
        "${compat_args[@]}" \
        --no-default-features \
        --features extension-module \
        --features "$flavor"

    wheel="$(single_wheel "$scratch/raw" sparrow_engine)"
    repack_wheel "$wheel" "$scratch" "$flavor"
    wheel="$(single_wheel "$scratch/packed" "$dist")"

    # Windows keeps maturin's native tag. Linux GPU publication requires
    # successful repair; neither the raw nor the merely repacked wheel escapes.
    if [[ "$flavor" == gpu && "$IS_WINDOWS" -eq 0 ]]; then
        mkdir -p "$scratch/repaired"
        auditwheel repair \
            --plat manylinux_2_28_x86_64 \
            --exclude libonnxruntime.so.1 \
            --wheel-dir "$scratch/repaired/" \
            "$wheel"
        wheel="$(single_wheel "$scratch/repaired" "$dist")"
    fi

    report_console_scripts "$wheel"
    mv -- "$wheel" "$wheels_dir/"
)

build_cpu() {
    echo "[build.sh] Building CPU wheel (sparrow-engine, onnxruntime)..."
    build_wheel cpu
    echo "[build.sh] CPU wheel built."
}

build_gpu() {
    echo "[build.sh] Building GPU wheel (sparrow-engine-gpu, onnxruntime-gpu)..."
    bash ../scripts/check_cuda_build_abi.sh
    build_wheel gpu
    echo "[build.sh] GPU wheel built and Provides-Dist patched."
}

case "$SPARROW_ENGINE_FLAVOR" in
    cpu)  build_cpu ;;
    gpu)  build_gpu ;;
    both) build_cpu; build_gpu ;;
    *)
        echo "[build.sh] ERROR: SPARROW_ENGINE_FLAVOR must be cpu / gpu / both (got '$SPARROW_ENGINE_FLAVOR')" >&2
        exit 1
        ;;
esac

ls -lh ../target/wheels/ 2>/dev/null || true
