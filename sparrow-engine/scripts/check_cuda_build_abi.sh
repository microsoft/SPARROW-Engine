#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARGO_TOML="${SPARROW_ENGINE_GPU_CARGO_TOML:-$SCRIPT_DIR/../sparrow-engine-gpu/Cargo.toml}"
NVCC_BIN="${NVCC:-nvcc}"

if ! command -v "$NVCC_BIN" >/dev/null 2>&1; then
    echo "CUDA ABI check: nvcc unavailable; cudarc uses fallback dynamic loading."
    exit 0
fi

feature="$(
    grep -oE 'cuda-[0-9]{5}' "$CARGO_TOML" \
        | sort -u \
        | head -1
)"
if [[ -z "$feature" ]]; then
    echo "FAIL: no cudarc cuda-XXXXX feature found in $CARGO_TOML" >&2
    exit 1
fi

code="${feature#cuda-}"
major="${code:0:2}"
minor_raw="${code:2:2}"
minor="$((10#$minor_raw))"
expected="${major}.${minor}"
actual="$(
    "$NVCC_BIN" --version \
        | sed -nE 's/.*release ([0-9]+\.[0-9]+),.*/\1/p' \
        | tail -1
)"
if [[ -z "$actual" ]]; then
    echo "FAIL: could not parse CUDA release from '$NVCC_BIN --version'" >&2
    exit 1
fi
if [[ "$actual" != "$expected" ]]; then
    echo "FAIL: CUDA toolkit $actual does not match cudarc feature $feature (expects $expected)." >&2
    echo "Build in a CUDA $expected image or change the feature and revalidate every GPU artifact." >&2
    exit 1
fi

echo "CUDA ABI check: PASS ($actual matches $feature)"
