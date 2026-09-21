#!/usr/bin/env bash
# Run cargo tests with ORT dynamic linking.
#
# GPU is the default. The script prefers onnxruntime-gpu over onnxruntime-cpu.
# The pre-built ORT static lib (ort-sys) requires glibc 2.38+.
# Ubuntu 22.04 has glibc 2.35, so we link against the pip onnxruntime
# shared library instead.
#
# Usage:
#   ./scripts/test.sh                    # run all tests, debug profile (GPU preferred)
#   ./scripts/test.sh --release          # run all tests, release profile
#   ./scripts/test.sh -p sparrow-engine-cpu --lib  # pass extra cargo test args
#   ./scripts/test.sh --release -p sparrow-engine-cpu --lib  # combine
#   ORT_DIR=/custom/path ./scripts/test.sh  # override ORT location
#   PYO3_PYTHON=/path/to/python3.11 ./scripts/test.sh  # pin the interpreter for the sparrow-engine-python tests

set -euo pipefail

# Shared ORT discovery: sets ORT_CAPI, ORT_LIB_LOCATION, ORT_PREFER_DYNAMIC_LINK, LD_LIBRARY_PATH.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/tests/test_download_models.sh"
"$SCRIPT_DIR/tests/test_check_ffi_symbols.sh"
"$SCRIPT_DIR/tests/test_installers.sh"
# Contract for the manylinux GPU wheel build path (release-locked builder recipe,
# build_all dispatch, no auditwheel escape, docs not overclaiming). Lightweight —
# does NOT run the Docker build.
"$SCRIPT_DIR/tests/test_gpu_wheel_build_contract.sh"
# Guards that retired public model aliases have not resurfaced in the CLI, the
# nvJPEG dlopen script, or the public user manual (current catalog IDs only).
"$SCRIPT_DIR/tests/test_catalog_id_contract.sh"
# Guards manual_test_setup.sh against the SIGPIPE-under-pipefail early-consumer
# regression (find | sort | head -1 / find | grep -q) and its sourced-only /
# no-shell-option-leak / no-internal-leak invariants. Lightweight — static +
# tiny-fixture idiom checks; does NOT run the Cargo build.
"$SCRIPT_DIR/tests/test_manual_setup_contract.sh"
# Guards ORT auto-discovery (CUDA-runtime compatibility filtering). Runs before
# ort-env.sh is sourced because it validates the very selection logic that
# source performs.
"$SCRIPT_DIR/tests/test_ort_env.sh"
# Keep this before ort-env.sh: the catalogue smoke test invokes python3, which
# must not inherit ORT/CUDA loader paths. /dev/null skips on-disk model loading.
SPARROW_CATALOG='' "$SCRIPT_DIR/smoke_test_models.sh" --model-dir /dev/null
source "$SCRIPT_DIR/ort-env.sh"

# sparrow-engine-python tests (--no-default-features) link to libpython, and
# pyo3 compiles that test binary against the interpreter it discovers. Both
# must be the SAME interpreter, and its libpython must be loadable on this
# host: a PATH-first Homebrew Python built against a newer glibc (GLIBC_2.38
# on a glibc 2.35 host) makes the test binary fail to start before running a
# single test (v0.1.30 manual run, finding S00-F02, 2026-09-21).
# Selection order: PYO3_PYTHON (explicit pin) > uv-managed Python 3.11 (the
# documented prerequisite floor) > any uv-found >= 3.11 > python3 on PATH.
if [[ -z "${PYO3_PYTHON:-}" ]]; then
    PYO3_PYTHON="$(uv python find 3.11 2>/dev/null || uv python find '>=3.11' 2>/dev/null || command -v python3 || true)"
fi
PYLIB_DIR=""
if [[ -n "${PYO3_PYTHON:-}" ]]; then
    export PYO3_PYTHON
    # The embedded interpreter locates its stdlib from the `python3` found on
    # PATH at run time; make that the selected one, or a PATH-first foreign
    # Python yields `init_fs_encoding ... No module named 'encodings'`.
    PYO3_BIN_DIR="$(dirname "$PYO3_PYTHON")"
    export PATH="$PYO3_BIN_DIR:$PATH"
    echo "sparrow-engine-python test interpreter: $PYO3_PYTHON ($("$PYO3_PYTHON" --version 2>&1))"
    PYLIB_DIR="$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")' 2>/dev/null || true)"
fi
if [[ -n "$PYLIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${ORT_CAPI}:${PYLIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Enforce: forbid eprintln!/println! in sparrow-engine-python/src/lib.rs (Phase 3.5 S6).
# Load-bearing — every former site is now tracing::warn!/tracing::info!; a new
# eprintln!/println! fails this script and the test pipeline. Set
# SPARROW_ENGINE_PY_GUARD_STRICT=0 to demote to warn for transient debugging.
"$SCRIPT_DIR/guard_no_print.sh"

# Extract --release from args so cargo sees it as a build flag, not a test-binary
# flag (cargo test routes anything after '--' to the test binary).
CARGO_FLAGS=()
CARGO_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--release" ]]; then
        CARGO_FLAGS+=("--release")
    else
        CARGO_ARGS+=("$arg")
    fi
done

# If no non-flag args are given, run the full suite. CPU, GPU, and mobile all
# publish a library named `sparrow_engine`, so GPU and mobile use isolated
# target dirs instead of sharing the workspace target and racing on the same
# rlib/cdylib filenames. The GPU package additionally compiles
# `sparrow-engine-cpu` as a dev-dependency; because it too is named
# `sparrow_engine`, its cdylib collides with the GPU crate's at the same output
# path. Isolated target dirs alone do not fix that intra-package collision, so
# the GPU `cargo test` below builds with a single job (`--jobs 1`) — see there.
#
# Handle sparrow-engine-python specially:
# it has `default = ["extension-module"]`, which asks pyo3 NOT to link libpython
# (symbols come from the host Python at runtime). `cargo test` builds a standalone
# lib-test binary that then cannot link — rust-lld errors on Py_* symbols.
# Workaround: run it separately without `extension-module`, while explicitly
# retaining the CPU engine feature, so pyo3 links libpython for the test binary.
# --test-threads=1 because the ORT engine is a process-global singleton.
if [[ ${#CARGO_ARGS[@]} -eq 0 ]]; then
    LITERT_DIR="${LITERT_LIB_DIR:-$SCRIPT_DIR/../artifacts}"
    if [[ ! -f "$LITERT_DIR/libLiteRt.so" ]]; then
        echo "ERROR: full tests require libLiteRt.so; set LITERT_LIB_DIR to its directory." >&2
        exit 1
    fi
    MOBILE_FFI_TARGET_DIR="${SPARROW_ENGINE_MOBILE_FFI_TARGET_DIR:-$SCRIPT_DIR/../target/mobile-ffi}"
    echo "Running: mobile FFI built-export gate"
    echo "---"
    LITERT_LIB_DIR="$LITERT_DIR" CARGO_TARGET_DIR="$MOBILE_FFI_TARGET_DIR" \
        cargo build -p sparrow-engine-mobile --release --features ffi
    LITERT_LIB_DIR="$LITERT_DIR" CARGO_TARGET_DIR="$MOBILE_FFI_TARGET_DIR" \
        cargo test -p sparrow-engine-mobile --release --features ffi \
        --test integration_ffi_symbols
    echo "---"

    MOBILE_TEST_TARGET_DIR="${SPARROW_ENGINE_MOBILE_TEST_TARGET_DIR:-$SCRIPT_DIR/../target/test-mobile}"
    echo "Running: isolated sparrow-engine-mobile tests ${CARGO_FLAGS[*]:-}"
    echo "---"
    LITERT_LIB_DIR="$LITERT_DIR" \
        LD_LIBRARY_PATH="$LITERT_DIR:${LD_LIBRARY_PATH:-}" \
        CARGO_TARGET_DIR="$MOBILE_TEST_TARGET_DIR" \
        cargo test -p sparrow-engine-mobile \
        ${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"} -- --test-threads=1
    echo "---"

    GPU_TEST_TARGET_DIR="${SPARROW_ENGINE_GPU_TEST_TARGET_DIR:-$SCRIPT_DIR/../target/test-gpu}"
    echo "Running: isolated sparrow-engine-gpu tests ${CARGO_FLAGS[*]:-}"
    echo "---"
    # `--jobs 1`: serialize the build so the `sparrow-engine-cpu` dev-dependency
    # (also lib name `sparrow_engine`) compiles before the GPU crate, leaving the
    # GPU `libsparrow_engine` as the deterministic last writer at the shared
    # output path. Without it, parallel Cargo jobs can leave the CPU artifact
    # there and the GPU integration tests fail to resolve GPU-only modules
    # (audio, kernels, models, decode). The full, unfiltered GPU test set
    # (lib + integration tests + examples + doctests) still runs.
    CARGO_TARGET_DIR="$GPU_TEST_TARGET_DIR" \
        cargo test --jobs 1 -p sparrow-engine-gpu \
        ${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"} -- --test-threads=1
    echo "---"

    echo "Running: cargo test --workspace --exclude sparrow-engine-python --exclude sparrow-engine-gpu --exclude sparrow-engine-mobile ${CARGO_FLAGS[*]:-} -- --test-threads=1"
    echo "---"
    cargo test --workspace \
        --exclude sparrow-engine-python \
        --exclude sparrow-engine-gpu \
        --exclude sparrow-engine-mobile \
        ${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"} -- --test-threads=1
    echo "---"
    echo "Running: cargo test -p sparrow-engine-python --lib --no-default-features --features cpu ${CARGO_FLAGS[*]:-} -- --test-threads=1"
    echo "---"
    exec cargo test -p sparrow-engine-python --lib --no-default-features --features cpu \
        ${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"} -- --test-threads=1
else
    echo "Running: cargo test ${CARGO_FLAGS[*]:-} ${CARGO_ARGS[*]}"
    echo "---"
    exec cargo test ${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"} "${CARGO_ARGS[@]}"
fi
