#!/usr/bin/env bash
# Lightweight contract test for the manylinux GPU wheel build path.
#
# It does NOT run the expensive Docker build — the real end-to-end proof is the
# manual rerun of scripts/build_gpu_wheel_manylinux.sh. This test asserts the
# release-locked builder recipe, the build_all_flavors dispatch, the absence of
# any auditwheel-escape, and that the public packaging docs do not overclaim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARROW_ENGINE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # .../sparrow-engine
REPO_ROOT="$(cd "$SPARROW_ENGINE_DIR/.." && pwd)"

HELPER="$SPARROW_ENGINE_DIR/scripts/build_gpu_wheel_manylinux.sh"
BUILD_ALL="$SPARROW_ENGINE_DIR/scripts/build_all_flavors.sh"
BUILD_SH="$SPARROW_ENGINE_DIR/sparrow-engine-python/build.sh"
PYPROJECT="$SPARROW_ENGINE_DIR/sparrow-engine-python/pyproject.toml"
PY_README="$SPARROW_ENGINE_DIR/sparrow-engine-python/README.md"
USER_MANUAL="$REPO_ROOT/docs/user-manual.md"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "[1] touched scripts parse (bash -n)"
for s in "$HELPER" "$BUILD_ALL" "$BUILD_SH"; do
    [ -f "$s" ] || fail "missing script: $s"
    bash -n "$s" || fail "syntax error in $s"
done

echo "[2] helper pins the exact release-locked builder environment (matches release.yml)"
grep -Fq 'nvidia/cuda:12.8.1-cudnn-devel-rockylinux8' "$HELPER" || fail "helper missing locked base image nvidia/cuda:12.8.1-cudnn-devel-rockylinux8"
grep -Eq '1\.96\.0' "$HELPER"                     || fail "helper missing locked Rust 1.96.0"
grep -Fq 'cp311-abi3-manylinux_2_28_x86_64' "$HELPER" || fail "helper missing manylinux_2_28 wheel-tag assertion"
grep -Eq 'auditwheel>=6' "$HELPER"                || fail "helper missing auditwheel>=6"
grep -Eq 'patchelf>=0\.14' "$HELPER"              || fail "helper missing patchelf>=0.14"
grep -Eq 'Name:.*sparrow-engine-gpu' "$HELPER" || fail "helper missing Name sparrow-engine-gpu assertion"
grep -Eq 'Requires-Dist:.*onnxruntime-gpu' "$HELPER" || fail "helper missing Requires-Dist onnxruntime-gpu assertion"
grep -Eq 'Provides-Dist:.*sparrow-engine' "$HELPER"  || fail "helper missing Provides-Dist sparrow-engine assertion"

echo "[3] helper runs the UNCHANGED build.sh hard path (no duplicated auditwheel/metadata logic)"
grep -Eq 'SPARROW_ENGINE_FLAVOR=gpu[[:space:]]+\./build\.sh' "$HELPER" \
    || fail "helper does not invoke the unchanged 'SPARROW_ENGINE_FLAVOR=gpu ./build.sh'"
# The helper must not INVOKE 'auditwheel repair' itself (build.sh owns it).
# Strip comment lines first so the docstring's descriptive mention is ignored.
if grep -vE '^[[:space:]]*#' "$HELPER" | grep -Eq 'auditwheel[[:space:]]+repair'; then
    fail "helper must not duplicate 'auditwheel repair' — build.sh owns it"
fi

echo "[4] helper is source-tree-safe (run-owned scratch under target/, --rm, no privileged/GPU, chown-back)"
grep -Eq 'mktemp -d .*/target/' "$HELPER"   || fail "helper does not create a run-owned scratch dir under target/"
grep -Fq 'docker run --rm' "$HELPER"        || fail "helper does not use 'docker run --rm'"
# Check the actual command lines (not the docstring) for forbidden flags.
helper_code="$(grep -vE '^[[:space:]]*#' "$HELPER")"
printf '%s\n' "$helper_code" | grep -Fq -- '--privileged' && fail "helper must not use --privileged"
printf '%s\n' "$helper_code" | grep -Fq -- '--gpus'       && fail "helper must not request host GPU (--gpus)"
grep -Eq 'chown -R' "$HELPER"               || fail "helper missing EXIT chown-back to host uid/gid"

echo "[5] build_all_flavors dispatches Linux GPU to the helper and CPU to build.sh"
grep -Fq 'build_gpu_wheel_manylinux.sh' "$BUILD_ALL" || fail "build_all_flavors does not call the manylinux helper"
grep -Eq 'SPARROW_ENGINE_FLAVOR=cpu[[:space:]]+\./build\.sh' "$BUILD_ALL" || fail "build_all_flavors does not build the CPU wheel via local build.sh"
grep -A5 'Linux)' "$BUILD_ALL" | grep -Fq 'build_gpu_wheel_manylinux.sh' \
    || fail "build_all_flavors Linux GPU branch does not dispatch to the manylinux helper"

echo "[6] the GPU wheel path keeps auditwheel as a HARD gate (no skip/non-fatal escape)"
grep -Eq 'auditwheel[[:space:]]+repair' "$BUILD_SH"   || fail "build.sh no longer runs 'auditwheel repair' for the GPU wheel"
grep -Fq 'manylinux_2_28_x86_64' "$BUILD_SH"          || fail "build.sh GPU repair no longer targets --plat manylinux_2_28_x86_64"
if grep -Eiq 'SPARROW_ENGINE_(SKIP|NO)_AUDITWHEEL|--skip-auditwheel|AUDITWHEEL_SKIP' "$HELPER" "$BUILD_ALL" "$BUILD_SH"; then
    fail "an auditwheel skip/escape flag was introduced"
fi
if grep -Eiq -- '--auditwheel[[:space:]]+skip' "$HELPER"; then
    fail "helper passes '--auditwheel skip' (would defeat the manylinux gate)"
fi

echo "[7] public packaging docs do not overclaim (no mechanical refusal / no implemented Conflicts-Dist)"
for d in "$BUILD_SH" "$PYPROJECT" "$PY_README" "$USER_MANUAL"; do
    [ -f "$d" ] || continue
    if grep -Eiq '(pip[[:space:]]+)?refuses[[:space:]]+(to[[:space:]]+install[[:space:]]+)?both|makes pip refuse|CANNOT coexist' "$d"; then
        fail "$(basename "$d") overclaims mechanical refusal (pip refuses both / cannot coexist)"
    fi
    if grep -Fq 'Conflicts-Dist' "$d"; then
        # Any Conflicts-Dist mention must be disclaimed (advisory / not / only / no).
        if grep -F 'Conflicts-Dist' "$d" | grep -Eivq 'not|only|advisory|no Conflicts|isn'; then
            fail "$(basename "$d") mentions Conflicts-Dist without disclaiming it as unimplemented"
        fi
    fi
done

echo "[8] helper's recursive delete is guarded (validates scratch; no silent 'rm -rf ... || true')"
if grep -Eq 'rm -rf[^#]*\|\|[[:space:]]*true' "$HELPER"; then
    fail "helper hides a recursive-delete failure with '|| true'"
fi
if grep -Eq 'rm -rf[^#]*2>/dev/null' "$HELPER"; then
    fail "helper silences a recursive-delete with 2>/dev/null"
fi
grep -Fq 'manylinux-gpu-build.*)' "$HELPER" || fail "helper cleanup does not confine the delete to a run-owned manylinux-gpu-build.* scratch dir"
grep -Fq 'pwd -P' "$HELPER"       || fail "helper cleanup does not canonically resolve the scratch path before deleting"
grep -Fq 'trap - EXIT' "$HELPER"  || fail "helper cleanup does not disable its own EXIT trap before deleting"
grep -Eiq 'cleanup FAILED|refusing to recursively delete' "$HELPER" \
    || fail "helper cleanup does not emit an explicit failure/refusal error"

echo "PASS: manylinux GPU wheel build contract (release-locked env, dispatch, no auditwheel escape, docs not overclaiming)"
