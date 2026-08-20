#!/usr/bin/env bash
# scripts/build_gpu_wheel_manylinux.sh
#
# Build the release-quality manylinux_2_28 GPU Python wheel (sparrow-engine-gpu)
# inside the release-locked Rocky 8 / CUDA 12.8 container, exactly matching the
# `.github/workflows/release.yml` `build-gpu-linux` job.
#
# Why this exists
#   A raw host `SPARROW_ENGINE_FLAVOR=gpu ./build.sh` on a newer-glibc host
#   (e.g. Ubuntu 22.04, glibc 2.35) compiles a `linux_x86_64` wheel and then
#   fails the hard `auditwheel repair --plat manylinux_2_28_x86_64` step inside
#   build.sh — auditwheel cannot relabel symbols newer than glibc 2.28. The only
#   correct portable-wheel path is to compile under the release-locked Rocky 8
#   (glibc 2.28) environment. This helper does exactly that; it does NOT weaken,
#   skip, or make auditwheel non-fatal.
#
# Safety
#   - Source-tree-safe: builds a copy of the CURRENT tree (including uncommitted
#     content) in a run-owned scratch dir under `sparrow-engine/target/`; never
#     builds against or mutates the primary source tree.
#   - Runs the container with `--rm`, no host GPU, no `--privileged`, no sudo.
#     Build steps run as the container's root (needed for dnf); an EXIT chown in
#     the container returns the bind-mounted tree to the host uid/gid so no
#     root-owned host artifacts remain.
#   - On success, copies ONLY the finished wheel into the primary
#     `sparrow-engine/target/wheels/` and removes only the exact scratch dir.
#   - On failure, surfaces the container/build error and leaves the primary
#     source unmodified.
#
# Locked release environment (keep in sync with release.yml build-gpu-linux):
#   image   nvidia/cuda:12.8.1-cudnn-devel-rockylinux8   (glibc 2.28)
#   rust    1.96.0
#   python  3.11
#   tools   auditwheel>=6.0.0, patchelf>=0.14, uv, maturin
set -euo pipefail

# --- locked release environment ---
BUILDER_IMAGE="nvidia/cuda:12.8.1-cudnn-devel-rockylinux8"
RUST_VERSION="1.96.0"
AUDITWHEEL_SPEC="auditwheel>=6.0.0"
PATCHELF_SPEC="patchelf>=0.14"
WHEEL_TAG_SUBSTR="cp311-abi3-manylinux_2_28_x86_64"

log() { echo "[build_gpu_wheel_manylinux] $*"; }
err() { echo "[build_gpu_wheel_manylinux] ERROR: $*" >&2; }

# --- resolve paths (cwd-independent) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARROW_ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # .../sparrow-engine
PY_DIR="$SPARROW_ENGINE_DIR/sparrow-engine-python"
WHEELS_DIR="$SPARROW_ENGINE_DIR/target/wheels"

# --- preflight ---
[ "$(uname -s)" = "Linux" ] || { err "Linux x86_64 host required (got $(uname -s)); the manylinux GPU wheel builds only on Linux"; exit 1; }
case "$(uname -m)" in
    x86_64|amd64) : ;;
    *) err "x86_64 host required (got $(uname -m))"; exit 1 ;;
esac
command -v docker >/dev/null 2>&1 || { err "docker not found on PATH"; exit 1; }
docker info >/dev/null 2>&1 || { err "docker daemon not reachable (start Docker and retry)"; exit 1; }
command -v tar >/dev/null 2>&1 || { err "tar not found on PATH"; exit 1; }
[ -f "$PY_DIR/build.sh" ] || { err "canonical build.sh not found at $PY_DIR/build.sh"; exit 1; }

if ! docker image inspect "$BUILDER_IMAGE" >/dev/null 2>&1; then
    log "base image $BUILDER_IMAGE not cached; pulling (network required)..."
    docker pull "$BUILDER_IMAGE" || { err "failed to pull $BUILDER_IMAGE"; exit 1; }
fi

# --- run-owned scratch dir under target/ (never touches the primary source) ---
mkdir -p "$SPARROW_ENGINE_DIR/target"
SCRATCH="$(mktemp -d "$SPARROW_ENGINE_DIR/target/manylinux-gpu-build.XXXXXX")"
SRC="$SCRATCH/src"
CONTAINER_SCRIPT="$SCRATCH/container_build.sh"
mkdir -p "$SRC"

# Guard: only ever recursively delete a path that is non-empty, canonically
# resolved to a directory DIRECTLY under $SPARROW_ENGINE_DIR/target/, and whose
# basename matches the run-owned scratch pattern. Returns nonzero (refuses)
# otherwise, and surfaces a real delete failure rather than hiding it.
_safe_rm_scratch() {
    local path="$1" resolved target_root
    [ -n "$path" ] || { err "cleanup: empty scratch path — refusing recursive delete"; return 2; }
    [ -e "$path" ] || return 0                     # already gone — nothing to remove
    target_root="$(cd "$SPARROW_ENGINE_DIR/target" 2>/dev/null && pwd -P)" \
        || { err "cleanup: cannot resolve target root '$SPARROW_ENGINE_DIR/target'"; return 2; }
    resolved="$(cd "$path" 2>/dev/null && pwd -P)" \
        || { err "cleanup: cannot resolve scratch path '$path'"; return 2; }
    if [ "$(dirname "$resolved")" != "$target_root" ]; then
        err "cleanup: refusing to recursively delete '$resolved' — not directly under $target_root"
        return 2
    fi
    case "$(basename "$resolved")" in
        manylinux-gpu-build.*) : ;;
        *) err "cleanup: refusing to recursively delete '$resolved' — basename is not manylinux-gpu-build.*"; return 2 ;;
    esac
    rm -rf "$resolved" || return 1                 # surface a real delete failure
    return 0
}

# EXIT cleanup: capture the original exit status, disable this trap, attempt the
# guarded removal, and make any cleanup failure EXPLICIT + nonzero (an interrupted
# container can leave root-owned files that rm cannot remove). Never silently
# swallow the failure; preserve the original build failure when cleanup succeeds.
cleanup() {
    local rc=$?
    trap - EXIT
    if _safe_rm_scratch "$SCRATCH"; then
        exit "$rc"                                 # cleanup ok: keep the original rc
    fi
    err "cleanup FAILED for scratch dir '$SCRATCH' — remove it manually (an interrupted container may have left root-owned files): rm -rf '$SCRATCH'"
    if [ "$rc" -ne 0 ]; then
        exit "$rc"                                 # build AND cleanup failed: both errors visible; keep build rc
    fi
    exit 3                                         # build ok but cleanup failed: nonzero
}
trap cleanup EXIT

# --- copy the CURRENT source (incl. uncommitted) minus target*/.git/.venv ---
log "copying current engine source into run-owned scratch (excluding target*, .git, .venv)..."
tar -C "$SPARROW_ENGINE_DIR" \
    --exclude='./target' \
    --exclude='./target-*' \
    --exclude='./.git' \
    --exclude='*/.venv' \
    -cf - . | tar -C "$SRC" -xf -

# --- generate the container-side build script (matches release.yml) ---
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
cat > "$CONTAINER_SCRIPT" <<CEOF
#!/usr/bin/env bash
set -euo pipefail
# Guarantee no root-owned host artifacts remain: chown the bind-mounted build
# tree back to the host uid/gid on exit (success OR failure).
trap 'chown -R ${HOST_UID}:${HOST_GID} /build 2>/dev/null || true' EXIT

# Rocky 8 / RHEL 8 build prerequisites (match release.yml build-gpu-linux).
dnf install -y --setopt=install_weak_deps=False \\
    ca-certificates curl git gcc gcc-c++ make pkgconfig \\
    python3.11 python3.11-devel python3.11-pip
ln -sf /usr/bin/python3.11 /usr/local/bin/python3
ln -sf /usr/bin/python3.11 /usr/local/bin/python

# auditwheel >=6 (manylinux_2_28 policy) + patchelf >=0.14 (repair needs it),
# from PyPI so patchelf is newer than Rocky 8's EPEL v0.12.
python3 -m pip install --user --upgrade "${AUDITWHEEL_SPEC}" "${PATCHELF_SPEC}"

# Rust, pinned to the release lock.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \\
    | sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal

# uv + maturin (match release.yml).
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:/root/.cargo/bin:\$PATH"
uv tool install maturin
export PATH="/root/.local/bin:/root/.cargo/bin:\$PATH"

# The UNCHANGED hard path: build.sh compiles the GPU wheel and runs
# \`auditwheel repair --plat manylinux_2_28_x86_64\`. Do NOT alter this recipe.
cd /build/sparrow-engine-python
SPARROW_ENGINE_FLAVOR=gpu ./build.sh
CEOF

# --- run the container (--rm, no GPU, no privileged, no sudo) ---
log "compiling the GPU wheel inside $BUILDER_IMAGE (full crate build; several minutes)..."
docker run --rm \
    -v "$SRC:/build" \
    -v "$CONTAINER_SCRIPT:/container_build.sh:ro" \
    -w /build \
    "$BUILDER_IMAGE" \
    bash /container_build.sh

# --- locate + verify exactly one freshly built GPU wheel ---
shopt -s nullglob
built_wheels=( "$SRC/target/wheels/sparrow_engine_gpu-"*.whl )
shopt -u nullglob
if [ "${#built_wheels[@]}" -eq 0 ]; then
    err "no sparrow_engine_gpu-*.whl was produced in the container"
    exit 1
fi
if [ "${#built_wheels[@]}" -gt 1 ]; then
    err "expected exactly one GPU wheel, found ${#built_wheels[@]}: ${built_wheels[*]}"
    exit 1
fi
built_wheel="${built_wheels[0]}"
wheel_base="$(basename "$built_wheel")"

case "$wheel_base" in
    *"$WHEEL_TAG_SUBSTR"*) : ;;
    *) err "wheel '$wheel_base' does not carry the required tag '$WHEEL_TAG_SUBSTR'"; exit 1 ;;
esac

# METADATA assertions read straight from the wheel zip.
meta="$(python3 - "$built_wheel" <<'PYEOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    names = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
    if not names:
        print("NO_METADATA")
    else:
        sys.stdout.write(z.read(names[0]).decode("utf-8", "replace"))
PYEOF
)"
printf '%s\n' "$meta" | grep -qE '^Name:[[:space:]]*sparrow-engine-gpu[[:space:]]*$' \
    || { err "METADATA Name is not 'sparrow-engine-gpu'"; exit 1; }
printf '%s\n' "$meta" | grep -qE '^Requires-Dist:[[:space:]]*onnxruntime-gpu([[:space:]<>=!~;].*)?$' \
    || { err "METADATA missing 'Requires-Dist: onnxruntime-gpu'"; exit 1; }
printf '%s\n' "$meta" | grep -qE '^Provides-Dist:[[:space:]]*sparrow-engine[[:space:]]*$' \
    || { err "METADATA missing 'Provides-Dist: sparrow-engine'"; exit 1; }

# --- copy ONLY the finished wheel into the primary target/wheels/ ---
mkdir -p "$WHEELS_DIR"
cp "$built_wheel" "$WHEELS_DIR/"
final_wheel="$WHEELS_DIR/$wheel_base"

log "GPU wheel built + verified (release-locked manylinux_2_28):"
printf '%s\n' "$final_wheel"
