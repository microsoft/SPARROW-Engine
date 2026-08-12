#!/usr/bin/env bash
# DEVELOPMENT ONLY — not needed by end users.
#
# This script exists because our dev machine (Ubuntu 22.04, glibc 2.35) can't
# use ORT's static lib (needs glibc 2.38+). We link dynamically against the
# pip onnxruntime-gpu package instead. This script finds that package and sets
# the linker paths cargo needs.
#
# End users don't need this:
#   - Docker: ORT is bundled in the container image
#   - CLI release tarball (RP-4 2026-05-26): bundles `libonnxruntime.so.X.Y.Z`
#     under `lib/` and resolves it via the in-binary
#     `ort_resolver::init_ort_env()` shim; no `LD_LIBRARY_PATH` shell setup
#     required. See `installer/sparrow-engine-install.sh --cli`.
#   - Python wheel (RP-3 2026-05-23): the `_discover_ort_dylib()` shim in
#     `sparrow_engine.__init__` sets `ORT_DYLIB_PATH` from the pip
#     `onnxruntime[-gpu]` install at import time.
#
# GPU is the default. Prefers onnxruntime-gpu over onnxruntime-cpu.
# Sets: ORT_CAPI, ORT_LIB_LOCATION, ORT_PREFER_DYNAMIC_LINK, LD_LIBRARY_PATH.
#
# Usage:
#   source scripts/ort-env.sh                          # from sparrow-engine/
#   source "$(dirname "$0")/../scripts/ort-env.sh"     # from tools/
#   source "$(dirname "$0")/ort-env.sh"                # from scripts/

# NOTE: No `set -euo pipefail` here — this file is sourced into the caller's
# shell. Setting -u would break zsh plugins (e.g., zvm_update_cursor) that
# reference unset variables. Callers (test.sh, etc.) set their own options.

# Find ORT shared library from pip onnxruntime package.
# Prefers GPU (onnxruntime-gpu) over CPU (onnxruntime).
#
# Multiple onnxruntime archives can coexist in the uv cache (e.g. a project
# bumps from 1.24.4 → 1.25.0 → 1.25.1 over time). `find -print -quit` returns
# the first match in filesystem-traversal order, which is non-deterministic
# across calls. That makes builds and runs pick different ORT versions and
# produces "version `VERS_1.X.Y' not found" at runtime. Pick the newest
# version deterministically by reading the ELF symbol-version table.
#
# Drop `archive-v0` segment so this survives a future uv cache-format bump
# (uv has different vN per subdir already: archive-v0, environments-v2,
# interpreter-v4, sdists-v9, simple-v18 on this dev box). The `-path "$1"`
# filter (e.g. `*/onnxruntime/capi/libonnxruntime_providers_cuda.so`) is
# specific enough to identify ORT capi dirs regardless of which uv subdir
# layout houses them. Mirrors the analogous fix in pick_newest_cudnn_dir
# below + setup.sh wheel-cache cleanup. Without this fix, a uv cache
# format bump silently degrades us to the python3 fallback at lines 92-102,
# losing the version-deterministic ELF-symbol read this function exists for.
pick_newest_ort_dir() {
    # Soft-fail if `strings` (binutils) is missing — without it we cannot
    # extract VERS_ symbols, every archive becomes a silent skip, and the
    # caller falls through to python3 with no diagnostic. Warn once.
    if ! command -v strings >/dev/null 2>&1; then
        echo >&2 "warn: 'strings' (binutils) not found — ORT version detection disabled; install with 'apt-get install binutils' or equivalent"
        return
    fi
    # NOTE: `find … | while … done` runs the while body in a subshell under
    # bash (per Pipelines manual §3.2.3) but in the parent shell under zsh.
    # This loop body only `printf`s to stdout (consumed by the trailing
    # sort|head|awk), so the divergence is invisible. DO NOT add cross-
    # iteration state (counters, "last seen" vars) — those would silently
    # be zero/empty under bash. See bash manual + BashFAQ #024.
    find "$HOME/.cache/uv" -path "$1" -print 2>/dev/null |
        while IFS= read -r p; do
            d=$(dirname "$p")
            # Use `find` not shell glob to stay safe under zsh's NO_MATCH
            # default. `sort -V` defensive in case a future ORT release ships
            # multiple .so files in one archive (debug variant, etc.) — pick
            # the highest.
            real=$(find "$d" -maxdepth 1 -name 'libonnxruntime.so.*.*.*' -type f 2>/dev/null | sort -V | tail -1)
            [[ -z "$real" ]] && continue
            v=$(strings "$real" 2>/dev/null |
                grep -E '^VERS_1\.[0-9]+\.[0-9]+' |
                sort -V | tail -1)
            if [[ -n "$v" ]]; then
                printf '%s %s\n' "$v" "$d"
            else
                # Custom/community ORT build without --version-script. Microsoft
                # pip wheels always export VERS_, so this is rare — but warn
                # rather than silently exclude the archive from selection.
                echo >&2 "warn: $real has no VERS_ symbols (custom ORT build?); excluding from version selection"
            fi
        done |
        sort -V -r | head -1 | awk '{print $2}'
}

ort_dir_has_runtime_lib() {
    local dir="$1"
    local versioned
    if [[ -f "$dir/libonnxruntime.so" ]]; then
        return 0
    fi
    versioned=$(find "$dir" -maxdepth 1 -name 'libonnxruntime.so.*.*.*' -type f -print -quit 2>/dev/null)
    [[ -n "$versioned" ]]
}

# ---------------------------------------------------------------------------
# Host CUDA-runtime compatibility filtering for auto-discovered GPU ORT builds.
#
# WHY: pick_newest_ort_dir ranks cached onnxruntime-gpu builds purely by the ELF
# VERS_ symbol version. Newer ORT wheels are built against newer CUDA majors —
# onnxruntime-gpu 1.28 NEEDs libcublasLt.so.13 / libcudart.so.13 (CUDA 13),
# while a CUDA-12 host exposes only *.so.12. Picking the newest build then
# succeeds at discovery but fails at model load with
# "libcublasLt.so.13: cannot open shared object file". The loader only cares
# whether the provider's CUDA DT_NEEDED sonames resolve on this host, so filter
# GPU candidates on exactly that BEFORE version ranking.
#
# Explicit ORT_DIR stays explicit (handled at the top of find_ort_dir): this
# filter governs automatic discovery only.
# ---------------------------------------------------------------------------

# Print the DT_NEEDED sonames of an ELF file, one per line. Prefers readelf,
# falls back to objdump. Returns 2 when neither tool exists so the caller can
# degrade gracefully instead of silently trusting an unverified build.
ort_elf_needed_sonames() {
    local so="$1"
    if command -v readelf >/dev/null 2>&1; then
        # `|| true` neutralises a non-zero readelf (e.g. non-ELF input) under
        # the caller's set -e/pipefail; empty output is handled downstream.
        readelf -d "$so" 2>/dev/null |
            sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p' || true
        return 0
    fi
    if command -v objdump >/dev/null 2>&1; then
        objdump -p "$so" 2>/dev/null |
            awk '$1 == "NEEDED" { print $2 }' || true
        return 0
    fi
    return 2
}

# True when the tooling needed to verify GPU CUDA compatibility is present:
# an ldconfig host oracle AND an ELF NEEDED reader (readelf or objdump). When
# false, find_ort_dir degrades to unfiltered newest-version selection so a host
# that currently works is never made worse.
ort_can_check_cuda_compat() {
    command -v ldconfig >/dev/null 2>&1 || return 1
    command -v readelf >/dev/null 2>&1 || command -v objdump >/dev/null 2>&1 || return 1
    return 0
}

# True when $1 (a bare soname such as libcublasLt.so.13) resolves on this host's
# dynamic loader: present in the ldconfig cache (default search path), under a
# directory already on LD_LIBRARY_PATH (covers pip nvidia-* wheels not yet in
# the cache), or under one of the common CUDA toolkit lib dirs that ort-env.sh
# itself later adds to LD_LIBRARY_PATH. Checking those dirs here keeps discovery
# in step with the runtime loader path and avoids false-rejecting a build whose
# CUDA lib lives under /usr/local/cuda but is not registered in the ldconfig
# cache. $2, when non-empty, is a pre-captured `ldconfig -p` first-field list,
# reused across lookups to avoid re-forking ldconfig per soname.
ort_host_has_soname() {
    local soname="$1"
    local ldcache="${2:-}"
    if [[ -z "$ldcache" ]] && command -v ldconfig >/dev/null 2>&1; then
        ldcache=$(ldconfig -p 2>/dev/null | awk '{ print $1 }' || true)
    fi
    if [[ -n "$ldcache" ]] && printf '%s\n' "$ldcache" | grep -qxF "$soname"; then
        return 0
    fi
    # Scan LD_LIBRARY_PATH and the common CUDA loader dirs via process
    # substitution rather than unquoted word splitting: zsh does not IFS-split
    # scalar expansions by default, so a `for dir in $LD_LIBRARY_PATH` loop
    # would misbehave when sourced in zsh. The fixed dirs mirror the locations
    # the cuDNN/CUDA blocks below feed into EXTRA_LIB_PATHS/LD_LIBRARY_PATH.
    local dir
    while IFS= read -r dir; do
        if [[ -n "$dir" && -e "$dir/$soname" ]]; then
            return 0
        fi
    done < <(
        # Trailing newline is required: without it the last LD_LIBRARY_PATH
        # entry would concatenate with the first fixed dir into one bad line.
        printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n'
        printf '%s\n' \
            /usr/local/cuda/lib64 \
            /usr/local/cuda/targets/x86_64-linux/lib \
            /usr/lib/x86_64-linux-gnu
    )
    return 1
}

# True when every CUDA-family soname the GPU provider .so DT_NEEDEDs resolves on
# this host. Non-CUDA sonames (libc, libstdc++, the libcuda.so.1 driver) are
# ignored: they are always present or ABI-stable. cuDNN is deliberately ignored
# too (see the case below). $2 is an optional pre-captured ldconfig-cache list.
# Returns 2 when NEEDED extraction tooling is missing.
#
# Fails CLOSED: a corrupt / non-ELF / unreadable provider yields no DT_NEEDED
# entries (or none in the CUDA family). Such a candidate is never treated as
# compatible — better to skip it than to select a build we could not verify.
ort_provider_cuda_compatible() {
    local provider_so="$1"
    local ldcache="${2:-}"
    local needed
    needed=$(ort_elf_needed_sonames "$provider_so") || return 2
    # No DT_NEEDED entries at all -> extraction failed or the file is not a
    # usable ELF provider. Fail closed.
    [[ -z "$needed" ]] && return 1
    local soname
    local saw_cuda=0
    while IFS= read -r soname; do
        [[ -z "$soname" ]] && continue
        case "$soname" in
            # cuDNN is resolved SEPARATELY by ort-env.sh (pick_newest_cudnn_dir
            # -> EXTRA_LIB_PATHS) AFTER selection, so it is not on the loader
            # path at discovery time and must NOT gate GPU-build selection —
            # otherwise every compatible cu12 build (which NEEDs libcudnn.so.9)
            # would be wrongly rejected.
            libcudnn.so.*) continue ;;
            # CUDA toolkit runtime/math libs: shipped by the system CUDA install
            # and visible via ldconfig. Their soname major encodes the CUDA
            # major, so a mismatch here (e.g. libcublasLt.so.13 on a CUDA-12
            # host) is exactly the incompatibility to screen out before ranking.
            libcudart.so.*|libcublas.so.*|libcublasLt.so.*|libcufft.so.*|\
            libcurand.so.*|libcusparse.so.*|libcusolver.so.*|libnvJitLink.so.*|\
            libnvrtc.so.*) saw_cuda=1 ;;
            *) continue ;;
        esac
        if ! ort_host_has_soname "$soname" "$ldcache"; then
            return 1
        fi
    done <<< "$needed"
    # A real libonnxruntime_providers_cuda.so always DT_NEEDEDs CUDA runtime
    # libs; if we saw none, the file is not a usable GPU provider -> fail closed.
    [[ "$saw_cuda" -eq 1 ]] || return 1
    return 0
}

# Auto-discovery twin of pick_newest_ort_dir for the GPU path: identical version
# ranking, but skips candidates whose CUDA DT_NEEDED sonames do not resolve on
# this host, so ranking only considers loader-compatible builds. Prints the
# newest compatible onnxruntime/capi dir, or nothing when none are compatible.
pick_newest_compatible_gpu_ort_dir() {
    if ! command -v strings >/dev/null 2>&1; then
        echo >&2 "warn: 'strings' (binutils) not found — ORT version detection disabled; install with 'apt-get install binutils' or equivalent"
        return
    fi
    # Capture the ldconfig cache once; the pipeline subshell below inherits this
    # parent-scope var and reuses it for every candidate's every soname.
    local ldcache=""
    if command -v ldconfig >/dev/null 2>&1; then
        ldcache=$(ldconfig -p 2>/dev/null | awk '{ print $1 }' || true)
    fi
    # Same find|while|sort|head|awk shape as pick_newest_ort_dir (see the NOTE
    # there about bash-subshell vs zsh-parent-shell: the body only printfs, so
    # do NOT add cross-iteration state here either).
    find "$HOME/.cache/uv" -path "$1" -print 2>/dev/null |
        while IFS= read -r p; do
            d=$(dirname "$p")
            if ! ort_provider_cuda_compatible "$p" "$ldcache"; then
                echo >&2 "info: skipping CUDA-incompatible onnxruntime-gpu (unresolved CUDA soname) at $d"
                continue
            fi
            real=$(find "$d" -maxdepth 1 -name 'libonnxruntime.so.*.*.*' -type f 2>/dev/null | sort -V | tail -1)
            [[ -z "$real" ]] && continue
            v=$(strings "$real" 2>/dev/null |
                grep -E '^VERS_1\.[0-9]+\.[0-9]+' |
                sort -V | tail -1)
            if [[ -n "$v" ]]; then
                printf '%s %s\n' "$v" "$d"
            fi
        done |
        sort -V -r | head -1 | awk '{print $2}'
}

find_ort_dir() {
    # Check explicit override first.
    if [[ -n "${ORT_DIR:-}" ]]; then
        if [[ ! -d "$ORT_DIR" ]]; then
            echo >&2 "error: ORT_DIR is not a directory: $ORT_DIR"
            echo >&2 "Check ORT_DIR points to an onnxruntime/capi directory."
            echo >&2 "Unset ORT_DIR to fall back to auto-discovery."
            return 1
        fi
        if ! ort_dir_has_runtime_lib "$ORT_DIR"; then
            echo >&2 "error: ORT_DIR does not contain libonnxruntime.so: $ORT_DIR"
            echo >&2 "Check ORT_DIR points to an onnxruntime/capi directory."
            echo >&2 "Unset ORT_DIR to fall back to auto-discovery."
            return 1
        fi
        echo "$ORT_DIR"
        return
    fi

    # Search uv cache for onnxruntime-gpu first (has CUDA provider .so).
    #
    # Filter candidates by host CUDA-runtime compatibility BEFORE version
    # ranking. The newest cached onnxruntime-gpu may be built against a CUDA
    # major this host does not provide (e.g. 1.28 -> libcublasLt.so.13 on a
    # CUDA-12 host); that fails at model load, not at discovery. See the
    # compatibility helpers above.
    local gpu_any
    gpu_any=$(find "$HOME/.cache/uv" -path "*/onnxruntime/capi/libonnxruntime_providers_cuda.so" -print -quit 2>/dev/null)

    if [[ -n "$gpu_any" ]]; then
        if ort_can_check_cuda_compat; then
            local gpu_candidate
            gpu_candidate=$(pick_newest_compatible_gpu_ort_dir "*/onnxruntime/capi/libonnxruntime_providers_cuda.so")
            if [[ -n "$gpu_candidate" ]]; then
                echo "$gpu_candidate"
                return
            fi
            # Cached GPU ORT exists but none resolves this host's CUDA runtime.
            # Fail loudly with a remedy instead of silently downgrading to a CPU
            # build (GPU is the default) or selecting an incompatible build that
            # only fails later at model load.
            echo >&2 "error: found cached onnxruntime-gpu, but none is compatible with this host's CUDA runtime."
            echo >&2 "  The cached GPU ONNX Runtime build(s) need CUDA sonames this host does not expose"
            echo >&2 "  (e.g. libcublasLt.so.13 while this host provides libcublasLt.so.12)."
            echo >&2 "  Remedy (pick one):"
            echo >&2 "    - install an onnxruntime-gpu wheel built for this host's CUDA major:"
            echo >&2 "        uv pip install onnxruntime-gpu"
            echo >&2 "    - install the CUDA runtime the cached build needs (matching libcudart/libcublasLt majors),"
            echo >&2 "    - or point ORT_DIR at a compatible onnxruntime/capi directory:"
            echo >&2 "        ORT_DIR=/path/to/onnxruntime/capi"
            return 1
        else
            # No ldconfig + ELF-reader tooling to verify compatibility. Do not
            # regress a host where selection currently works: fall back to the
            # unfiltered newest-version GPU pick, but say why.
            echo >&2 "warn: cannot verify CUDA compatibility (need ldconfig plus readelf or objdump); selecting newest onnxruntime-gpu unfiltered"
            local gpu_candidate
            gpu_candidate=$(pick_newest_ort_dir "*/onnxruntime/capi/libonnxruntime_providers_cuda.so")
            if [[ -n "$gpu_candidate" ]]; then
                echo "$gpu_candidate"
                return
            fi
        fi
    fi

    # Fallback: any onnxruntime capi directory (CPU).
    local candidate
    candidate=$(pick_newest_ort_dir "*/onnxruntime/capi/libonnxruntime.so")

    if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return
    fi

    # Last resort: ask Python where onnxruntime lives.
    local pyort
    pyort=$(python3 -c "
import onnxruntime, pathlib
print(pathlib.Path(onnxruntime.__file__).parent / 'capi')
" 2>/dev/null || true)

    if [[ -n "$pyort" && -f "$pyort/libonnxruntime.so" ]]; then
        echo "$pyort"
        return
    fi

    # Caller signals total-failure via empty stdout. We do NOT call `exit 1`
    # here: this script is sourced (see header), and `exit` from a sourced
    # function terminates the caller's interactive shell — verified in bash
    # and zsh. agent-health rule 10 calls this out explicitly: a sourced
    # `exit` from the user's tmux pane can kill the entire tmux server when
    # that pane is the last one. Print the diagnostic, return empty, and let
    # the caller `return 1` from the sourced script.
    echo >&2 "error: cannot find ORT shared library."
    echo >&2 "Install onnxruntime-gpu: uv pip install onnxruntime-gpu"
    echo >&2 "Or set ORT_DIR=/path/to/onnxruntime/capi"
    return 1
}

ORT_CAPI=$(find_ort_dir)
# Empty ORT_CAPI means find_ort_dir hit the diagnostic path. Return from the
# sourced script — never `exit` from a sourced context.
[[ -z "$ORT_CAPI" ]] && return 1

# Ensure symlinks exist and point at the currently-selected versioned .so.
# `find` not shell glob to stay safe under zsh's NO_MATCH default when ORT_CAPI is empty.
# `sort -V | tail -1` for defensive coding (currently one .so per archive,
# but a future ORT debug variant would expose head -1's non-determinism).
#
# `[[ ! -e "$so1" ]] && ln -sf` would skip when the symlink already exists,
# even if it points to the WRONG (older) versioned .so — the same bug class
# d23861d closed on the picker side. Compare via `readlink` and refresh
# whenever the link target diverges. Self-healing on stale, idempotent on
# correct, no-op on dangling (then -z current → refresh).
if [[ -n "$ORT_CAPI" ]]; then
    real=$(find "$ORT_CAPI" -maxdepth 1 -name 'libonnxruntime.so.*.*.*' -type f 2>/dev/null | sort -V | tail -1)
    if [[ -n "$real" ]]; then
        expected=$(basename "$real")
        for alias_name in libonnxruntime.so.1 libonnxruntime.so; do
            current=$(readlink "$ORT_CAPI/$alias_name" 2>/dev/null || true)
            [[ "$current" != "$expected" ]] && ln -sf "$expected" "$ORT_CAPI/$alias_name"
        done
    fi
fi

export ORT_LIB_LOCATION="$ORT_CAPI"
export ORT_PREFER_DYNAMIC_LINK=1

# For GPU: ORT CUDA EP needs CUDA runtime (libcudart) and cuDNN (libcudnn) at runtime.
# Auto-discover common locations.
# Exported so aliases like `sparrow-engine-gpu` can read $EXTRA_LIB_PATHS directly.
export EXTRA_LIB_PATHS=""

# RP-24 dev support: ORT's TensorRT EP dlopens TensorRT 10 runtime libraries
# (libnvinfer, libnvinfer_plugin, libnvonnxparser). Production Docker images
# install those libs via apt; local GPU tests usually get them from the
# `tensorrt-cu12` pip package's sibling `tensorrt_libs` directory.
find_tensorrt_libs_dir() {
    if [[ -n "${TENSORRT_LIBS_DIR:-}" ]]; then
        if [[ -f "$TENSORRT_LIBS_DIR/libnvinfer.so.10" ]]; then
            echo "$TENSORRT_LIBS_DIR"
            return 0
        fi
        echo >&2 "warn: TENSORRT_LIBS_DIR does not contain libnvinfer.so.10: $TENSORRT_LIBS_DIR"
        return 1
    fi

    local candidate
    candidate=$(
        for search_root in \
            "${VIRTUAL_ENV:-}" \
            "$HOME/.local/lib" \
            "$HOME/.venvs" \
            "$HOME/.cache/uv"; do
            [[ -z "$search_root" || ! -d "$search_root" ]] && continue
            find "$search_root" -path '*/site-packages/tensorrt_libs/libnvinfer.so.10' -type f 2>/dev/null
        done |
            while IFS= read -r p; do
                lib_dir=$(dirname "$p")
                site_dir=$(dirname "$lib_dir")
                dist_info=$(find "$site_dir" -maxdepth 1 -name 'tensorrt_cu12_libs-*.dist-info' -type d 2>/dev/null | sort -V | tail -1)
                if [[ -n "$dist_info" ]]; then
                    version=$(basename "$dist_info" | sed -E 's/^tensorrt_cu12_libs-([0-9][^/]+)\.dist-info$/\1/')
                else
                    version=0
                fi
                printf '%s %s\n' "$version" "$lib_dir"
            done |
            sort -V -r | head -1 | awk '{print $2}'
    )
    if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi

    for search_root in \
        "${VIRTUAL_ENV:-}" \
        "$HOME/.local/lib" \
        "$HOME/.venvs" \
        "$HOME/.cache/uv"; do
        [[ -z "$search_root" || ! -d "$search_root" ]] && continue
        candidate=$(find "$search_root" -path '*/site-packages/tensorrt_libs/libnvinfer.so.10' -type f 2>/dev/null | sort -V | tail -1)
        if [[ -n "$candidate" ]]; then
            dirname "$candidate"
            return 0
        fi
    done

    candidate=$(python3 -c "
import pathlib
import site
import sys

roots = []
for getter in (getattr(site, 'getusersitepackages', None),):
    if getter is not None:
        try:
            roots.append(getter())
        except Exception:
            pass
try:
    roots.extend(site.getsitepackages())
except Exception:
    pass
roots.extend(sys.path)

seen = set()
for root in roots:
    if not root or root in seen:
        continue
    seen.add(root)
    p = pathlib.Path(root) / 'tensorrt_libs' / 'libnvinfer.so.10'
    if p.is_file():
        print(p.parent)
        raise SystemExit(0)
" 2>/dev/null || true)
    if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi

    return 1
}

tensorrt_libs_dir=$(find_tensorrt_libs_dir || true)
if [[ -n "$tensorrt_libs_dir" ]]; then
    EXTRA_LIB_PATHS="${tensorrt_libs_dir}:${EXTRA_LIB_PATHS}"
    echo "TensorRT: $tensorrt_libs_dir"
else
    echo >&2 "warn: TensorRT libs not found; install tensorrt-cu12 or set TENSORRT_LIBS_DIR for TRT EP dev tests"
fi

# cuDNN: we require 9.10+ for SpeciesNet on sm_89 (cuDNN 9.8 has a Conv engine
# bug with asymmetric padding — "No valid engine configs for ConvFwd_").
# PyTorch/TF bundle 9.8, so we prefer a standalone nvidia-cudnn-cu12>=9.10 if
# installed to ~/.local/cudnn or ~/.cache/uv.
# Install: uv pip install --target ~/.local/cudnn 'nvidia-cudnn-cu12>=9.10'
#
# Pick the newest cuDNN dir from the uv cache deterministically. This mirrors
# `pick_newest_ort_dir` (commit d23861d) — `find -print -quit` returned the
# first FS-traversal hit, so build vs. runtime could resolve to different
# versions. Also enforces the documented 9.10+ floor: sub-9.10 candidates are
# filtered out before the version-sort. (Drop `archive-v0` segment so this
# survives a future uv cache-format bump — see analogous fix in setup.sh.)
pick_newest_cudnn_dir() {
    # NOTE: same bash-subshell-vs-zsh-parent-shell pattern as pick_newest_ort_dir.
    # Body only `printf`s; if you add state, hoist it out of the pipeline
    # (use `< <(find …)` process-substitution instead). See the NOTE above
    # pick_newest_ort_dir's find|while pipeline.
    find "$HOME/.cache/uv" -path '*/nvidia/cudnn/lib/libcudnn.so.9.*.*.*' -type f 2>/dev/null |
        while IFS= read -r p; do
            d=$(dirname "$p")
            # Filename is libcudnn.so.MAJOR.MINOR.PATCH.PATCH2
            v=$(basename "$p" | sed 's/^libcudnn\.so\.//')
            # Enforce 9.10+ floor: skip 9.0..9.9 (PyTorch/TF bundles 9.8 with
            # the buggy asymmetric-padding Conv engine).
            major=$(echo "$v" | cut -d. -f1)
            minor=$(echo "$v" | cut -d. -f2)
            [[ "$major" -lt 9 ]] && continue
            [[ "$major" -eq 9 && "$minor" -lt 10 ]] && continue
            printf '%s %s\n' "$v" "$d"
        done |
        sort -V -r | head -1 | awk '{print $2}'
}

cudnn_cache_dir=$(pick_newest_cudnn_dir)

for cudnn_dir in \
    "$HOME/.local/cudnn/nvidia/cudnn/lib" \
    "$cudnn_cache_dir" \
    /usr/lib/python3/dist-packages/torch/lib \
    /usr/local/cuda/lib64 \
    /usr/lib/x86_64-linux-gnu; do
    if [[ -n "$cudnn_dir" && ( -f "$cudnn_dir/libcudnn.so" || -f "$cudnn_dir/libcudnn.so.9" ) ]]; then
        # Default loader directories must not be forced through
        # LD_LIBRARY_PATH. Homebrew-linked hosts such as dotnet use their own
        # glibc loader and crash if the system glibc directory is prepended.
        [[ "$cudnn_dir" != "/usr/lib/x86_64-linux-gnu" ]] &&
            EXTRA_LIB_PATHS="${cudnn_dir}:${EXTRA_LIB_PATHS}"
        echo "cuDNN: $cudnn_dir"
        break
    fi
done

# CUDA runtime: usually in system lib path or /usr/local/cuda. The
# /usr/local/cuda/targets/x86_64-linux/lib entry mirrors the same standard
# toolkit location the compatibility oracle (ort_host_has_soname) probes, so
# discovery-time compatibility and the runtime loader path stay in step.
for cuda_dir in \
    /usr/lib/x86_64-linux-gnu \
    /usr/local/cuda/lib64 \
    /usr/local/cuda/targets/x86_64-linux/lib; do
    if [[ -f "$cuda_dir/libcudart.so" ]] || [[ -f "$cuda_dir/libcudart.so.12" ]]; then
        # The system directory is already in the native loader's default
        # search path. Adding it explicitly can make foreign-loader hosts
        # resolve an incompatible glibc before their bundled runtime.
        [[ "$cuda_dir" != "/usr/lib/x86_64-linux-gnu" ]] &&
            EXTRA_LIB_PATHS="${cuda_dir}:${EXTRA_LIB_PATHS}"
        break
    fi
done

# Build LD_LIBRARY_PATH idempotently — re-sourcing must not multiply entries.
# Empirically, the prior `${ORT_CAPI}:${EXTRA_LIB_PATHS}${LD_LIBRARY_PATH:-}`
# pattern grew the path linearly: 138 → 276 → 414 chars over 3 sources, with
# every component triplicated. The dynamic linker dedupes at lookup so this
# was never functional, but `echo $LD_LIBRARY_PATH` for diagnostics became
# unreadable. Concat the new prefix (ORT_CAPI + EXTRA_LIB_PATHS + prior
# LD_LIBRARY_PATH), then de-duplicate via awk's first-seen idiom — preserves
# left-most occurrence (so newly-resolved ORT/cuDNN paths win priority over
# stale entries left over from a prior source). Portable across bash + zsh.
_combined_libpath="${ORT_CAPI}:${EXTRA_LIB_PATHS%:}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
LD_LIBRARY_PATH=$(printf '%s' "$_combined_libpath" | tr ':' '\n' | awk 'NF && !seen[$0]++' | paste -sd:)
unset _combined_libpath
export LD_LIBRARY_PATH

echo "ORT: $ORT_CAPI"
