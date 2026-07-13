#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARROW_ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: scripts/check_ffi_symbols.sh [--build] [--require-flavor <flavor>|--require-flavor all]

Checks each built sparrow-engine flavor cdylib against its own exports.def.
CPU and GPU must declare the same ABI. Mobile intentionally exposes a smaller,
generic ABI and is validated against its separate definition.

Options:
  --build                  Build CPU and GPU cdylibs first in isolated target dirs:
                           target/cpu/ and target/gpu/. Mobile is not built here
                           because it needs the LiteRT/TFLite native toolchain.
  --require-flavor FLAVOR  Require a non-CPU flavor cdylib to be present and match
                           its declared export set. Repeatable. Valid: gpu, mobile, all.
                           The SPARROW_ENGINE_REQUIRED_FFI_FLAVORS env var accepts
                           the same values as a comma-separated list.
  -h, --help               Show this help.
EOF
}

TARGET_ROOT="${SPARROW_ENGINE_FFI_TARGET_ROOT:-$SPARROW_ENGINE_DIR/target}"

build_requested=0
required_flavors=()

add_required_flavor() {
    local flavor="$1"
    case "$flavor" in
        ""|cpu)
            ;;
        gpu|mobile)
            required_flavors+=("$flavor")
            ;;
        all)
            required_flavors+=("gpu" "mobile")
            ;;
        *)
            echo "error: invalid required flavor: $flavor" >&2
            exit 2
            ;;
    esac
}

if [[ -n "${SPARROW_ENGINE_REQUIRED_FFI_FLAVORS:-}" ]]; then
    IFS=',' read -r -a env_required_flavors <<< "$SPARROW_ENGINE_REQUIRED_FFI_FLAVORS"
    for flavor in "${env_required_flavors[@]}"; do
        add_required_flavor "$flavor"
    done
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            build_requested=1
            shift
            ;;
        --require-flavor)
            if [[ $# -lt 2 ]]; then
                echo "error: --require-flavor needs a value" >&2
                usage >&2
                exit 2
            fi
            add_required_flavor "$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

target_dir_for() {
    local flavor="$1"
    printf '%s/%s' "$TARGET_ROOT" "$flavor"
}

cdylib_for() {
    local flavor="$1"
    printf '%s/release/libsparrow_engine.so' "$(target_dir_for "$flavor")"
}

build_flavor() {
    local flavor="$1"
    local package="$2"

    echo "[check_ffi_symbols] building $flavor cdylib in $(target_dir_for "$flavor")"
    (
        cd "$SPARROW_ENGINE_DIR"
        CARGO_TARGET_DIR="$(target_dir_for "$flavor")" \
            cargo build -p "$package" --release --features ffi
    )
}

exports_def_for() {
    local flavor="$1"
    printf '%s/sparrow-engine-%s/exports.def' "$SPARROW_ENGINE_DIR" "$flavor"
}

declared_symbol_set() {
    local flavor="$1"
    local def
    def="$(exports_def_for "$flavor")"
    [[ -f "$def" ]] || {
        echo "error: exports definition not found: $def" >&2
        return 1
    }
    awk '/^[[:space:]]+sparrow_engine_/ { print $1 }' "$def" | sort -u
}

built_symbol_set() {
    local cdylib="$1"
    local nm_output
    if ! command -v nm >/dev/null 2>&1; then
        echo "error: nm not found; install binutils to inspect cdylib exports" >&2
        return 1
    fi
    if ! nm_output=$(nm -D --defined-only "$cdylib"); then
        echo "error: nm failed while reading exports from $cdylib" >&2
        return 1
    fi
    printf '%s\n' "$nm_output" \
        | awk '$2 == "T" && $3 ~ /^sparrow_engine_/ { print $3 }' \
        | sort -u
}

set_count() {
    local symbols="$1"
    if [[ -z "$symbols" ]]; then
        printf '0\n'
    else
        grep -c . <<< "$symbols"
    fi
}

verify_flavor_exports() {
    local flavor="$1"
    local cdylib="$2"
    local declared
    local built
    local declared_n
    local built_n

    declared="$(declared_symbol_set "$flavor")" || return 1
    built="$(built_symbol_set "$cdylib")" || return 1
    declared_n="$(set_count "$declared")"
    built_n="$(set_count "$built")"
    echo "[check_ffi_symbols] $flavor declared=$declared_n built=$built_n ($cdylib)"

    if [[ "$built" != "$declared" ]]; then
        echo "[check_ffi_symbols] mismatch: $flavor cdylib differs from $(exports_def_for "$flavor")" >&2
        diff -u \
            <(printf '%s\n' "$declared") \
            <(printf '%s\n' "$built") >&2 || true
        return 1
    fi
}

flavor_is_required() {
    local flavor="$1"
    local required
    for required in "${required_flavors[@]}"; do
        if [[ "$required" == "$flavor" ]]; then
            return 0
        fi
    done
    return 1
}

print_build_command() {
    local flavor="$1"
    local package="$2"
    cat <<EOF
[check_ffi_symbols] $flavor cdylib not found: $(cdylib_for "$flavor")
[check_ffi_symbols] build it with:
  cd "$SPARROW_ENGINE_DIR" && CARGO_TARGET_DIR="$(target_dir_for "$flavor")" cargo build -p $package --release --features ffi
EOF
}

if [[ "$build_requested" -eq 1 ]]; then
    build_flavor cpu sparrow-engine-cpu
    build_flavor gpu sparrow-engine-gpu
fi

cpu_cdylib="$(cdylib_for cpu)"
if [[ ! -f "$cpu_cdylib" ]]; then
    print_build_command cpu sparrow-engine-cpu >&2
    echo "[check_ffi_symbols] cannot establish CPU baseline N; aborting." >&2
    exit 2
fi

cpu_declared="$(declared_symbol_set cpu)"
gpu_declared="$(declared_symbol_set gpu)"
if [[ "$cpu_declared" != "$gpu_declared" ]]; then
    echo "[check_ffi_symbols] CPU/GPU exports.def files differ." >&2
    diff -u \
        <(printf '%s\n' "$cpu_declared") \
        <(printf '%s\n' "$gpu_declared") >&2 || true
    exit 1
fi

baseline_n="$(set_count "$cpu_declared")"
if [[ "$baseline_n" -eq 0 ]]; then
    echo "[check_ffi_symbols] CPU exports.def is empty; refusing to pass." >&2
    exit 1
fi
if ! verify_flavor_exports cpu "$cpu_cdylib"; then
    exit 1
fi
echo "[check_ffi_symbols] CPU/GPU declared ABI count=$baseline_n"

mismatch=0
missing_required=0
found_flavors=("cpu")
skipped_flavors=()

check_flavor() {
    local flavor="$1"
    local package="$2"
    local cdylib

    cdylib="$(cdylib_for "$flavor")"
    if [[ ! -f "$cdylib" ]]; then
        print_build_command "$flavor" "$package"
        if flavor_is_required "$flavor"; then
            echo "[check_ffi_symbols] required $flavor cdylib is missing." >&2
            missing_required=1
        else
            echo "[check_ffi_symbols] skipping optional $flavor because it is not built yet."
            skipped_flavors+=("$flavor")
        fi
        return 0
    fi

    found_flavors+=("$flavor")
    if ! verify_flavor_exports "$flavor" "$cdylib"; then
        mismatch=1
    fi
}

check_flavor gpu sparrow-engine-gpu
check_flavor mobile sparrow-engine-mobile

printf '[check_ffi_symbols] found flavor cdylibs:'
printf ' %s' "${found_flavors[@]}"
printf '\n'
if [[ "${#skipped_flavors[@]}" -gt 0 ]]; then
    printf '[check_ffi_symbols] skipped optional flavor cdylibs:'
    printf ' %s' "${skipped_flavors[@]}"
    printf '\n'
fi

if [[ "$missing_required" -ne 0 ]]; then
    echo "[check_ffi_symbols] FAIL: one or more required flavor cdylibs are missing." >&2
    exit 1
fi

if [[ "$mismatch" -ne 0 ]]; then
    echo "[check_ffi_symbols] FAIL: one or more built FFI symbol sets differ from exports.def." >&2
    exit 1
fi

echo "[check_ffi_symbols] PASS: every checked flavor matches exports.def; CPU/GPU declarations are identical. Use --require-flavor all for full CPU/GPU/mobile verification."
