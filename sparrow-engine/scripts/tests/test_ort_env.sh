#!/usr/bin/env bash
#
# Deterministic regression tests for ORT auto-discovery in scripts/ort-env.sh.
#
# Guards the CUDA-runtime compatibility filter: automatically discovered
# onnxruntime-gpu candidates must be filtered against host-visible CUDA
# dependency sonames BEFORE version ranking, so a newer build compiled against a
# CUDA major this host does not provide (e.g. 1.28 -> libcublasLt.so.13 on a
# CUDA-12 host) is never selected over an older but loader-compatible build.
#
# Fully hermetic: a temporary fixture HOME holds fake uv-cache ORT candidates,
# and stubbed `readelf` / `ldconfig` inject the ELF NEEDED sonames and the host
# CUDA soname set, so the result does not depend on the machine's real CUDA
# install. The "incompatible" candidate uses a sentinel CUDA major (99) whose
# sonames (libcudart.so.99 ...) cannot exist on ANY host — ldconfig cache,
# LD_LIBRARY_PATH, or the standard /usr/local/cuda + /usr/lib dirs the oracle
# probes — so the test stays host-independent even on a real CUDA-13 box. Real
# `strings` reads the fake version markers. Candidates carry an explicit
# DT_NEEDED list (see make_gpu_candidate); the cu12/cu99 sets include
# libcudnn.so.9 (absent from the stub host) to lock in that cuDNN — resolved
# separately by ort-env.sh — does not gate selection. Cases:
#   [1] newer incompatible cu99 + older compatible cu12 -> newest compatible cu12
#   [2] only incompatible cu99 -> fail clearly with remedy
#   [3] explicit ORT_DIR wins
#   [4] a required soname resolvable only via LD_LIBRARY_PATH is honoured
#   [5] a malformed / non-ELF provider is rejected (fail closed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_ENV="$SCRIPT_DIR/ort-env.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

STUB_DIR="$TEST_ROOT/stubbin"
mkdir -p "$STUB_DIR"

# Stub `ldconfig -p`: emulate a CUDA-12 host (only *.so.12 present, no *.so.13).
cat > "$STUB_DIR/ldconfig" <<'LDSTUB'
#!/usr/bin/env bash
# Test stub. Only `ldconfig -p` is consumed by ort-env.sh.
cat <<'LDP'
	libcudart.so.12 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudart.so.12
	libcudart.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudart.so
	libcublas.so.12 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcublas.so.12
	libcublasLt.so.12 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcublasLt.so.12
LDP
LDSTUB
chmod +x "$STUB_DIR/ldconfig"

# Stub `readelf -d <file>`: emit one NEEDED line per `NEEDED=<soname>` entry in
# the target provider file. A malformed / non-ELF provider has no NEEDED= lines
# and yields no output — exactly as real readelf produces no NEEDED lines for a
# corrupt file — exercising the fail-closed path.
cat > "$STUB_DIR/readelf" <<'RESTUB'
#!/usr/bin/env bash
# Test stub for `readelf -d <so>`.
so=""
for a in "$@"; do
    case "$a" in
        -*) ;;            # flags such as -d
        *) so="$a" ;;
    esac
done
while IFS= read -r line; do
    case "$line" in
        NEEDED=*) printf ' 0x0000000000000001 (NEEDED)             Shared library: [%s]\n' "${line#NEEDED=}" ;;
    esac
done < "$so" 2>/dev/null
RESTUB
chmod +x "$STUB_DIR/readelf"

# Emit the standard CUDA-<major> provider NEEDED set. Includes libcudnn.so.9
# (must be ignored by the gate), the libcuda.so.1 driver and libc.so.6 (ignored
# as non-CUDA/ABI-stable), so the ignore paths are exercised too.
cuda_needed() {
    local maj="$1"
    printf 'libcudart.so.%s libcublasLt.so.%s libcublas.so.%s libcudnn.so.9 libcuda.so.1 libc.so.6' \
        "$maj" "$maj" "$maj"
}

# Create a fake uv-cache onnxruntime-gpu candidate. Prints its capi dir.
#   $1 fixture HOME   $2 slug   $3 version (e.g. 1.26.0)   $4.. DT_NEEDED sonames
# With NO NEEDED sonames the provider .so is written empty, emulating a corrupt
# / non-ELF build (the stub readelf then reports no NEEDED entries).
make_gpu_candidate() {
    local home="$1" slug="$2" ver="$3"; shift 3
    local capi="$home/.cache/uv/archive-v0/$slug/onnxruntime/capi"
    mkdir -p "$capi"
    local pso="$capi/libonnxruntime_providers_cuda.so"
    : > "$pso"
    local n
    for n in "$@"; do
        printf 'NEEDED=%s\n' "$n" >> "$pso"
    done
    printf 'VERS_%s\n' "$ver" > "$capi/libonnxruntime.so.$ver"
    printf '%s' "$capi"
}

# Source ort-env.sh in a clean shell (stubs first on PATH, fixture HOME). The
# default LD_LIBRARY_PATH='' comes FIRST so a caller VAR=VAL override (e.g.
# LD_LIBRARY_PATH=...) still wins; HOME/PATH/ORT_ENV come LAST so they cannot be
# overridden. Emits `RC=<n>` and `ORT_CAPI=<dir>` then ort-env.sh's own output;
# caller captures stdout+stderr combined.
run_source() {
    local fixture_home="$1"; shift
    env \
        LD_LIBRARY_PATH='' \
        "$@" \
        HOME="$fixture_home" \
        PATH="$STUB_DIR:$PATH" \
        ORT_ENV="$ORT_ENV" \
        bash --norc --noprofile -c '
            source "$ORT_ENV"
            rc=$?
            printf "RC=%s\n" "$rc"
            printf "ORT_CAPI=%s\n" "${ORT_CAPI:-}"
        ' 2>&1
}

# ---------------------------------------------------------------------------
echo "[1] newer incompatible cu99 + older compatible cu12 -> selects newest compatible cu12"
HOME_A="$TEST_ROOT/home_a"
make_gpu_candidate "$HOME_A" gpu_128_cu99 1.28.0 $(cuda_needed 99) >/dev/null   # newest, incompatible (sentinel major)
WANT_A="$(make_gpu_candidate "$HOME_A" gpu_126_cu12 1.26.0 $(cuda_needed 12))"  # newest COMPATIBLE
make_gpu_candidate "$HOME_A" gpu_125_cu12 1.25.1 $(cuda_needed 12) >/dev/null   # older compatible
OUT_A="$(run_source "$HOME_A")"
printf '%s\n' "$OUT_A" | grep -qxF "RC=0" \
    || { echo "FAIL[1]: expected RC=0"; printf '%s\n' "$OUT_A" >&2; exit 1; }
printf '%s\n' "$OUT_A" | grep -qxF "ORT_CAPI=$WANT_A" \
    || { echo "FAIL[1]: expected newest compatible cu12 dir ($WANT_A)"; printf '%s\n' "$OUT_A" >&2; exit 1; }
if printf '%s\n' "$OUT_A" | grep -q "ORT_CAPI=.*gpu_128_cu99"; then
    echo "FAIL[1]: selected the incompatible sentinel-major build"; printf '%s\n' "$OUT_A" >&2; exit 1
fi
echo "    ok -> $WANT_A"

# ---------------------------------------------------------------------------
echo "[2] only incompatible cu99 GPU candidates -> fail clearly (no CPU downgrade)"
HOME_B="$TEST_ROOT/home_b"
make_gpu_candidate "$HOME_B" gpu_128_cu99 1.28.0 $(cuda_needed 99) >/dev/null
OUT_B="$(run_source "$HOME_B")"
if printf '%s\n' "$OUT_B" | grep -qxF "RC=0"; then
    echo "FAIL[2]: expected non-zero return when no compatible GPU build exists"; printf '%s\n' "$OUT_B" >&2; exit 1
fi
printf '%s\n' "$OUT_B" | grep -qxF "ORT_CAPI=" \
    || { echo "FAIL[2]: expected empty ORT_CAPI"; printf '%s\n' "$OUT_B" >&2; exit 1; }
printf '%s\n' "$OUT_B" | grep -qF "none is compatible with this host's CUDA runtime" \
    || { echo "FAIL[2]: missing clear incompatibility diagnostic"; printf '%s\n' "$OUT_B" >&2; exit 1; }
printf '%s\n' "$OUT_B" | grep -qF "ORT_DIR=" \
    || { echo "FAIL[2]: diagnostic must offer the ORT_DIR remedy"; printf '%s\n' "$OUT_B" >&2; exit 1; }
echo "    ok -> failed clearly with remedy"

# ---------------------------------------------------------------------------
echo "[3] explicit ORT_DIR still wins over auto-discovery (bypasses compat filter)"
HOME_C="$TEST_ROOT/home_c"
make_gpu_candidate "$HOME_C" gpu_128_cu99 1.28.0 $(cuda_needed 99) >/dev/null   # incompatible cache present
EXPLICIT="$TEST_ROOT/explicit_ort/capi"
mkdir -p "$EXPLICIT"
: > "$EXPLICIT/libonnxruntime.so"
OUT_C="$(run_source "$HOME_C" ORT_DIR="$EXPLICIT")"
printf '%s\n' "$OUT_C" | grep -qxF "RC=0" \
    || { echo "FAIL[3]: expected RC=0"; printf '%s\n' "$OUT_C" >&2; exit 1; }
printf '%s\n' "$OUT_C" | grep -qxF "ORT_CAPI=$EXPLICIT" \
    || { echo "FAIL[3]: explicit ORT_DIR must be honoured verbatim ($EXPLICIT)"; printf '%s\n' "$OUT_C" >&2; exit 1; }
echo "    ok -> $EXPLICIT"

# ---------------------------------------------------------------------------
echo "[4] required soname resolvable only via LD_LIBRARY_PATH is honoured (and required)"
HOME_D="$TEST_ROOT/home_d"
LDP_D="$TEST_ROOT/ldpath_d"
mkdir -p "$LDP_D"
# Fake soname: version 42 is absent from the stub ldconfig AND from every real
# system CUDA dir, so it can only resolve via the fixture LD_LIBRARY_PATH.
: > "$LDP_D/libcufft.so.42"
WANT_D="$(make_gpu_candidate "$HOME_D" gpu_127_ldpath 1.27.0 \
    libcudart.so.12 libcublasLt.so.12 libcufft.so.42 libcudnn.so.9 libcuda.so.1)"
# Positive: with LD_LIBRARY_PATH set, libcufft.so.42 resolves -> candidate kept.
OUT_D="$(run_source "$HOME_D" LD_LIBRARY_PATH="$LDP_D")"
printf '%s\n' "$OUT_D" | grep -qxF "RC=0" \
    || { echo "FAIL[4]: expected RC=0 with LD_LIBRARY_PATH set"; printf '%s\n' "$OUT_D" >&2; exit 1; }
printf '%s\n' "$OUT_D" | grep -qxF "ORT_CAPI=$WANT_D" \
    || { echo "FAIL[4]: candidate resolvable via LD_LIBRARY_PATH should be selected ($WANT_D)"; printf '%s\n' "$OUT_D" >&2; exit 1; }
# Negative control: same candidate, no LD_LIBRARY_PATH -> libcufft.so.42
# unresolved -> candidate rejected -> hard fail. Proves the LD_LIBRARY_PATH
# lookup, not some other path, is what made it compatible.
OUT_D2="$(run_source "$HOME_D")"
if printf '%s\n' "$OUT_D2" | grep -qxF "RC=0"; then
    echo "FAIL[4]: without LD_LIBRARY_PATH the unresolved soname must reject the candidate"; printf '%s\n' "$OUT_D2" >&2; exit 1
fi
printf '%s\n' "$OUT_D2" | grep -qF "none is compatible with this host's CUDA runtime" \
    || { echo "FAIL[4]: negative control should fail with the incompatibility diagnostic"; printf '%s\n' "$OUT_D2" >&2; exit 1; }
echo "    ok -> resolved via LD_LIBRARY_PATH; rejected without it"

# ---------------------------------------------------------------------------
echo "[5] malformed / non-ELF provider candidate is rejected (fail closed)"
HOME_E="$TEST_ROOT/home_e"
make_gpu_candidate "$HOME_E" gpu_130_malformed 1.30.0 >/dev/null                # newest, NO NEEDED (malformed)
WANT_E="$(make_gpu_candidate "$HOME_E" gpu_126_cu12 1.26.0 $(cuda_needed 12))"  # older, compatible
OUT_E="$(run_source "$HOME_E")"
printf '%s\n' "$OUT_E" | grep -qxF "RC=0" \
    || { echo "FAIL[5]: expected RC=0 (older compatible build should win)"; printf '%s\n' "$OUT_E" >&2; exit 1; }
printf '%s\n' "$OUT_E" | grep -qxF "ORT_CAPI=$WANT_E" \
    || { echo "FAIL[5]: malformed newest build must be rejected, older compatible selected ($WANT_E)"; printf '%s\n' "$OUT_E" >&2; exit 1; }
if printf '%s\n' "$OUT_E" | grep -q "ORT_CAPI=.*gpu_130_malformed"; then
    echo "FAIL[5]: selected the malformed provider"; printf '%s\n' "$OUT_E" >&2; exit 1
fi
echo "    ok -> $WANT_E"

echo "ort-env auto-discovery tests: PASS"
