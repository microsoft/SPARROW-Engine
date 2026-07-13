#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$SCRIPT_DIR/check_ffi_symbols.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

FAKE_BIN="$TEST_ROOT/bin"
TARGET_ROOT="$TEST_ROOT/target"
mkdir -p "$FAKE_BIN"

for flavor in cpu gpu mobile; do
    mkdir -p "$TARGET_ROOT/$flavor/release"
    : > "$TARGET_ROOT/$flavor/release/libsparrow_engine.so"
done

cat > "$FAKE_BIN/nm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cdylib="${@: -1}"
flavor="$(basename "$(dirname "$(dirname "$cdylib")")")"
def="$FAKE_ENGINE_DIR/sparrow-engine-$flavor/exports.def"
mapfile -t symbols < <(awk '/^[[:space:]]+sparrow_engine_/ { print $1 }' "$def")

case "${FAKE_NM_MODE:-valid}" in
    valid)
        ;;
    mobile_missing)
        if [[ "$flavor" == "mobile" ]]; then
            unset "symbols[$((${#symbols[@]} - 1))]"
        fi
        ;;
    gpu_wrong)
        if [[ "$flavor" == "gpu" ]]; then
            symbols[0]="sparrow_engine_wrong_symbol"
        fi
        ;;
    *)
        echo "unknown FAKE_NM_MODE: $FAKE_NM_MODE" >&2
        exit 2
        ;;
esac

for symbol in "${symbols[@]}"; do
    printf '0000000000000000 T %s\n' "$symbol"
done
EOF
chmod +x "$FAKE_BIN/nm"

run_checker() {
    local mode="$1"
    shift
    env \
        PATH="$FAKE_BIN:$PATH" \
        FAKE_ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)" \
        FAKE_NM_MODE="$mode" \
        SPARROW_ENGINE_FFI_TARGET_ROOT="$TARGET_ROOT" \
        bash "$CHECKER" "$@"
}

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_contains() {
    local needle="$1"
    local file="$2"
    grep -Fq "$needle" "$file" || fail "missing '$needle' in $file"
}

expect_failure() {
    local output="$1"
    local mode="$2"
    shift 2
    if run_checker "$mode" "$@" > "$output" 2>&1; then
        fail "expected checker failure: $mode $*"
    fi
}

echo "[1] CPU/GPU 37-symbol ABI and mobile 18-symbol ABI pass together"
run_checker valid --require-flavor all > "$TEST_ROOT/valid.out" 2>&1
assert_contains "cpu declared=37 built=37" "$TEST_ROOT/valid.out"
assert_contains "gpu declared=37 built=37" "$TEST_ROOT/valid.out"
assert_contains "mobile declared=18 built=18" "$TEST_ROOT/valid.out"
assert_contains "PASS: every checked flavor matches exports.def" "$TEST_ROOT/valid.out"

echo "[2] missing mobile export fails against the mobile definition"
expect_failure "$TEST_ROOT/mobile-missing.out" mobile_missing --require-flavor all
assert_contains "mobile cdylib differs from" "$TEST_ROOT/mobile-missing.out"

echo "[3] substituted GPU export fails even when the count stays 37"
expect_failure "$TEST_ROOT/gpu-wrong.out" gpu_wrong --require-flavor all
assert_contains "gpu cdylib differs from" "$TEST_ROOT/gpu-wrong.out"

echo "[4] missing required mobile artifact fails"
rm "$TARGET_ROOT/mobile/release/libsparrow_engine.so"
expect_failure "$TEST_ROOT/mobile-required.out" valid --require-flavor all
assert_contains "required mobile cdylib is missing" "$TEST_ROOT/mobile-required.out"

echo "check_ffi_symbols tests: PASS"
