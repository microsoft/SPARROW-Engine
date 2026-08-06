#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$SCRIPT_DIR/check_cuda_build_abi.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

cat > "$TEST_ROOT/Cargo.toml" <<'EOF'
[dependencies]
cudarc = { version = "=0.19.4", features = ["cuda-12080", "cublas"] }
EOF

cat > "$TEST_ROOT/nvcc" <<'EOF'
#!/usr/bin/env bash
echo "Cuda compilation tools, release ${FAKE_CUDA_VERSION}, V${FAKE_CUDA_VERSION}.0"
EOF
chmod +x "$TEST_ROOT/nvcc"

run_check() {
    FAKE_CUDA_VERSION="$1" \
    NVCC="$TEST_ROOT/nvcc" \
    SPARROW_ENGINE_GPU_CARGO_TOML="$TEST_ROOT/Cargo.toml" \
        bash "$CHECKER"
}

echo "[1] matching CUDA toolkit passes"
run_check 12.8 | grep -F "PASS (12.8 matches cuda-12080)"

echo "[2] mismatched CUDA toolkit fails closed"
if run_check 12.6 > "$TEST_ROOT/mismatch.log" 2>&1; then
    echo "FAIL: expected CUDA mismatch to fail" >&2
    exit 1
fi
grep -F "CUDA toolkit 12.6 does not match cudarc feature cuda-12080" \
    "$TEST_ROOT/mismatch.log"

echo "CUDA build ABI tests: PASS"
