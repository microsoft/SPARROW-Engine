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

echo "[9] both flavors stage packaging only under target/wheels and regenerate RECORD"
if grep -Eq 'FLAVOR_SENTINEL|write_flavor_sentinel|remove_flavor_sentinel|python/sparrow_engine/_flavor\.py|\.console-scripts\.|\.wheel-metadata\.|pyproject\.toml\.bak|sed -i|trash-put|ls -t' "$BUILD_SH"; then
    fail "build.sh contains source-local packaging writes, undeclared cleanup, or stale-wheel selection"
fi
grep -Fq 'wheels_dir="$(cd .. && pwd -P)/target/wheels"' "$BUILD_SH" \
    || fail "packaging output root is not the declared target/wheels"
grep -Fq 'mktemp -d "$wheels_dir/.wheel-packaging.XXXXXX"' "$BUILD_SH" \
    || fail "packaging scratch is not output-owned and run-unique"
grep -Fq -- '--out "$scratch/raw"' "$BUILD_SH" || fail "maturin does not isolate newly built wheels"
grep -Fq 'repack_wheel "$wheel" "$scratch" "$flavor"' "$BUILD_SH" || fail "flavors do not share wheel repacking"
grep -Fq 'build_wheel cpu' "$BUILD_SH" || fail "CPU does not use output-owned packaging"
grep -Fq 'build_wheel gpu' "$BUILD_SH" || fail "GPU does not use output-owned packaging"
grep -Fq 'wheel unpack "$wheel" -d "$scratch/unpacked"' "$BUILD_SH" || fail "wheel unpack escapes scratch"
grep -Fq 'wheel pack "$unpacked" -d "$scratch/packed"' "$BUILD_SH" || fail "RECORD-regenerating wheel pack is missing"
grep -Fq -- '--wheel-dir "$scratch/repaired/"' "$BUILD_SH" || fail "auditwheel repair escapes scratch"
console_code="$(sed -n '/^report_console_scripts() {/,/^}/p' "$BUILD_SH")"
printf '%s\n' "$console_code" | grep -Fq 'with ZipFile(' || fail "console metadata is not read directly from the wheel"
if printf '%s\n' "$console_code" | grep -Eq 'mktemp|wheel unpack|mkdir'; then
    fail "console inspection must not create packaging scratch"
fi
grep -Fq 'cleanup FAILED' "$BUILD_SH" || fail "packaging cleanup failures are not surfaced"
grep -Fq 'trap - EXIT' "$BUILD_SH" || fail "packaging cleanup does not disable its own trap"

echo "[10] real wheel unpack/repack preserves flavor metadata, RECORD, and cleanup failures"
# Only compilation/repair are fixture commands here; packaging runs the actual
# build.sh functions with its existing wheel dependency. Manual tests build both
# native wheels and exercise the real Linux auditwheel gate separately.
uv run --no-project --with wheel python - "$BUILD_SH" "$SPARROW_ENGINE_DIR/target/wheels" <<'PY'
import base64
import csv
from email.parser import Parser
import hashlib
import io
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from zipfile import ZipFile

source = Path(sys.argv[1]).read_text(encoding="utf-8")
helpers = source[source.index("cleanup_packaging_dir() {"):source.index('\ncase "$SPARROW_ENGINE_FLAVOR" in')]
output_root = Path(sys.argv[2]).resolve()
output_root.mkdir(parents=True, exist_ok=True)
stem = "sparrow_engine-0.0.0"
metadata = (
    "Metadata-Version: 2.1\nName: sparrow-engine\nVersion: 0.0.0\n"
    "Summary: API (sparrow-engine CPU pipeline)\n"
    "Requires-Dist: onnxruntime>=1.25.1,<1.26\nRequires-Dist: numpy>=1.26\n\n"
    "Unchanged body: Name: sparrow-engine; onnxruntime.\n"
)

def digest(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()

def make_seed(path):
    payload = {
        "sparrow_engine/__init__.py": b"# wheel fixture\n",
        f"{stem}.data/data/example.txt": b"wheel data\n",
        f"{stem}.dist-info/METADATA": metadata.encode(),
        f"{stem}.dist-info/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{stem}.dist-info/entry_points.txt": b"[console_scripts]\nexample = sparrow_engine:main\n",
    }
    record = f"{stem}.dist-info/RECORD"
    rows = io.StringIO()
    csv.writer(rows, lineterminator="\n").writerows(
        [(name, digest(data), str(len(data))) for name, data in payload.items()] + [(record, "", "")]
    )
    payload[record] = rows.getvalue().encode()
    with ZipFile(path, "w") as wheel:
        for name, data in payload.items():
            wheel.writestr(name, data)

with tempfile.TemporaryDirectory(prefix=".wheel-contract.", dir=output_root) as temporary:
    root = Path(temporary)
    seed = root / f"{stem}-py3-none-any.whl"
    make_seed(seed)
    fixture_maturin = """
fixture_maturin() {
    local output=""
    while [[ "$#" -gt 0 ]]; do
        case "$1" in --out) output="$2"; shift 2 ;; *) shift ;; esac
    done
    [[ -n "$output" ]] || return 65
    cp -- "$seed" "$output/"
}
"""

    def run_case(label, command, expected=0, cleanup_failed=False, stale=False):
        case = root / label
        checkout = case / "workspace" / "sparrow-engine-python"
        package = checkout / "python" / "sparrow_engine"
        package.mkdir(parents=True)
        (checkout / "pyproject.toml").write_text("unchanged CPU template\n")
        (package / "__init__.py").write_text("unchanged source\n")
        before = {p.relative_to(checkout): p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
        wheels = case / "workspace" / "target" / "wheels"
        wheels.mkdir(parents=True)
        if stale:
            (wheels / seed.name).write_bytes(seed.read_bytes())
        script = case / "run.sh"
        script.write_text(
            "set -euo pipefail\n" + helpers + "\nIS_WINDOWS=1\nMATURIN=fixture_maturin\n"
            + "seed=" + shlex.quote(str(seed)) + "\n" + fixture_maturin + command + "\n"
        )
        result = subprocess.run(["bash", str(script)], cwd=checkout, capture_output=True, text=True)
        assert result.returncode == expected, (label, result.returncode, result.stdout, result.stderr)
        after = {p.relative_to(checkout): p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
        assert before == after, (label, "source checkout changed")
        scratch = list(wheels.glob(".wheel-packaging.*"))
        assert bool(scratch) == cleanup_failed, (label, "unexpected scratch cleanup state", scratch)
        assert ("cleanup FAILED" in result.stderr) == cleanup_failed, (label, result.stderr)
        print(f"PASS: {label} (exit {result.returncode})")
        return wheels, result

    for flavor in ("cpu", "gpu"):
        wheels, result = run_case(flavor, f"build_wheel {flavor}")
        paths = list(wheels.glob("*.whl"))
        assert len(paths) == 1, paths
        expected_stem = stem if flavor == "cpu" else stem.replace("sparrow_engine-", "sparrow_engine_gpu-")
        assert paths[0].name == f"{expected_stem}-py3-none-any.whl"
        with ZipFile(paths[0]) as wheel:
            names = wheel.namelist()
            assert len(names) == len(set(names)), "duplicate wheel members"
            meta = wheel.read(f"{expected_stem}.dist-info/METADATA").decode()
            headers = Parser().parsestr(meta)
            assert headers["Name"] == ("sparrow-engine" if flavor == "cpu" else "sparrow-engine-gpu")
            assert headers["Version"] == "0.0.0"
            assert headers.get_all("Conflicts-Dist", []) == []
            assert headers.get_all("Provides-Dist", []) == ([] if flavor == "cpu" else ["sparrow-engine"])
            dependencies = headers.get_all("Requires-Dist", [])
            runtime = "onnxruntime" if flavor == "cpu" else "onnxruntime-gpu"
            assert f"{runtime}>=1.25.1,<1.26" in dependencies
            assert "numpy>=1.26" in dependencies
            if flavor == "cpu":
                assert meta == metadata
            else:
                assert "(sparrow-engine GPU pipeline)" in headers["Summary"]
                for dependency in ("nvidia-cudnn-cu12>=9,<10", "nvidia-cublas-cu12", "nvidia-curand-cu12", "nvidia-cufft-cu12"):
                    assert f'{dependency}; sys_platform == "linux"' in dependencies
                assert len(dependencies) == 6
                assert not any(name.startswith(stem + ".dist-info/") or name.startswith(stem + ".data/") for name in names)
            assert meta.endswith("Unchanged body: Name: sparrow-engine; onnxruntime.\n")
            assert wheel.read(f"{expected_stem}.data/data/example.txt") == b"wheel data\n"
            assert wheel.read("sparrow_engine/_flavor.py").decode().endswith(f'FLAVOR = "{flavor}"\n')
            record = f"{expected_stem}.dist-info/RECORD"
            rows = list(csv.reader(io.StringIO(wheel.read(record).decode())))
            assert len(rows) == len(names) and {row[0] for row in rows} == set(names)
            for name, checksum, size in rows:
                if name == record:
                    assert checksum == size == ""
                else:
                    data = wheel.read(name)
                    assert checksum == digest(data) and size == str(len(data)), name
        assert "example = sparrow_engine:main" in result.stdout

    wheels, result = run_case("missing-output", "fixture_maturin() { :; }\nbuild_wheel cpu", 1, stale=True)
    assert "expected exactly one new" in result.stderr
    assert (wheels / seed.name).read_bytes() == seed.read_bytes(), "stale wheel was changed"
    wheels, _ = run_case("build-failure", "fixture_maturin() { return 42; }\nbuild_wheel cpu", 42)
    assert not list(wheels.glob("*.whl"))
    wheels, _ = run_case("repair-failure", "IS_WINDOWS=0\nauditwheel() { return 53; }\nbuild_wheel gpu", 53)
    assert not list(wheels.glob("*.whl")), "unrepaired GPU wheel escaped"
    wheels, result = run_case("missing-repair-output", "IS_WINDOWS=0\nauditwheel() { :; }\nbuild_wheel gpu", 1)
    assert "expected exactly one new" in result.stderr and not list(wheels.glob("*.whl"))
    run_case("cleanup-failure", "rm() { return 47; }\nbuild_wheel cpu", 3, cleanup_failed=True)
    run_case("build-and-cleanup-failure", "fixture_maturin() { return 42; }\nrm() { return 47; }\nbuild_wheel cpu", 42, cleanup_failed=True)
    _, result = run_case("cleanup-refuses-source", 'cleanup_packaging_dir "$PWD" "$(cd ../target/wheels && pwd -P)"', 2)
    assert "cleanup refuses scratch outside" in result.stderr
print("PASS: CPU/GPU metadata, RECORD regeneration, fresh-output selection, hard repair, and exact cleanup")
PY

echo "PASS: manylinux GPU wheel build contract (release-locked env, dispatch, no auditwheel escape, docs not overclaiming)"
