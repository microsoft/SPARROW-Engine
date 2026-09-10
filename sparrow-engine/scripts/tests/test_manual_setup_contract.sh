#!/usr/bin/env bash
#
# Lightweight contract test for scripts/manual_test_setup.sh.
#
# Guards a SIGPIPE-under-pipefail regression: sourcing the setup from a strict
# shell (`set -euo pipefail` / zsh `setopt ERR_EXIT NO_UNSET PIPE_FAIL`) must not
# die with SIGPIPE (141) because an early-consumer pipeline (`find | sort | head -1`,
# `find | grep -q`) makes its upstream `find`/`sort` SIGPIPE under pipefail. Also
# statically guards the setup's sourced-only / no-shell-option-leak /
# no-internal-`_spe_*`-leak invariants.
#
# Intentionally does NOT source the setup (that runs the expensive Cargo build) —
# the real strict-shell source is the documented per-shell E2E. This test is
# static + exercises the replacement idioms and the actual embedded roster
# validator against tiny run-owned fixtures.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP="$SCRIPT_DIR/manual_test_setup.sh"
[[ -f "$SETUP" ]] || { echo "test_manual_setup_contract: missing $SETUP" >&2; exit 2; }

PASS=0; FAILED=0
ok()  { printf '  ok: %s\n' "$1"; PASS=$((PASS + 1)); }
bad() { printf '  FAIL: %s\n' "$1" >&2; FAILED=$((FAILED + 1)); }

# Setup source with shell comments stripped, so a comment that merely NAMES a
# forbidden pattern never trips the static guards. `has` greps a here-string (the
# shell backs it with a regular file), never a `producer | grep -q` pipeline —
# which is the very early-consumer hazard this test guards against.
SETUP_CODE="$(sed 's/#.*$//' "$SETUP")"
has() { grep -qE "$1" <<< "$SETUP_CODE"; }

# Run a snippet in a fresh strict shell; echo its exit code (never aborts us).
strict_rc() { local r; bash -c "set -euo pipefail; $1" >/dev/null 2>&1 && r=0 || r=$?; echo "$r"; }

echo "[1] setup parses under bash and zsh"
if bash -n "$SETUP"; then ok "bash -n"; else bad "bash -n"; fi
if command -v zsh >/dev/null 2>&1; then
  if zsh -n "$SETUP"; then ok "zsh -n"; else bad "zsh -n"; fi
else
  echo "  (zsh absent — skipped)"
fi

echo "[2] no SIGPIPE-prone early-consumer pipeline remains"
if has 'find\b.*\|[[:space:]]*head\b'; then bad "'find ... | head' early-consumer pipeline present"; else ok "no 'find | head' pipeline"; fi
if has 'find\b.*\|[[:space:]]*grep[[:space:]]+-q'; then bad "'find ... | grep -q' early-consumer pipeline present"; else ok "no 'find | grep -q' pipeline"; fi
if has '\|[[:space:]]*head\b'; then bad "'| head' pipeline present anywhere (use 'sed -n 1p', which reads to EOF)"; else ok "no '| head' pipeline anywhere"; fi
# The safe replacements must actually be in place.
if has "sort[[:space:]]*\|[[:space:]]*sed -n '1p'"; then ok "first-file selection uses 'sort | sed -n 1p'"; else bad "expected 'sort | sed -n 1p' first-file idiom"; fi
if has 'find\b.*-print[[:space:]]+-quit'; then ok "non-empty check uses 'find -print -quit'"; else bad "expected 'find -print -quit' non-empty idiom"; fi

echo "[3] replacement idioms are correct + SIGPIPE-safe under set -euo pipefail"
FIX="$(mktemp -d)"; trap 'rm -rf "$FIX"' EXIT
# Deterministic lexical order; the lexically-first name carries spaces.
: > "$FIX/A first with space.jpg"
: > "$FIX/M_middle.jpeg"
: > "$FIX/Z_last.png"
EMPTY="$FIX/empty"; mkdir -p "$EMPTY"

# first-sorted-file idiom (inherits our pipefail): lexically-first, spaces intact, rc 0.
set +e
sel="$(find "$FIX" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | sort | sed -n '1p')"
selrc=$?
set -e
if [ "$selrc" -eq 0 ] && [ "$sel" = "$FIX/A first with space.jpg" ]; then
  ok "first-sorted returns the lexically-first file (spaces preserved), rc 0"
else
  bad "first-sorted wrong (rc=$selrc sel=[$sel])"
fi

# empty dir -> empty selection, rc 0 (no crash under pipefail).
set +e
esel="$(find "$EMPTY" -maxdepth 1 -type f 2>/dev/null | sort | sed -n '1p')"
esrc=$?
set -e
if [ "$esrc" -eq 0 ] && [ -z "$esel" ]; then ok "first-sorted on empty dir -> empty, rc 0"; else bad "empty-dir first-sorted (rc=$esrc sel=[$esel])"; fi

# non-empty `find -print -quit` idiom: non-empty for a populated dir, empty otherwise.
set +e
probe="$(find "$FIX" -maxdepth 1 -type f -print -quit 2>/dev/null)"; prc=$?
eprobe="$(find "$EMPTY" -maxdepth 1 -type f -print -quit 2>/dev/null)"; eprc=$?
set -e
if [ "$prc" -eq 0 ] && [ -n "$probe" ]; then ok "find -print -quit: non-empty dir -> non-empty, rc 0"; else bad "print-quit non-empty (rc=$prc probe=[$probe])"; fi
if [ "$eprc" -eq 0 ] && [ -z "$eprobe" ]; then ok "find -print -quit: empty dir -> empty, rc 0"; else bad "print-quit empty (rc=$eprc probe=[$eprobe])"; fi

# SIGPIPE-safety property on a large producer (> pipe buffer): the chosen idiom
# survives pipefail where the retired `head -1` early-consumer does not.
if [ "$(strict_rc 'seq 200000 | sort | sed -n "1p"')" -eq 0 ]; then
  ok "'sort | sed -n 1p' survives pipefail on a large producer (rc 0)"
else
  bad "'sort | sed -n 1p' unexpectedly failed under pipefail on a large producer"
fi
if [ "$(strict_rc 'seq 200000 | sort | head -1')" -ne 0 ]; then
  ok "contrast: retired 'sort | head -1' DOES SIGPIPE under pipefail (proves the fix is required)"
else
  bad "expected retired 'head -1' to fail under pipefail on a large producer"
fi

echo "[4] static sourced-only / no-option-leak / no-internal-leak invariants"
if has 'must be SOURCED, not executed'; then ok "sourced-only guard present"; else bad "sourced-only guard missing"; fi
if has 'ZSH_EVAL_CONTEXT' && has 'BASH_SOURCE\[0\]'; then ok "bash + zsh sourced detection present"; else bad "sourced detection incomplete"; fi
# The setup must set NO persistent shell option in the caller.
if has '\bset[[:space:]]+-[A-Za-z]*[eux]' || has '\bset[[:space:]]+-o[[:space:]]+pipefail' || has '\bset[[:space:]]+\+o\b' || has '\bsetopt\b'; then
  bad "setup sets/clears a shell option (would leak into the caller)"
else
  ok "setup sets no shell option (no errexit/nounset/pipefail leak)"
fi
# Every top-level internal name is torn down before returning.
for u in 'unset _spe_sourced' 'unset _spe_self_raw' 'unset _spe_rc' 'unset -f _spe_main' 'unset -f _spe_check'; do
  if has "$u"; then ok "tail teardown: $u"; else bad "missing teardown: $u"; fi
done
# Fixture-selection vars are function-local (so img/wav/etc. never leak either).
if has 'local .*\bimg\b' && has 'local .*\bwav\b' && has 'local .*\boverhead_probe\b'; then
  ok "fixture-selection vars (img/wav/overhead_probe) declared local"
else
  bad "fixture-selection vars not all declared local"
fi

echo "[5] fallible command substitutions are guarded so caller errexit reaches the handler"
if has 'model_report="\$\(' && has '\)"[[:space:]]*\|\|[[:space:]]*rc=\$\?'; then
  ok "roster-validator capture guarded ( )\" || rc=\$? ) — a missing-model exit reaches the error block"
else
  bad "roster-validator capture is unguarded (caller set -e would abort before the missing-roster message)"
fi
if has 'common_git="\$\(git.*\|\|[[:space:]]*common_git='; then
  ok "optional git common-dir lookup guarded (|| common_git=…) — a non-git checkout does not abort the setup"
else
  bad "optional git common-dir lookup is unguarded (set -e would abort on a non-git checkout)"
fi

echo "[6] the guarded-capture idiom reaches its handler where the unguarded form aborts (set -euo pipefail)"
g="$(bash -c 'set -euo pipefail; rc=0; out="$(sh -c "echo COUNT 42; exit 1")" || rc=$?; printf "reached rc=%s out=[%s]" "$rc" "$out"' 2>/dev/null || true)"
if [ "$g" = "reached rc=1 out=[COUNT 42]" ]; then
  ok "guarded capture: an expected-nonzero producer reaches the handler (rc + payload captured)"
else
  bad "guarded capture idiom broken: [$g]"
fi
u="$(bash -c 'set -euo pipefail; out="$(sh -c "echo COUNT; exit 1")"; printf reached' 2>/dev/null || true)"
if [ -z "$u" ]; then
  ok "contrast: the UNGUARDED capture aborts before its handler under set -e (proves the guard is required)"
else
  bad "expected the unguarded capture to abort under set -e, got [$u]"
fi

echo "[7] schema-1.2 roster validation uses hosting state and all four formats"
if python3 - "$SETUP" "$FIX" <<'PY'
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]).read_text()
marker = "python3 - <<'PY'\n"
assert source.count(marker) == 1, "expected one embedded roster validator"
validator = source.split(marker, 1)[1].split("\nPY\n", 1)[0]
compile(validator, str(sys.argv[1]) + ":roster", "exec")
descriptors = {
    "onnx": "manifest.toml",
    "tflite": "manifest.toml",
    "cascade": "pipeline.toml",
    "ensemble": "ensemble.toml",
}
compliance = {"MODEL_CARD.md", "LICENSE.md", "ATTRIBUTION.md", "CONVERSION.md", "SOURCE.md"}
rows = [
    ("hosted-onnx", "onnx", "hosted", "active"),
    ("restricted-onnx", "onnx", "hosted_restricted", "active"),
    ("mobile", "tflite", "hosted", "active"),
    ("cascade", "cascade", "hosted", "active"),
    ("ensemble", "ensemble", "hosted", "active"),
    ("candidate", "onnx", "hosted", "candidate"),
    ("metadata", "onnx", "link_only", "active"),
]
checks = 0

def check_case(label, mutate=None, error=None):
    global checks
    with tempfile.TemporaryDirectory(prefix="roster-", dir=sys.argv[2]) as temp:
        root = Path(temp)
        model_dir = root / "models"
        model_dir.mkdir()
        models = [
            dict(zip(("id", "format", "hosting_status", "status"), row))
            for row in rows
        ]
        for model in models:
            package = model_dir / model["id"]
            package.mkdir()
            names = compliance if model["hosting_status"] == "link_only" else [descriptors[model["format"]]]
            for name in names:
                (package / name).write_text("fixture\n")
        if mutate:
            mutate(models, model_dir)
        catalog = root / "catalog.toml"
        catalog.write_text('schema_version = "1.2"\n' + "".join(
            "\n[[model]]\n" + "".join(f"{key} = {json.dumps(value)}\n" for key, value in model.items())
            for model in models
        ))
        result = subprocess.run(
            [sys.executable, "-c", validator],
            env={**os.environ, "SPE_CATALOG": str(catalog), "SPE_MODEL_DIR": str(model_dir)},
            capture_output=True, text=True, check=False,
        )
        details = f"{label}: rc={result.returncode}\n{result.stdout}{result.stderr}"
        if error is None:
            assert result.returncode == 0, details
            assert result.stdout.splitlines() == ["COUNT 7", "RUNTIME 6", "METADATA 1"], details
        else:
            assert result.returncode == 1, details
            assert error in result.stdout, details
        assert not result.stderr, details
        checks += 1
        print(f"  ok: {label}")

check_case("hosted, hosted_restricted, candidate, four runtime formats, exact-five metadata and counts")

for mid, fmt, hosting, status in rows[:-1]:
    check_case(
        f"missing {mid} descriptor rejected (including candidate)",
        lambda models, root, mid=mid, fmt=fmt: (root / mid / descriptors[fmt]).unlink(),
        f"{mid} ({fmt}): missing {descriptors[fmt]}",
    )

def wrong_ensemble(models, root):
    (root / "ensemble" / "ensemble.toml").rename(root / "ensemble" / "manifest.toml")

check_case("ensemble cannot use manifest.toml", wrong_ensemble, "ensemble (ensemble): missing ensemble.toml")

def missing_package(models, root):
    (root / "candidate").rename(root / "non-catalog-extra")

check_case("non-catalog directory cannot replace missing candidate", missing_package, "candidate (onnx): missing manifest.toml")

def runtime_symlink(models, root, fmt, hosting, descriptor_only):
    model = models[0]
    package = root / model["id"]
    original = package / descriptors[model["format"]]
    model.update(format=fmt, hosting_status=hosting)
    descriptor = package / descriptors[fmt]
    if original != descriptor:
        original.rename(descriptor)
    path = descriptor if descriptor_only else package
    target = root.parent / "outside-model-root"
    path.rename(target)
    path.symlink_to(target, target_is_directory=not descriptor_only)

for hosting in ("hosted", "hosted_restricted"):
    for fmt in descriptors:
        for descriptor_only in (False, True):
            kind = "descriptor" if descriptor_only else "package"
            error = (
                f"hosted-onnx: {descriptors[fmt]} must be a regular non-symlink file"
                if descriptor_only else
                "hosted-onnx: runtime package must be a non-symlink directory"
            )
            check_case(
                f"{hosting} {fmt} rejects {kind} symlink escape",
                lambda models, root, fmt=fmt, hosting=hosting, descriptor_only=descriptor_only:
                    runtime_symlink(models, root, fmt, hosting, descriptor_only),
                error,
            )

def selected_root_symlink(models, root):
    target = root.parent / "selected-model-root"
    root.rename(target)
    root.symlink_to(target, target_is_directory=True)

check_case("selected model root may itself be a symlink", selected_root_symlink)

check_case(
    "missing metadata package rejected",
    lambda models, root: shutil.rmtree(root / "metadata"),
    "metadata: link_only package must be a non-symlink directory",
)

for name in ("manifest.toml", "pipeline.toml", "ensemble.toml", "model.onnx", "model.tflite", "extra.txt", ".hidden"):
    check_case(
        f"metadata rejects extra {name}",
        lambda models, root, name=name: (root / "metadata" / name).write_text("extra\n"),
        f"metadata: unexpected link_only entry {name}",
    )

def nested_extra(models, root):
    nested = root / "metadata" / "nested"
    nested.mkdir()
    (nested / "model.onnx").write_text("payload\n")

check_case("metadata rejects nested payload", nested_extra, "metadata: unexpected link_only entry nested")
check_case(
    "metadata rejects empty hidden directory",
    lambda models, root: (root / "metadata" / ".hidden").mkdir(),
    "metadata: unexpected link_only entry .hidden",
)

for name in sorted(compliance):
    check_case(
        f"metadata rejects missing {name}",
        lambda models, root, name=name: (root / "metadata" / name).unlink(),
        f"metadata: {name} must be a regular non-symlink file",
    )
    check_case(
        f"metadata rejects empty {name}",
        lambda models, root, name=name: (root / "metadata" / name).write_text(""),
        f"metadata: empty compliance file {name}",
    )

def compliance_symlink(models, root, dangling=False):
    path = root / "metadata" / "LICENSE.md"
    path.unlink()
    path.symlink_to("missing.md" if dangling else "SOURCE.md")

check_case("metadata rejects compliance symlink", compliance_symlink, "LICENSE.md must be a regular non-symlink file")
check_case(
    "metadata rejects dangling compliance symlink",
    lambda models, root: compliance_symlink(models, root, dangling=True),
    "LICENSE.md must be a regular non-symlink file",
)

def package_symlink(models, root):
    package = root / "metadata"
    package.rename(root / "metadata-target")
    package.symlink_to("metadata-target", target_is_directory=True)

check_case("metadata rejects package symlink", package_symlink, "metadata: link_only package must be a non-symlink directory")
check_case(
    "metadata rejects extra symlink",
    lambda models, root: (root / "metadata" / "extra-link").symlink_to("SOURCE.md"),
    "metadata: unexpected link_only entry extra-link",
)

def compliance_directory(models, root):
    path = root / "metadata" / "LICENSE.md"
    path.unlink()
    path.mkdir()
    (path / "LICENSE.md").write_text("nested\n")

check_case("metadata rejects directory in place of compliance file", compliance_directory, "LICENSE.md must be a regular non-symlink file")

def compliance_fifo(models, root):
    path = root / "metadata" / "LICENSE.md"
    path.unlink()
    os.mkfifo(path)

check_case("metadata rejects non-regular compliance file", compliance_fifo, "LICENSE.md must be a regular non-symlink file")

for hosting in ("unknown", "pending_rights", ""):
    check_case(
        f"unsupported hosting_status {hosting!r} rejected",
        lambda models, root, hosting=hosting: models[0].update(hosting_status=hosting),
        f"hosted-onnx: unsupported hosting_status {hosting!r}",
    )
check_case(
    "missing hosting_status rejected",
    lambda models, root: models[0].pop("hosting_status"),
    "hosted-onnx: unsupported hosting_status None",
)
for index in (0, 6):
    for fmt in ("unknown", ""):
        check_case(
            f"{rows[index][0]} rejects unsupported format {fmt!r}",
            lambda models, root, index=index, fmt=fmt: models[index].update(format=fmt),
            f"{rows[index][0]}: unsupported format {fmt!r}",
        )
    check_case(
        f"{rows[index][0]} rejects missing format",
        lambda models, root, index=index: models[index].pop("format"),
        f"{rows[index][0]}: unsupported format None",
    )
check_case(
    "missing catalog id is not silently skipped",
    lambda models, root: models[0].pop("id"),
    "invalid catalog id None",
)
check_case("empty catalog rejected", lambda models, root: models.clear(), "catalog has no [[model]] entries")
check_case(
    "duplicate catalog id rejected",
    lambda models, root: models.append(models[0].copy()),
    "hosted-onnx: duplicate catalog id",
)
print(f"  schema-1.2 roster: {checks} hermetic cases passed")
PY
then
  ok "schema-1.2 roster acceptance and rejection cases"
else
  bad "schema-1.2 roster validation"
fi

echo "---"
if [ "$FAILED" -gt 0 ]; then
  echo "FAIL: $FAILED checks failed ($PASS passed)"
  exit 1
fi
echo "PASS: manual_test_setup contract ($PASS checks)"
