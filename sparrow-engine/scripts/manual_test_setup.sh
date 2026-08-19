#!/usr/bin/env bash
# manual_test_setup.sh — source-only per-shell setup for the canonical manual test plan.
#
# WHAT: one reproducible command that a tester sources (bash OR zsh) to reach the
# state every plan section assumes: ORT dynamic-link env selected, fixtures and
# the 75-model cache validated, BOTH engine flavors built into isolated target
# dirs, a run-owned 10-image subset + output dir prepared, and every documented
# path exported. `spe` / `spe-gpu` resolve on PATH afterward.
#
# USAGE (must be SOURCED, never executed):
#   source SPARROW-Engine/sparrow-engine/scripts/manual_test_setup.sh   # bash or zsh
#
# RESOLUTION / OVERRIDES (all optional except where noted):
#   SPARROW_ENGINE_SOURCE          public repo root — resolved from THIS script's path.
#   SPARROW_ENGINE_DEV             dev companion — override, else the documented sibling
#                                  `<parent>/sparrow-engine-dev`; validated.
#   SPARROW_ENGINE_TEST_FILES_ROOT default /home/miao/repos/SparrowOPS/backups/test_files
#   SPARROW_ENGINE_MODEL_DIR       default ~/.sparrow-engine/models   (validated; never written)
#   SPARROW_ENGINE_OUT_DIR / OUT_DIR         default ~/.sparrow-engine/manual-test/out
#   SPARROW_ENGINE_TEST_DIR_SMALL            default ~/.sparrow-engine/manual-test/subset10
#
# It NEVER downloads or mutates the model cache, deletes caches, requires sudo, a
# special cwd, or a hidden required env var. The two documented overrides above
# (test-files root, model dir) have sensible defaults; a fresh tester needs none.
#
# Source-safety: this file changes NO persistent shell options in the caller
# (no `set -e`/`set -u`/pipefail leak) — all logic runs in `_spe_main` with
# explicit error handling; only the documented variables + PATH are exported.

# --------------------------------------------------------------------------
# Sourced-not-executed guard (bash + zsh). Executed -> clear error + non-zero.
# --------------------------------------------------------------------------
_spe_sourced=0
if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in *:file:*|*:file) _spe_sourced=1 ;; esac
elif [ -n "${BASH_VERSION:-}" ]; then
    [ "${BASH_SOURCE[0]}" != "${0}" ] && _spe_sourced=1
else
    echo "error: manual_test_setup.sh supports bash or zsh only." >&2
    exit 1
fi
if [ "$_spe_sourced" -ne 1 ]; then
    echo "error: manual_test_setup.sh must be SOURCED, not executed." >&2
    echo "  It exports variables into your shell; running it as a program does nothing useful." >&2
    echo "  Run:  source ${0##*/}      (from bash or zsh, any cwd)" >&2
    exit 1
fi
unset _spe_sourced

# --------------------------------------------------------------------------
# Resolve this script's own absolute path (bash + zsh), symlinks included.
# --------------------------------------------------------------------------
if [ -n "${ZSH_VERSION:-}" ]; then
    _spe_self_raw="${(%):-%x}"
else
    _spe_self_raw="${BASH_SOURCE[0]}"
fi

_spe_main() {
    local self_dir scripts_dir ws src dev tfr model_dir out_dir subset_dir
    local test_dir overhead_dir audio_dir test_img test_wav
    local expected_models found_models img wav
    local rc miss model_report d f common_git primary_dev

    self_dir="$(cd "$(dirname "$_spe_self_raw")" >/dev/null 2>&1 && pwd -P)" || {
        echo "error: cannot resolve manual_test_setup.sh directory." >&2; return 1; }
    scripts_dir="$self_dir"                                  # .../sparrow-engine/scripts
    ws="$(cd "$scripts_dir/.." >/dev/null 2>&1 && pwd -P)"   # .../sparrow-engine
    src="$(cd "$ws/.." >/dev/null 2>&1 && pwd -P)"           # public repo root
    if [ ! -f "$ws/Cargo.toml" ] || [ ! -f "$scripts_dir/ort-env.sh" ]; then
        echo "error: script path does not look like <repo>/sparrow-engine/scripts (ws=$ws)." >&2
        return 1
    fi
    SPARROW_ENGINE_SOURCE="$src"

    # --- dev companion: explicit override wins; else the documented sibling;
    # else, for a Git-worktree public checkout, the sibling beside the PRIMARY
    # checkout (a linked worktree's parent is not the sibling layout). ---
    if [ -n "${SPARROW_ENGINE_DEV:-}" ]; then
        dev="$SPARROW_ENGINE_DEV"
    else
        # (1) normal layout: sibling beside the public source.
        dev="$(cd "$src/.." >/dev/null 2>&1 && pwd -P)/sparrow-engine-dev"
        # (2) Git-worktree layout: when the public source is a LINKED worktree
        # (its .git is a file, not a dir) and the normal sibling is absent,
        # resolve the PRIMARY checkout from the common git dir and try the
        # sibling beside that primary checkout instead.
        if { [ ! -d "$dev" ] || [ ! -d "$dev/docs" ]; } && [ -f "$src/.git" ] && command -v git >/dev/null 2>&1; then
            common_git="$(git -C "$src" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
            if [ -n "$common_git" ] && [ -d "$common_git" ]; then
                # common_git = <primary>/.git  ->  <primary>/.git/../.. = <primary-parent>
                primary_dev="$(cd "$common_git/../.." >/dev/null 2>&1 && pwd -P)/sparrow-engine-dev"
                if [ -d "$primary_dev" ] && [ -d "$primary_dev/docs" ]; then
                    dev="$primary_dev"
                fi
            fi
        fi
    fi
    if [ ! -d "$dev" ] || [ ! -d "$dev/docs" ]; then
        echo "error: dev companion not found/invalid: $dev" >&2
        echo "  Tried the sibling beside the public checkout (and, for a Git worktree, beside the" >&2
        echo "  primary checkout). Set SPARROW_ENGINE_DEV=/path/to/sparrow-engine-dev to override." >&2
        return 1
    fi
    dev="$(cd "$dev" >/dev/null 2>&1 && pwd -P)"
    SPARROW_ENGINE_DEV="$dev"

    # --- ORT dynamic-link env (queue-001 compatibility auto-selection) ---
    # shellcheck source=/dev/null
    source "$scripts_dir/ort-env.sh" || {
        echo "error: failed sourcing ort-env.sh (no loader-compatible ONNX Runtime?)." >&2; return 1; }

    # --- fixtures ---
    tfr="${SPARROW_ENGINE_TEST_FILES_ROOT:-/home/miao/repos/SparrowOPS/backups/test_files}"
    if [ ! -d "$tfr" ]; then
        echo "error: test-files root not found: $tfr (set SPARROW_ENGINE_TEST_FILES_ROOT)." >&2; return 1; fi
    tfr="$(cd "$tfr" >/dev/null 2>&1 && pwd -P)"
    SPARROW_ENGINE_TEST_FILES_ROOT="$tfr"
    test_dir="$tfr/test_cameratrap"
    overhead_dir="$tfr/test_overhead"
    audio_dir="$tfr/test_audio"
    for d in "$test_dir" "$overhead_dir" "$audio_dir"; do
        if [ ! -d "$d" ]; then echo "error: missing fixture corpus: $d" >&2; return 1; fi
    done
    img="$(find "$test_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | sort | head -1)"
    if [ -z "$img" ]; then echo "error: no camera-trap image under $test_dir." >&2; return 1; fi
    wav="$(find "$audio_dir" -maxdepth 1 -type f -iname '*.wav' 2>/dev/null | sort | head -1)"
    if [ -z "$wav" ]; then echo "error: no .wav under $audio_dir." >&2; return 1; fi
    if ! find "$overhead_dir" -maxdepth 1 -type f 2>/dev/null | grep -q .; then
        echo "error: no overhead tiles under $overhead_dir." >&2; return 1; fi
    test_img="$img"
    test_wav="$wav"
    SPARROW_ENGINE_TEST_DIR="$test_dir"
    SPARROW_ENGINE_OVERHEAD_DIR="$overhead_dir"
    SPARROW_ENGINE_TEST_DIR_AUDIO="$audio_dir"
    SPARROW_ENGINE_TEST_IMG="$test_img"
    SPARROW_ENGINE_TEST_WAV="$test_wav"

    # --- models: validate EVERY catalog [[model]] resolves; NEVER download/mutate ---
    # Exact per-id iteration, NOT a raw manifest count: parse scripts/catalog.toml
    # with Python tomllib (Python >=3.11 is a documented prerequisite) and require,
    # for each catalog id, <model_dir>/<id>/pipeline.toml when format="cascade",
    # else <model_dir>/<id>/manifest.toml. Non-catalog directories are ignored, so
    # a missing catalog id fails even when extra manifest dirs exist. The model
    # cache is only read, never written.
    model_dir="${SPARROW_ENGINE_MODEL_DIR:-$HOME/.sparrow-engine/models}"
    if [ ! -d "$model_dir" ]; then
        echo "error: model dir not found: $model_dir" >&2
        echo "  Fetch the roster first with scripts/download_models.sh --all (NOT done by this setup)." >&2
        return 1
    fi
    model_dir="$(cd "$model_dir" >/dev/null 2>&1 && pwd -P)"
    SPARROW_ENGINE_MODEL_DIR="$model_dir"
    if ! command -v python3 >/dev/null 2>&1; then
        echo "error: python3 (>=3.11, for tomllib) is required to validate the model roster." >&2
        return 1
    fi
    # NOTE: the Python program body is at column 0 by necessity (Python is
    # indentation-sensitive; a quoted heredoc preserves leading whitespace).
    model_report="$(SPE_CATALOG="$scripts_dir/catalog.toml" SPE_MODEL_DIR="$model_dir" python3 - <<'PY'
import os, sys, tomllib
try:
    with open(os.environ["SPE_CATALOG"], "rb") as f:
        data = tomllib.load(f)
except Exception as exc:
    print("ERROR could not read catalog.toml: %s" % exc)
    sys.exit(2)
md = os.environ["SPE_MODEL_DIR"]
ids = [m for m in data.get("model", []) if m.get("id")]
missing = []
for m in ids:
    fname = "pipeline.toml" if m.get("format") == "cascade" else "manifest.toml"
    if not os.path.isfile(os.path.join(md, m["id"], fname)):
        missing.append("%s (%s): missing %s" % (m["id"], m.get("format"), fname))
print("COUNT %d" % len(ids))
for x in missing:
    print("MISSING %s" % x)
sys.exit(1 if missing else 0)
PY
)"
    rc=$?
    expected_models="$(printf '%s\n' "$model_report" | sed -n 's/^COUNT //p')"
    [ -z "$expected_models" ] && expected_models="?"
    if [ "$rc" -ne 0 ]; then
        echo "error: model cache does not satisfy the catalog roster ($expected_models catalog entries):" >&2
        printf '%s\n' "$model_report" | grep -v '^COUNT ' | sed 's/^/  - /' >&2
        echo "  Stage the full roster with scripts/download_models.sh --all (this setup NEVER downloads)." >&2
        return 1
    fi
    found_models="$expected_models"

    # --- build BOTH flavors into ISOLATED target dirs (queue-002 discipline) ---
    # Both cdylibs are libsparrow_engine.so (locked same-library-name invariant);
    # separate target-cpu/ and target-gpu/ keep them from colliding at the shared
    # output path, so no cargo clean and no cross-flavor race. Current feature
    # syntax: --no-default-features --features <flavor> for the dispatch crates,
    # --features ffi for the flavor cdylib crate.
    echo "[manual_test_setup] building CPU flavor (target-cpu/release) ..."
    CARGO_TARGET_DIR="$ws/target-cpu" cargo build --manifest-path "$ws/Cargo.toml" \
        --release --no-default-features --features cpu -p sparrow-engine-cli --bin spe \
        || { echo "error: CPU CLI (spe) build failed." >&2; return 1; }
    CARGO_TARGET_DIR="$ws/target-cpu" cargo build --manifest-path "$ws/Cargo.toml" \
        --release --no-default-features --features cpu -p sparrow-engine-server \
        || { echo "error: CPU server build failed." >&2; return 1; }
    CARGO_TARGET_DIR="$ws/target-cpu" cargo build --manifest-path "$ws/Cargo.toml" \
        --release --features ffi -p sparrow-engine-cpu \
        || { echo "error: CPU cdylib build failed." >&2; return 1; }

    echo "[manual_test_setup] building GPU flavor (target-gpu/release) ..."
    CARGO_TARGET_DIR="$ws/target-gpu" cargo build --manifest-path "$ws/Cargo.toml" \
        --release --no-default-features --features gpu -p sparrow-engine-cli --bin spe-gpu \
        || { echo "error: GPU CLI (spe-gpu) build failed." >&2; return 1; }
    CARGO_TARGET_DIR="$ws/target-gpu" cargo build --manifest-path "$ws/Cargo.toml" \
        --release --no-default-features --features gpu -p sparrow-engine-server \
        || { echo "error: GPU server build failed." >&2; return 1; }
    CARGO_TARGET_DIR="$ws/target-gpu" cargo build --manifest-path "$ws/Cargo.toml" \
        --release --features ffi -p sparrow-engine-gpu \
        || { echo "error: GPU cdylib build failed." >&2; return 1; }

    SPARROW_ENGINE_BIN_CPU="$ws/target-cpu/release/spe"
    SPARROW_ENGINE_BIN_GPU="$ws/target-gpu/release/spe-gpu"
    SPARROW_ENGINE_BIN="$SPARROW_ENGINE_BIN_CPU"
    SPARROW_ENGINE_SERVER_BIN_CPU="$ws/target-cpu/release/sparrow-engine-server"
    SPARROW_ENGINE_SERVER_BIN_GPU="$ws/target-gpu/release/sparrow-engine-server"
    SPARROW_ENGINE_LIB_CPU="$ws/target-cpu/release/libsparrow_engine.so"
    SPARROW_ENGINE_LIB_GPU="$ws/target-gpu/release/libsparrow_engine.so"

    # --- run-owned outputs: prepare ONLY the 10-image subset + output dir ---
    out_dir="${SPARROW_ENGINE_OUT_DIR:-${OUT_DIR:-$HOME/.sparrow-engine/manual-test/out}}"
    subset_dir="${SPARROW_ENGINE_TEST_DIR_SMALL:-$HOME/.sparrow-engine/manual-test/subset10}"
    mkdir -p "$out_dir" || { echo "error: cannot create OUT_DIR: $out_dir" >&2; return 1; }
    mkdir -p "$subset_dir" || { echo "error: cannot create subset dir: $subset_dir" >&2; return 1; }
    out_dir="$(cd "$out_dir" >/dev/null 2>&1 && pwd -P)"
    subset_dir="$(cd "$subset_dir" >/dev/null 2>&1 && pwd -P)"
    # Refresh the run-owned subset to exactly the 10 deterministic (sorted) images.
    # Delete both regular files AND symlinks at depth 1 (never directories): a
    # user-overridden SPARROW_ENGINE_TEST_DIR_SMALL may already hold symlinks, and
    # a plain `-type f` sweep would leave those stale links behind.
    find "$subset_dir" -mindepth 1 -maxdepth 1 \( -type f -o -type l \) -delete 2>/dev/null
    local n=0
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        cp -f "$f" "$subset_dir/" || { echo "error: cannot copy subset image." >&2; return 1; }
        n=$((n + 1)); [ "$n" -ge 10 ] && break
    done < <(find "$test_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | sort)
    if [ "$n" -lt 10 ]; then
        echo "error: only $n images available for the 10-image subset under $test_dir." >&2; return 1; fi
    SPARROW_ENGINE_OUT_DIR="$out_dir"
    OUT_DIR="$out_dir"
    SPARROW_ENGINE_TEST_DIR_SMALL="$subset_dir"

    # --- make spe / spe-gpu resolve via PATH (idempotent; subshell-safe) ---
    case ":$PATH:" in *":$ws/target-cpu/release:"*) ;; *) PATH="$ws/target-cpu/release:$PATH" ;; esac
    case ":$PATH:" in *":$ws/target-gpu/release:"*) ;; *) PATH="$ws/target-gpu/release:$PATH" ;; esac

    export SPARROW_ENGINE_SOURCE SPARROW_ENGINE_DEV SPARROW_ENGINE_TEST_FILES_ROOT \
        SPARROW_ENGINE_MODEL_DIR SPARROW_ENGINE_TEST_DIR SPARROW_ENGINE_TEST_DIR_SMALL \
        SPARROW_ENGINE_OVERHEAD_DIR SPARROW_ENGINE_TEST_DIR_AUDIO SPARROW_ENGINE_TEST_IMG \
        SPARROW_ENGINE_TEST_WAV SPARROW_ENGINE_OUT_DIR OUT_DIR \
        SPARROW_ENGINE_BIN SPARROW_ENGINE_BIN_CPU SPARROW_ENGINE_BIN_GPU \
        SPARROW_ENGINE_SERVER_BIN_CPU SPARROW_ENGINE_SERVER_BIN_GPU \
        SPARROW_ENGINE_LIB_CPU SPARROW_ENGINE_LIB_GPU PATH

    # --- validate every exported path exists; bounded summary ---
    rc=0; miss=0
    echo "==== sparrow-engine manual test setup ===="
    printf '  %-26s %s\n' "SPARROW_ENGINE_SOURCE" "$SPARROW_ENGINE_SOURCE"
    printf '  %-26s %s\n' "SPARROW_ENGINE_DEV" "$SPARROW_ENGINE_DEV"
    printf '  %-26s %s\n' "ORT_CAPI" "${ORT_CAPI:-<unset>}"
    printf '  %-26s %s catalog models present (per-id verified)\n' "models" "$expected_models"
    _spe_check() {  # $1 label  $2 path  $3 d|f
        local ok="ok"
        if [ "$3" = d ] && [ ! -d "$2" ]; then ok="MISSING"; miss=$((miss+1)); fi
        if [ "$3" = f ] && [ ! -e "$2" ]; then ok="MISSING"; miss=$((miss+1)); fi
        printf '  %-26s %-7s %s\n' "$1" "$ok" "$2"
    }
    _spe_check "MODEL_DIR"        "$SPARROW_ENGINE_MODEL_DIR"     d
    _spe_check "TEST_DIR"         "$SPARROW_ENGINE_TEST_DIR"      d
    _spe_check "TEST_DIR_SMALL"   "$SPARROW_ENGINE_TEST_DIR_SMALL" d
    _spe_check "OVERHEAD_DIR"     "$SPARROW_ENGINE_OVERHEAD_DIR"  d
    _spe_check "TEST_DIR_AUDIO"   "$SPARROW_ENGINE_TEST_DIR_AUDIO" d
    _spe_check "OUT_DIR"          "$SPARROW_ENGINE_OUT_DIR"       d
    _spe_check "TEST_IMG"         "$SPARROW_ENGINE_TEST_IMG"      f
    _spe_check "TEST_WAV"         "$SPARROW_ENGINE_TEST_WAV"      f
    _spe_check "BIN_CPU (spe)"    "$SPARROW_ENGINE_BIN_CPU"       f
    _spe_check "BIN_GPU (spe-gpu)" "$SPARROW_ENGINE_BIN_GPU"      f
    _spe_check "SERVER_BIN_CPU"   "$SPARROW_ENGINE_SERVER_BIN_CPU" f
    _spe_check "SERVER_BIN_GPU"   "$SPARROW_ENGINE_SERVER_BIN_GPU" f
    _spe_check "LIB_CPU"          "$SPARROW_ENGINE_LIB_CPU"       f
    _spe_check "LIB_GPU"          "$SPARROW_ENGINE_LIB_GPU"       f
    local subset_count
    subset_count="$(find "$SPARROW_ENGINE_TEST_DIR_SMALL" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')"
    printf '  %-26s %s files\n' "TEST_DIR_SMALL count" "$subset_count"
    if command -v spe >/dev/null 2>&1 && command -v spe-gpu >/dev/null 2>&1; then
        printf '  %-26s %s\n' "spe / spe-gpu on PATH" "yes"
    else
        printf '  %-26s %s\n' "spe / spe-gpu on PATH" "NO"; miss=$((miss+1))
    fi
    unset -f _spe_check 2>/dev/null || true
    if [ "$miss" -ne 0 ] || [ "$subset_count" -ne 10 ]; then
        echo "==== setup INCOMPLETE ($miss missing path(s); subset=$subset_count) ====" >&2
        rc=1
    else
        echo "==== setup OK — all paths valid, both flavors built, subset=10 ===="
    fi
    return "$rc"
}

_spe_main
_spe_rc=$?
unset -f _spe_main 2>/dev/null || true
unset _spe_self_raw 2>/dev/null || true
# Preserve the actual source status via explicit branches, but leave no internal
# `_spe_*` variable in the caller: unset `_spe_rc` before returning in each path.
if [ "$_spe_rc" -eq 0 ]; then
    unset _spe_rc 2>/dev/null || true
    return 0 2>/dev/null || true
else
    unset _spe_rc 2>/dev/null || true
    return 1 2>/dev/null || true
fi
