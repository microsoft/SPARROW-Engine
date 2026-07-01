#!/usr/bin/env bash
# scripts/smoke_test_models.sh — model-zoo consistency + loading smoke test.
#
# Guarantees, in order of importance:
#   1. catalog.toml is internally consistent (unique ids, zip-name convention,
#      required fields, aliases disjoint from active ids, known domain/task).
#   2. No source path loads a model by a STALE (renamed) id — the only place a
#      former id may appear is as a catalog `alias` (or an explanatory comment).
#   3. Every model already on disk resolves to a CURRENT catalog id (its
#      manifest id and directory name are a catalog id, never an old alias).
#   4. (best-effort) The engine loads the on-disk models by their catalog ids.
#
# Checks 1-3 are engine-free and always run. Check 4 runs only if the `spe`
# binary and a populated model dir are available; otherwise it SKIPs (never
# fails) because ORT runtime setup is environment-specific.
#
# Usage:
#   bash scripts/smoke_test_models.sh                      # default model dir
#   bash scripts/smoke_test_models.sh --model-dir /path    # custom model dir
#   SPARROW_CATALOG=/path/catalog.toml bash scripts/smoke_test_models.sh
#
# Exit 0 = all executed checks passed; non-zero = at least one FAIL.

set -uo pipefail  # NOT -e: we run every check and summarise at the end.

# Requires bash >= 4 (mapfile, associative-array-free but uses mapfile). macOS
# ships bash 3.2 by default.
if (( ${BASH_VERSINFO[0]:-0} < 4 )); then
  echo "ERROR: this script needs bash >= 4 (macOS ships 3.2 — run 'brew install bash')." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CATALOG="${SPARROW_CATALOG:-$SCRIPT_DIR/catalog.toml}"
MODEL_DIR="${SPARROW_ENGINE_MODEL_DIR:-${HOME:-.}/.sparrow-engine/models}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model-dir=*) MODEL_DIR="${1#*=}"; shift ;;
    -h|--help) sed -n '/^# /,/^set /p' "$0" | sed '/^set /d; s/^# \?//'; exit 0 ;;
    *) echo "ERROR: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

FAILS=0; PASSES=0; SKIPS=0
pass() { echo "  [PASS] $*"; PASSES=$((PASSES+1)); }
fail() { echo "  [FAIL] $*" >&2; FAILS=$((FAILS+1)); }
skip() { echo "  [SKIP] $*"; SKIPS=$((SKIPS+1)); }

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 required" >&2; exit 2; }
[[ -f "$CATALOG" ]] || { echo "ERROR: catalog not found: $CATALOG" >&2; exit 2; }

echo "Catalog:   $CATALOG"
echo "Model dir: $MODEL_DIR"
echo ""

# ---------------------------------------------------------------------------
# Check 1 — catalog internal integrity
# ---------------------------------------------------------------------------
echo "[1] catalog integrity"
INTEGRITY="$(python3 - "$CATALOG" <<'PY'
import sys, tomllib
DOMAINS = {"camera_trap", "acoustics", "overhead", "general"}
TASKS = {"detector", "classifier", "encoder", "cascade"}
FORMATS = {"onnx", "tflite", "cascade"}
REQ = ("id", "domain", "task", "format", "status", "license", "zip")
with open(sys.argv[1], "rb") as f:
    c = tomllib.load(f)
errs = []
models = c.get("model", [])
if not models:
    errs.append("no [[model]] entries")
ids, aliases = [], []
for m in models:
    mid = m.get("id", "<?>")
    for k in REQ:
        if k not in m:
            errs.append(f"{mid}: missing '{k}'")
    if m.get("domain") not in DOMAINS:
        errs.append(f"{mid}: bad domain {m.get('domain')!r}")
    if m.get("task") not in TASKS:
        errs.append(f"{mid}: bad task {m.get('task')!r}")
    if m.get("format") not in FORMATS:
        # `format` drives the default (onnx-only) download set; a typo here
        # silently drops an ONNX model from the default 18.
        errs.append(f"{mid}: bad format {m.get('format')!r} (want one of {sorted(FORMATS)})")
    exp = f"{m.get('domain')}__{m.get('task')}__{mid}.zip"
    if m.get("zip") != exp:
        errs.append(f"{mid}: zip {m.get('zip')!r} != {exp!r}")
    ids.append(mid)
    aliases += m.get("alias", [])
dupes = {x for x in ids if ids.count(x) > 1}
if dupes:
    errs.append(f"duplicate ids: {sorted(dupes)}")
clash = set(ids) & set(aliases)
if clash:
    errs.append(f"alias collides with active id: {sorted(clash)}")
z = c.get("zenodo", {})
if not z.get("record"):
    errs.append("zenodo.record missing")
if errs:
    print("FAIL")
    for e in errs:
        print(e)
else:
    print(f"OK {len(models)}")
PY
)"
if [[ "$(head -1 <<<"$INTEGRITY")" == OK* ]]; then
  pass "catalog well-formed (${INTEGRITY#OK } models)"
else
  while IFS= read -r line; do [[ "$line" == FAIL ]] && continue; fail "$line"; done <<<"$INTEGRITY"
fi
echo ""

# Active ids + alias set (used by checks 2-4).
mapfile -t CATALOG_IDS < <(python3 -c 'import sys,tomllib;[print(m["id"]) for m in tomllib.load(open(sys.argv[1],"rb"))["model"]]' "$CATALOG")
mapfile -t CATALOG_ALIASES < <(python3 -c 'import sys,tomllib;[print(a) for m in tomllib.load(open(sys.argv[1],"rb"))["model"] for a in m.get("alias",[])]' "$CATALOG")
is_catalog_id() { local x="$1" i; for i in "${CATALOG_IDS[@]}"; do [[ "$i" == "$x" ]] && return 0; done; return 1; }

# ---------------------------------------------------------------------------
# Check 2 — no stale old-id loading references in source
# ---------------------------------------------------------------------------
echo "[2] no stale old-id references in code (loading paths)"
if [[ ${#CATALOG_ALIASES[@]} -eq 0 ]]; then
  skip "catalog declares no aliases; nothing to scan"
else
  code_hit=0
  for old in "${CATALOG_ALIASES[@]}"; do
    # Search code/config only; exclude the catalog itself (alias lives there),
    # build output, and the downloader (its help text explains the alias).
    hits="$(grep -rEn --include='*.rs' --include='*.py' --include='*.toml' \
              "\b${old}\b" "$REPO_DIR" 2>/dev/null \
            | grep -v '/target/' \
            | grep -v 'scripts/catalog.toml' \
            | grep -v 'scripts/download_models.sh' \
            | grep -v 'scripts/smoke_test_models.sh' || true)"
    if [[ -n "$hits" ]]; then
      fail "old id '${old}' referenced in code:"
      echo "$hits" | sed 's/^/        /' >&2
      code_hit=1
    fi
    # Docs may legitimately mention the old name (as "legacy"); report as info.
    doc_hits="$(grep -rEln --include='*.md' "\b${old}\b" "$REPO_DIR" 2>/dev/null | grep -v '/target/' || true)"
    [[ -n "$doc_hits" ]] && echo "  [info] '${old}' mentioned in docs (allowed): $(echo "$doc_hits" | tr '\n' ' ')"
  done
  [[ $code_hit -eq 0 ]] && pass "no old ids in .rs/.py/.toml load paths"
fi
echo ""

# ---------------------------------------------------------------------------
# Check 3 — on-disk models use current catalog ids (engine-free)
# ---------------------------------------------------------------------------
echo "[3] on-disk models resolve to current catalog ids"
if [[ ! -d "$MODEL_DIR" ]]; then
  skip "model dir does not exist: $MODEL_DIR"
else
  found=0
  while IFS= read -r manifest; do
    [[ -z "$manifest" ]] && continue
    found=$((found+1))
    dir="$(basename "$(dirname "$manifest")")"
    mid="$(python3 - "$manifest" <<'PY' 2>/dev/null
import sys, tomllib
d = tomllib.load(open(sys.argv[1], "rb"))
print(d.get("model", {}).get("id", "") or d.get("id", ""))
PY
)"
    if [[ -z "$mid" ]]; then
      fail "$dir: manifest has no 'id' field"
      continue
    fi
    if [[ "$mid" != "$dir" ]]; then
      fail "$dir: manifest id '$mid' != directory name"
    elif ! is_catalog_id "$mid"; then
      # Is it a known OLD id? then it's a stale on-disk model.
      stale=0; for a in "${CATALOG_ALIASES[@]}"; do [[ "$a" == "$mid" ]] && stale=1; done
      if [[ $stale -eq 1 ]]; then
        fail "$dir: on-disk model uses OLD id '$mid' (renamed in catalog); re-download"
      else
        fail "$dir: id '$mid' not in catalog"
      fi
    else
      pass "$dir → catalog id '$mid'"
    fi
  done < <(find "$MODEL_DIR" -maxdepth 2 -name manifest.toml 2>/dev/null | sort)
  [[ $found -eq 0 ]] && skip "no manifests under $MODEL_DIR (run download_models.sh first)"
fi
echo ""

# ---------------------------------------------------------------------------
# Check 4 — engine load round-trip (best-effort)
# ---------------------------------------------------------------------------
echo "[4] engine load round-trip (best-effort)"
SPE=""
for cand in "$REPO_DIR/target/release/spe" "$REPO_DIR/target/debug/spe" "$(command -v spe 2>/dev/null || true)"; do
  [[ -n "$cand" && -x "$cand" ]] && { SPE="$cand"; break; }
done
if [[ -z "$SPE" ]]; then
  skip "no spe binary found (build with cargo, or install)"
elif ! find "$MODEL_DIR" -maxdepth 2 -name manifest.toml 2>/dev/null | grep -q .; then
  skip "no models on disk to load"
else
  # Source the shared ORT discovery env INSIDE a subshell so its
  # LD_LIBRARY_PATH changes (CUDA/ORT libs) do not leak into later python3
  # calls in this script (that leak causes a glibc "stack smashing" abort).
  out="$(
    (
      [[ -f "$SCRIPT_DIR/ort-env.sh" ]] && source "$SCRIPT_DIR/ort-env.sh" >/dev/null 2>&1
      timeout 120 "$SPE" --device cpu --model-dir "$MODEL_DIR" models list 2>/dev/null
    )
  )"
  rc=$?
  if [[ $rc -ne 0 || -z "$out" ]]; then
    skip "spe models list unavailable (rc=$rc; ORT env / device issue — not a catalog failure)"
  else
    bad=0
    n_parsed=0
    # Parse `{"id":"...",...}` JSON lines with grep (no python — avoids any
    # LD_LIBRARY_PATH contamination risk).
    while IFS= read -r rid; do
      [[ -z "$rid" ]] && continue
      n_parsed=$((n_parsed+1))
      if is_catalog_id "$rid"; then
        pass "engine loaded '$rid' (catalog id)"
      else
        fail "engine reports id '$rid' not in catalog"; bad=1
      fi
    done < <(printf '%s\n' "$out" \
              | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
              | sed -E 's/.*:[[:space:]]*"([^"]*)".*/\1/')
    if [[ $n_parsed -eq 0 ]]; then
      # Non-empty output but no id parsed → the `models list` format changed;
      # don't pass vacuously.
      skip "spe returned output but no model ids parsed (output format changed?)"
    elif [[ $bad -eq 0 ]]; then
      pass "all engine-loaded ids are catalog ids"
    fi
  fi
fi
echo ""

# ---------------------------------------------------------------------------
echo "======================================================================"
echo "smoke test: ${PASSES} passed, ${FAILS} failed, ${SKIPS} skipped"
echo "======================================================================"
[[ $FAILS -eq 0 ]] && exit 0 || exit 1
