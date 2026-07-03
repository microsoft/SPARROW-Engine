#!/usr/bin/env bash
# scripts/download_models.sh — download the sparrow-engine model zoo from Zenodo.
#
# The model list, Zenodo record, and per-model ZIP names are ALL read from
# `catalog.toml` (the single source of truth, next to this script). To add,
# remove, or re-point a model, edit catalog.toml — never this script.
#
# Each model is stored on Zenodo as a per-model ZIP named
# `<domain>__<task>__<id>.zip` (Zenodo has no real folders; the "__" delimiter
# groups models when the file list is sorted). Every ZIP unpacks to `<id>/`, so
# the engine resolves models by flat id: `<dest>/<id>/manifest.toml` +
# `model.onnx` + `labels.txt`. Integrity is verified against the per-file MD5
# checksums published by the Zenodo record API.
#
# Usage:
#   bash scripts/download_models.sh                     # 36 desktop ONNX models -> ~/.sparrow-engine/models/
#   bash scripts/download_models.sh --all               # all 42 (incl. mobile .tflite + cascade)
#   bash scripts/download_models.sh --dest /path        # custom destination dir
#   bash scripts/download_models.sh MDV6-yolov10-e ...  # specific model(s) only
#   bash scripts/download_models.sh --list              # show available models (from catalog)
#   bash scripts/download_models.sh --force             # re-download even if present
#   bash scripts/download_models.sh --no-verify         # skip MD5 check (faster, unsafe)
#
# Old model ids still work via catalog aliases (e.g. `Species_Net_MDV5a`
# resolves to `MDV5a`).
#
# After the script completes, point sparrow-engine at the directory:
#   export SPARROW_ENGINE_MODEL_DIR=$(realpath ~/.sparrow-engine/models)
#   spe list-models
#   spe detect --model MDV6-yolov10-e --image /path/to/image.jpg
#
# (No explicit env var is needed if the default ~/.sparrow-engine/models is
# used — the CLI / server / Python wheels all default to that path.)
#
# Override the Zenodo record (e.g. to test a newer draft):
#   ZENODO_RECORD=<id> bash scripts/download_models.sh

set -euo pipefail

# Requires bash >= 4 (associative arrays). macOS ships bash 3.2 by default.
if (( ${BASH_VERSINFO[0]:-0} < 4 )); then
  echo "ERROR: this script needs bash >= 4 (macOS ships 3.2 — run 'brew install bash')." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG="${SPARROW_CATALOG:-$SCRIPT_DIR/catalog.toml}"

# ---- Tool check (python3 needed to parse the TOML catalog) ----
for tool in curl unzip md5sum python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: required tool '$tool' not found in PATH." >&2
    exit 1
  fi
done

if [[ ! -f "$CATALOG" ]]; then
  echo "ERROR: catalog not found at $CATALOG (set SPARROW_CATALOG to override)." >&2
  exit 1
fi

# ---- Read record + version from the catalog (env override wins) ----
read -r CATALOG_RECORD ZENODO_VERSION ZENODO_CONCEPT_DOI < <(
  python3 - "$CATALOG" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    c = tomllib.load(f)
z = c.get("zenodo", {})
print(z.get("record", ""), z.get("version", "?"), z.get("concept_doi", ""))
PY
)
ZENODO_RECORD="${ZENODO_RECORD:-$CATALOG_RECORD}"
ZENODO_DOI="10.5281/zenodo.${ZENODO_RECORD}"
ZENODO_BASE="https://zenodo.org/records/${ZENODO_RECORD}/files"
ZENODO_API="https://zenodo.org/api/records/${ZENODO_RECORD}"

# ---- Defaults ----
# Default destination matches the CLI / server / Python default
# (`dirs_default_model_dir` in sparrow-engine-cli/src/main.rs). A no-arg
# download followed by a no-arg `spe detect` works without env-var ceremony.
DEST="${HOME:-.}/.sparrow-engine/models"
VERIFY=1
FORCE=0
ALL=0
SELECTED=()

# ---- Argument parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)       DEST="$2"; shift 2 ;;
    --dest=*)     DEST="${1#*=}"; shift ;;
    --no-verify)  VERIFY=0; shift ;;
    --force)      FORCE=1; shift ;;
    --all)        ALL=1; shift ;;
    --list)
      echo "Sparrow Model Zoo v${ZENODO_VERSION} — Zenodo record ${ZENODO_RECORD} (DOI ${ZENODO_DOI})"
      echo ""
      python3 - "$CATALOG" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    c = tomllib.load(f)
models = c["model"]
w = max(len(m["id"]) for m in models)
last = None
for m in sorted(models, key=lambda m: (m["domain"], m["task"], m["id"])):
    grp = f"{m['domain']}/{m['task']}"
    if grp != last:
        print(f"\n[{grp}]")
        last = grp
    fam = ",".join(m.get("family", []))
    fam = f"  ({fam})" if fam else ""
    # Mark non-default (mobile) artifacts so users know they need --all / an
    # explicit name to fetch them.
    tag = "" if m.get("format") == "onnx" else f"  [{m.get('format')}, mobile — needs --all or explicit name]"
    print(f"  {m['id']:<{w}}  {m['license']}{fam}{tag}")
n_onnx = sum(1 for m in models if m.get("format") == "onnx")
print(f"\n{len(models)} models total; {n_onnx} desktop ONNX models fetched by default.")
print("Mobile .tflite / cascade artifacts are fetched only when named explicitly or with --all.")
PY
      exit 0
      ;;
    -h|--help)
      sed -n '/^# /,/^set/p' "$0" | sed '/^set/d; s/^# \?//'
      exit 0
      ;;
    -*)
      echo "ERROR: unknown flag '$1'. Use --help for usage." >&2
      exit 1
      ;;
    *)
      SELECTED+=("$1")
      shift
      ;;
  esac
done

# ---- Resolve the selection against the catalog (id or alias) ----
# Emits `id<TAB>zip` per resolved model. Unknown ids abort with a clear error.
# With no selection: default to desktop ONNX models only (format == "onnx");
# `--all` (SPARROW_ALL=1) expands to every catalog entry. Explicitly named
# models are always fetched regardless of format.
RESOLVED="$(
  SPARROW_ALL="$ALL" python3 - "$CATALOG" "${SELECTED[@]+"${SELECTED[@]}"}" <<'PY'
import os, sys, tomllib
with open(sys.argv[1], "rb") as f:
    c = tomllib.load(f)
models = c["model"]
by_id = {m["id"]: m for m in models}
alias = {}
for m in models:
    for a in m.get("alias", []):
        alias[a] = m["id"]

selected = sys.argv[2:]
if not selected:
    if os.environ.get("SPARROW_ALL") == "1":
        chosen = list(by_id)
    else:
        chosen = [m["id"] for m in models if m.get("format") == "onnx"]
else:
    chosen, unknown = [], []
    for s in selected:
        if s in by_id:
            rid = s
        elif s in alias:
            rid = alias[s]
            print(f"note: '{s}' is an old id; resolving to '{rid}'", file=sys.stderr)
        else:
            unknown.append(s)
            continue
        if rid not in chosen:
            chosen.append(rid)
    if unknown:
        print("ERROR: unknown model id(s): " + ", ".join(unknown), file=sys.stderr)
        print("Run with --list to see available models.", file=sys.stderr)
        sys.exit(1)

for rid in chosen:
    print(f"{rid}\t{by_id[rid]['zip']}")
PY
)"

# Parse resolved (id, zip) pairs into arrays.
IDS=(); ZIPS=()
while IFS=$'\t' read -r rid rzip; do
  [[ -z "$rid" ]] && continue
  IDS+=("$rid"); ZIPS+=("$rzip")
done <<< "$RESOLVED"

TOTAL_CATALOG="$(python3 -c 'import sys,tomllib;print(len(tomllib.load(open(sys.argv[1],"rb"))["model"]))' "$CATALOG")"

# ---- Prep ----
mkdir -p "$DEST"

echo "Sparrow Model Zoo v${ZENODO_VERSION}"
echo "Zenodo record: ${ZENODO_RECORD} (DOI ${ZENODO_DOI})"
echo "Destination:   $(realpath "$DEST")"
if [[ ${#SELECTED[@]} -gt 0 ]]; then
  SEL_NOTE="explicitly selected"
elif [[ $ALL -eq 1 ]]; then
  SEL_NOTE="all"
else
  SEL_NOTE="desktop ONNX default; --all for mobile too"
fi
echo "Models:        ${#IDS[@]} of ${TOTAL_CATALOG} (${SEL_NOTE})"
echo ""

# ---- Fetch per-file MD5 checksums once (from the Zenodo record API) ----
declare -A MD5
if [[ $VERIFY -eq 1 ]]; then
  echo "Fetching checksums from ${ZENODO_API} ..."
  API_TMP="$(mktemp)"
  if curl -fsSL "$ZENODO_API" -o "$API_TMP"; then
    while IFS=$'\t' read -r fkey fmd5; do
      [[ -n "$fkey" ]] && MD5["$fkey"]="$fmd5"
    done < <(python3 - "$API_TMP" <<'PY'
import sys, json
with open(sys.argv[1]) as fh:
    d = json.load(fh)
for f in d.get("files", []):
    ck = f.get("checksum", "")
    md5 = ck.split(":", 1)[1] if ck.startswith("md5:") else ""
    print(f"{f['key']}\t{md5}")
PY
)
    rm -f "$API_TMP"
    if [[ ${#MD5[@]} -eq 0 ]]; then
      echo "WARN: record API returned no file checksums; proceeding without MD5 verification" >&2
      VERIFY=0
    fi
  else
    rm -f "$API_TMP"
    echo "WARN: failed to fetch record API JSON; proceeding without MD5 verification" >&2
    VERIFY=0
  fi
fi

# ---- Download + unpack each model ----
for i in "${!IDS[@]}"; do
  m="${IDS[$i]}"
  zipname="${ZIPS[$i]}"
  echo ""
  echo "==> ${m}"

  # Skip only if the model is already fully present. `manifest.toml` covers
  # onnx/tflite models; `pipeline.toml` covers cascade descriptors.
  if [[ $FORCE -eq 0 ]] && [[ -f "$DEST/$m/manifest.toml" || -f "$DEST/$m/pipeline.toml" ]]; then
    echo "  already present; skipping. Use --force to re-download."
    continue
  fi

  ZIP_URL="$ZENODO_BASE/${zipname}"
  ZIP_PATH="$DEST/${zipname}"

  echo "  downloading ${zipname} ..."
  curl -fL --progress-bar -o "$ZIP_PATH" "$ZIP_URL"

  # MD5 verification against the Zenodo record API (v0.10.0 records ship no
  # checksums.sha256 file; the per-file API md5 is the source of truth).
  if [[ $VERIFY -eq 1 ]]; then
    expected="${MD5[$zipname]:-}"
    if [[ -z "$expected" ]]; then
      # The API returned checksums for other files but none for this one — a
      # genuine integrity gap. Fail closed (use --no-verify to override).
      echo "  [FAIL] no MD5 published for ${zipname}; refusing to install unverified." >&2
      echo "         re-run with --no-verify to skip integrity checks." >&2
      rm -f "$ZIP_PATH"
      exit 1
    fi
    actual="$(md5sum "$ZIP_PATH" | awk '{print $1}')"
    if [[ "$actual" == "$expected" ]]; then
      echo "  [OK] MD5 verified"
    else
      echo "  [FAIL] MD5 mismatch for ${zipname} (expected ${expected}, got ${actual})" >&2
      echo "         download is corrupt or tampered; aborting." >&2
      rm -f "$ZIP_PATH"
      exit 1
    fi
  fi

  echo "  unpacking..."
  # Unpack into a private staging dir, then atomically move `<id>/` into place.
  # This guarantees an interrupted unzip never leaves a half-written model dir
  # (manifest present but weights missing) that later runs would treat as
  # "already installed". The real `$DEST/<id>/` is only ever replaced on success.
  STAGE="$(mktemp -d "${DEST}/.stage.XXXXXX")"
  if ! unzip -q -o "$ZIP_PATH" -d "$STAGE"; then
    echo "  [FAIL] unzip failed for ${zipname}; leaving nothing behind." >&2
    rm -rf "$STAGE"; rm -f "$ZIP_PATH"
    exit 1
  fi
  rm -f "$ZIP_PATH"
  if [[ -d "$STAGE/$m" ]]; then
    # Guarded: $m is always a non-empty catalog id, but :? aborts if it isn't.
    rm -rf "${DEST:?}/${m:?}"
    mv "$STAGE/$m" "$DEST/$m"
  else
    echo "  WARN: ${zipname} did not contain ${m}/ at its root; not installed." >&2
  fi
  rm -rf "$STAGE"

  if [[ ! -d "$DEST/$m" ]]; then
    echo "  WARN: expected $DEST/$m/ after unpack but it is missing" >&2
  fi
done

# ---- Summary ----
echo ""
echo "======================================================================"
echo "Downloaded ${#IDS[@]} model(s) to: $(realpath "$DEST")"
echo ""
echo "Load with sparrow-engine:"
echo "  export SPARROW_ENGINE_MODEL_DIR=$(realpath "$DEST")"
echo "  spe list-models"
echo "  spe detect --model MDV6-yolov10-e --image /path/to/image.jpg"
echo ""
echo "If you use these models, please cite:"
echo "  Zenodo DOI: ${ZENODO_DOI}"
echo "  URL:        https://doi.org/${ZENODO_DOI}"
echo ""
echo "Per-model LICENSE.md inside each ${DEST}/<model_id>/ directory describes"
echo "the upstream license terms (mix of AGPL-3.0, CC-BY-NC-SA, Apache, MIT)."
echo "======================================================================"
