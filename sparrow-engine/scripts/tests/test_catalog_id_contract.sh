#!/usr/bin/env bash
# Guard: retired public model aliases must not resurface in the CURRENT shipping
# surfaces — the CLI main, the nvJPEG dlopen script, and the public user manual.
# Also confirms the current detector default is a real catalog id and that the
# CLI + nvJPEG script use it.
#
# Scope is deliberately limited to those three user-facing surfaces. It does NOT
# scan legacy integration fixtures under tests/ — their retired IDs are
# intentional historical test data, not a public contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARROW_ENGINE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # .../sparrow-engine
REPO_ROOT="$(cd "$SPARROW_ENGINE_DIR/.." && pwd)"

CLI_MAIN="$SPARROW_ENGINE_DIR/sparrow-engine-cli/src/main.rs"
NVJPEG="$SPARROW_ENGINE_DIR/scripts/test_nvjpeg_dlopen.sh"
USER_MANUAL="$REPO_ROOT/docs/user-manual.md"
CATALOG="$SPARROW_ENGINE_DIR/scripts/catalog.toml"
DEFAULT_DETECTOR="MDV6-yolov10-e"

fail() { echo "FAIL: $*" >&2; exit 1; }

SURFACES=("$CLI_MAIN" "$NVJPEG" "$USER_MANUAL")

echo "[1] retired public aliases must not appear in the current shipping surfaces"
for f in "${SURFACES[@]}"; do
    [ -f "$f" ] || fail "missing surface: $f"
    b="$(basename "$f")"
    # The CLI-main surface is the user-facing command code (args, defaults,
    # help). Its `#[cfg(test)]` unit-test module uses generic, arbitrary model
    # IDs as pipeline-validation fixtures; those test-only IDs are intentional
    # and out of scope for the retired-alias guard, so scan only the code above
    # the test module for main.rs.
    if [ "$f" = "$CLI_MAIN" ]; then
        content="$(sed '/^#\[cfg(test)\]/,$d' "$f")"
    else
        content="$(cat "$f")"
    fi
    # Most specific first. `-F` = fixed-string, case-SENSITIVE (so the current
    # `SpeciesNet-Crop` id is NOT flagged by the retired lowercase `speciesnet-crop`).
    if printf '%s\n' "$content" | grep -Fq 'megadetector-v6-yolov10e-prov'; then
        fail "$b references the nonexistent 'megadetector-v6-yolov10e-prov' (provenance is manifest metadata, not a catalog model)"
    fi
    if printf '%s\n' "$content" | grep -Fq 'megadetector-v6-yolov10e'; then
        fail "$b still references the retired detector alias 'megadetector-v6-yolov10e' (current id: $DEFAULT_DETECTOR)"
    fi
    if printf '%s\n' "$content" | grep -Fq 'speciesnet-crop'; then
        fail "$b references the retired lowercase alias 'speciesnet-crop' (current id: SpeciesNet-Crop)"
    fi
    if printf '%s\n' "$content" | grep -Fq 'megadet-speciesnet'; then
        fail "$b claims the retired 'megadet-speciesnet' (use the custom alias mdv6-speciesnet; it is not a catalog entry)"
    fi
done

echo "[2] the current detector default is a real catalog id"
grep -Fq "id = \"$DEFAULT_DETECTOR\"" "$CATALOG" \
    || fail "detector default $DEFAULT_DETECTOR is not present in scripts/catalog.toml"

echo "[3] CLI main carries the current detector default constant"
grep -Fq "\"$DEFAULT_DETECTOR\"" "$CLI_MAIN" \
    || fail "sparrow-engine-cli/src/main.rs does not reference the current detector default $DEFAULT_DETECTOR"

echo "[4] nvJPEG dlopen script defaults to the current detector id"
grep -Fq "$DEFAULT_DETECTOR" "$NVJPEG" \
    || fail "test_nvjpeg_dlopen.sh does not default to $DEFAULT_DETECTOR"

echo "PASS: catalog id contract (no retired aliases in current surfaces; current detector default present in catalog + CLI + nvJPEG)"
