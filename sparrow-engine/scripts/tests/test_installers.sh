#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$ENGINE_DIR/.." && pwd)"
INSTALLER_DIR="$REPO_ROOT/installer"
SH_INSTALLER="$INSTALLER_DIR/sparrow-engine-install.sh"
PS_INSTALLER="$INSTALLER_DIR/sparrow-engine-install.ps1"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

cli_version="$(
    awk '
        /^\[package\][[:space:]]*$/ { in_package = 1; next }
        in_package && /^\[/ { in_package = 0 }
        in_package && /^version[[:space:]]*=/ {
            match($0, /"[^"]+"/)
            print substr($0, RSTART + 1, RLENGTH - 2)
            exit
        }
    ' "$ENGINE_DIR/sparrow-engine-cli/Cargo.toml"
)"

echo "[1] installer defaults match the engine version"
sh_version="$(bash "$SH_INSTALLER" --version)"
ps_version="$(
    sed -nE "s/.*else \\{ '([^']+)' \\}.*/\\1/p" "$PS_INSTALLER" | head -1
)"
[[ "$sh_version" == "$cli_version" ]] ||
    fail "shell installer version $sh_version != engine version $cli_version"
[[ "$ps_version" == "$cli_version" ]] ||
    fail "PowerShell installer version $ps_version != engine version $cli_version"

echo "[2] installer release and helper URLs use the current repository"
grep -Fq "github.com/microsoft/SPARROW-Engine/releases/download" "$SH_INSTALLER" ||
    fail "shell release URL is stale"
grep -Fq "raw.githubusercontent.com/microsoft/SPARROW-Engine" "$SH_INSTALLER" ||
    fail "shell helper URL is stale"
grep -Fq "github.com/microsoft/SPARROW-Engine/releases/download" "$PS_INSTALLER" ||
    fail "PowerShell release URL is stale"
grep -Fq "raw.githubusercontent.com/microsoft/SPARROW-Engine" "$PS_INSTALLER" ||
    fail "PowerShell helper URL is stale"

echo "[3] complete shell installer parses and dry-runs all modes"
bash -n "$SH_INSTALLER"
for mode in pip cli docker; do
    env \
        SPARROW_ENGINE_PREFIX="$TEST_ROOT/prefix-$mode" \
        SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        bash "$SH_INSTALLER" "--$mode" --flavor cpu --dry-run -y \
        >"$TEST_ROOT/$mode.out" 2>&1
done

echo "[4] shell installer tail truncation fails before execution"
head -n -1 "$SH_INSTALLER" >"$TEST_ROOT/truncated-install.sh"
if bash -n "$TEST_ROOT/truncated-install.sh" >/dev/null 2>&1; then
    fail "truncated shell installer still parses"
fi
if env \
    SPARROW_ENGINE_PREFIX="$TEST_ROOT/truncated-prefix" \
    SPARROW_ENGINE_NO_MODIFY_PATH=1 \
    bash "$TEST_ROOT/truncated-install.sh" --cli --flavor cpu --dry-run -y \
    >"$TEST_ROOT/truncated.out" 2>&1; then
    fail "truncated shell installer exited successfully"
fi
[[ ! -e "$TEST_ROOT/truncated-prefix" ]] ||
    fail "truncated shell installer created an install prefix"

echo "[5] complete PowerShell installer parses; truncated tail does not"
if command -v pwsh >/dev/null 2>&1; then
    env PS_PARSE_PATH="$PS_INSTALLER" pwsh -NoProfile -NonInteractive -Command \
        '$errors = $null; [void][System.Management.Automation.Language.Parser]::ParseFile($env:PS_PARSE_PATH, [ref]$null, [ref]$errors); if ($errors.Count) { exit 1 }'
    head -n -1 "$PS_INSTALLER" >"$TEST_ROOT/truncated-install.ps1"
    if env PS_PARSE_PATH="$TEST_ROOT/truncated-install.ps1" pwsh -NoProfile -NonInteractive -Command \
        '$errors = $null; [void][System.Management.Automation.Language.Parser]::ParseFile($env:PS_PARSE_PATH, [ref]$null, [ref]$errors); if ($errors.Count) { exit 1 }'; then
        fail "truncated PowerShell installer still parses"
    fi
else
    echo "SKIP: pwsh not installed; PowerShell parse check unavailable"
fi

echo "PASS: installer version, repository, mode, and truncation gates"
