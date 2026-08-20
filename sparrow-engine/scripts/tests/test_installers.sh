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

# ===========================================================================
# expanded installer contract coverage — exit-code + idempotency.
# Uses ONLY run-owned HOME / prefix / local-release fixtures. No network
# (release fetch uses a local file:// fixture), no sudo, no real rc/install
# mutation, no layer bypassing/mocking. Keeps gates [1]-[5] above.
# ===========================================================================

expect_rc() {
    # expect_rc WANT LABEL -- CMD...
    local want=$1 label=$2
    shift 2
    [[ "${1:-}" == "--" ]] && shift
    local rc=0
    "$@" >"$TEST_ROOT/last.out" 2>&1 || rc=$?
    [[ "$rc" == "$want" ]] ||
        fail "$label: expected exit $want, got $rc :: $(tail -2 "$TEST_ROOT/last.out" | tr '\n' ' ')"
}

# OS/ARCH exactly as the installer's detect_os_arch computes them.
case "$(uname -s)" in Linux) TEST_OS=linux ;; Darwin) TEST_OS=macos ;; *) TEST_OS=linux ;; esac
case "$(uname -m)" in x86_64|amd64) TEST_ARCH=x86_64 ;; arm64|aarch64) TEST_ARCH=aarch64 ;; *) TEST_ARCH=x86_64 ;; esac

# Local release fixture: a valid CLI tarball + .sha256 sidecar the installer
# fetches via file:// (curl), so --cli installs complete with zero network.
RELDIR="$TEST_ROOT/release"
FX_ROOT="sparrow-engine-cpu-$cli_version-$TEST_OS-$TEST_ARCH"
mkdir -p "$RELDIR" "$TEST_ROOT/build/$FX_ROOT/bin" "$TEST_ROOT/build/$FX_ROOT/lib"
printf '#!/bin/sh\necho "spe fixture %s"\n' "$cli_version" >"$TEST_ROOT/build/$FX_ROOT/bin/spe"
chmod +x "$TEST_ROOT/build/$FX_ROOT/bin/spe"
: >"$TEST_ROOT/build/$FX_ROOT/lib/libsparrow_engine.so"
tar -czf "$RELDIR/$FX_ROOT.tar.gz" -C "$TEST_ROOT/build" "$FX_ROOT"
sha256sum -b "$RELDIR/$FX_ROOT.tar.gz" >"$RELDIR/$FX_ROOT.tar.gz.sha256"
FILE_BASE="file://$RELDIR"

echo "[6] shell installer rejects a --flavor / env-var disagreement with exit 3"
expect_rc 3 "shell conflict env=gpu --flavor cpu" -- \
    env SPARROW_ENGINE_PREFIX="$TEST_ROOT/c6" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        SPARROW_ENGINE_INSTALL_FLAVOR=gpu bash "$SH_INSTALLER" --cli --flavor cpu -y
grep -qi 'flavor disagreement' "$TEST_ROOT/last.out" ||
    fail "shell exit-3 message not actionable"
[[ ! -e "$TEST_ROOT/c6" ]] ||
    fail "shell conflict mutated state before failing"
# --flavor auto ignores the env override: --probe-only resolves the flavor via
# the hardware probe and exits 0 BEFORE gpu_quality_check, so a degraded host
# GPU cannot spuriously turn this flavor-resolution assertion into exit 11.
expect_rc 0 "shell --flavor auto ignores env" -- \
    env SPARROW_ENGINE_PREFIX="$TEST_ROOT/c6a" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        SPARROW_ENGINE_INSTALL_FLAVOR=gpu bash "$SH_INSTALLER" --cli --flavor auto --probe-only -y
# Matching explicit + env continues normally.
expect_rc 0 "shell matching explicit+env" -- \
    env SPARROW_ENGINE_PREFIX="$TEST_ROOT/c6m" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        SPARROW_ENGINE_INSTALL_FLAVOR=cpu bash "$SH_INSTALLER" --cli --flavor cpu --dry-run -y

echo "[7] PowerShell installer rejects a -Flavor / env-var disagreement with exit 3"
if command -v pwsh >/dev/null 2>&1; then
    # PS wrapper is Windows-targeted; run-owned Windows env fixtures
    # (LOCALAPPDATA/USERPROFILE/SystemRoot/TEMP) let its setup block AND probe.ps1
    # run under Linux pwsh without touching real user state.
    ps_win="$TEST_ROOT/win"
    mkdir -p "$ps_win/localappdata" "$ps_win/userprofile" "$ps_win/systemroot" "$ps_win/temp"
    ps_env=(LOCALAPPDATA="$ps_win/localappdata" USERPROFILE="$ps_win/userprofile"
            SystemRoot="$ps_win/systemroot" TEMP="$ps_win/temp" TMP="$ps_win/temp")
    expect_rc 3 "PowerShell conflict env=gpu -Flavor cpu" -- \
        env "${ps_env[@]}" SPARROW_ENGINE_INSTALL_FLAVOR=gpu \
            pwsh -NoProfile -NonInteractive -File "$PS_INSTALLER" -Cli -Flavor cpu -ProbeOnly -Yes
    grep -qi 'flavor disagreement' "$TEST_ROOT/last.out" ||
        fail "PowerShell exit-3 message not actionable"
    # Matching explicit+env: -ProbeOnly resolves the flavor and exits BEFORE the
    # install / Windows-arch step, so the match is an EXACT 0 (not merely "not 3")
    # and must not report a disagreement.
    expect_rc 0 "PowerShell matching explicit+env (-ProbeOnly)" -- \
        env "${ps_env[@]}" SPARROW_ENGINE_INSTALL_FLAVOR=cpu \
            pwsh -NoProfile -NonInteractive -File "$PS_INSTALLER" -Cli -Flavor cpu -ProbeOnly -Yes
    ! grep -qi 'flavor disagreement' "$TEST_ROOT/last.out" ||
        fail "PowerShell matching explicit+env must not report a disagreement"
    # -Flavor auto ignores the env override entirely: -ProbeOnly must exit EXACTLY
    # 0 and report no disagreement even with SPARROW_ENGINE_INSTALL_FLAVOR set.
    expect_rc 0 "PowerShell --flavor auto ignores env (-ProbeOnly)" -- \
        env "${ps_env[@]}" SPARROW_ENGINE_INSTALL_FLAVOR=gpu \
            pwsh -NoProfile -NonInteractive -File "$PS_INSTALLER" -Cli -Flavor auto -ProbeOnly -Yes
    ! grep -qi 'flavor disagreement' "$TEST_ROOT/last.out" ||
        fail "PowerShell --flavor auto must not report a disagreement"
else
    echo "SKIP: pwsh not installed; PowerShell conflict runtime check unavailable"
fi

echo "[8] required tool missing surfaces exit 8"
# Run-owned PATH with every current tool EXCEPT curl (installer needs curl for --cli).
FAKEBIN="$TEST_ROOT/fakebin-nocurl"
mkdir -p "$FAKEBIN"
_oldifs="$IFS"; IFS=:
for _d in $PATH; do
    [ -d "$_d" ] || continue
    for _f in "$_d"/*; do
        [ -e "$_f" ] || continue
        _b="${_f##*/}"
        [ -e "$FAKEBIN/$_b" ] || ln -s "$_f" "$FAKEBIN/$_b" 2>/dev/null || true
    done
done
IFS="$_oldifs"
rm -f "$FAKEBIN/curl"
command -v curl >/dev/null 2>&1 || fail "test harness has no curl to base the exit-8 fixture on"
expect_rc 8 "required-tool exit 8 (curl absent)" -- \
    env PATH="$FAKEBIN" HOME="$TEST_ROOT/h8" \
        SPARROW_ENGINE_PREFIX="$TEST_ROOT/p8" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        bash "$SH_INSTALLER" --cli --flavor cpu -y

echo "[9] GPU-quality failure surfaces exit 11 (genuine isolated install; never SKIP)"
# Determinism requires the ACTUAL shell installer to reach `die 11` regardless
# of what CUDA/cuDNN the host exposes. We build a run-owned minimal rootfs that
# contains ONLY bash + uname + basename + dirname (the tools the installer uses
# before die 11: detect_os_arch -> uname; locate_helper -> basename/dirname;
# probe sourcing -> bash) plus each tool's ldd deps + dynamic loader, plus the
# real installer and its adjacent probe_gpu_quality.sh. Inside that rootfs NO
# GPU runtime sidecar (libcudart.so.12 …) exists, so probe_gpu_quality's very
# first check (pure-bash sidecar search) returns cudnn_err and the installer
# dies 11 — the real code path, no mocked probe, no overridden tools.
_G9_LIBDIRS=""
_g9_add_libdir() {
    case ":$_G9_LIBDIRS:" in
        *":$1:"*) ;;
        *) _G9_LIBDIRS="${_G9_LIBDIRS:+$_G9_LIBDIRS:}$1" ;;
    esac
}
_g9_copy_path() {  # copy an absolute host lib to the identical path under $G9_ROOT
    local p="$1" d
    [ -n "$p" ] && [ -e "$p" ] || return 0
    d="$(dirname "$p")"
    mkdir -p "$G9_ROOT$d"
    [ -e "$G9_ROOT$p" ] || cp -L "$p" "$G9_ROOT$p" 2>/dev/null || true
    _g9_add_libdir "$d"
}
_g9_copy_bin() {  # copy a tool into $G9_ROOT/bin + all its ldd deps (identical paths)
    local name="$1" real deps line p
    real="$(command -v "$name")" || return 20
    cp -L "$real" "$G9_ROOT/bin/$name" || return 21
    deps="$(ldd "$real" 2>/dev/null || true)"
    while IFS= read -r line; do
        p=""
        case "$line" in
            *"=>"*) p="$(printf '%s\n' "$line" | awk '{print $3}')" ;;
            */*)    p="$(printf '%s\n' "$line" | awk '{print $1}')" ;;
        esac
        case "$p" in /*) _g9_copy_path "$p" ;; esac
    done <<G9DEPS
$deps
G9DEPS
}
_g9_build_rootfs() {
    _G9_LIBDIRS=""
    mkdir -p "$G9_ROOT/bin" "$G9_ROOT/opt/inst" "$G9_ROOT/dev" "$G9_ROOT/root"
    local t
    for t in bash uname basename dirname; do
        _g9_copy_bin "$t" || fail "gate[9]: could not stage '$t' (+deps) into the rootfs"
    done
    cp -L "$SH_INSTALLER" "$G9_ROOT/opt/inst/sparrow-engine-install.sh" ||
        fail "gate[9]: could not stage the installer"
    cp -L "$INSTALLER_DIR/probe_gpu_quality.sh" "$G9_ROOT/opt/inst/probe_gpu_quality.sh" ||
        fail "gate[9]: could not stage probe_gpu_quality.sh"
    : > "$G9_ROOT/dev/null"
}
_g9_run_unshare() {  # unprivileged user namespace + chroot; returns the installer's exit code
    unshare --user --map-root-user --mount bash -c '
        set -e
        R="$1"; LD="$2"
        # /dev/null inside the namespace (bind the host device over the seed file;
        # if the bind is refused the seed regular file still absorbs redirects).
        mount --bind /dev/null "$R/dev/null" 2>/dev/null || true
        exec chroot "$R" /bin/bash -c \
            "export PATH=/bin HOME=/root LD_LIBRARY_PATH=$LD; exec /bin/bash /opt/inst/sparrow-engine-install.sh --cli --flavor gpu -y"
    ' _ "$G9_ROOT" "$_G9_LIBDIRS"
}
_g9_run_docker() {  # executable fallback: import the SAME run-owned rootfs, run --network none
    local tag="sparrow-engine-installer-exit11-$$-${RANDOM}:fixture" rc=0
    tar -C "$G9_ROOT" -c . | docker import - "$tag" >/dev/null 2>&1 ||
        { echo "docker import failed"; return 90; }
    docker run --rm --network none \
        -e PATH=/bin -e HOME=/root -e LD_LIBRARY_PATH="$_G9_LIBDIRS" \
        "$tag" /bin/bash /opt/inst/sparrow-engine-install.sh --cli --flavor gpu -y || rc=$?
    docker rmi "$tag" >/dev/null 2>&1 || true   # remove that exact image
    return "$rc"
}
if [[ "$TEST_OS" == "linux" ]]; then
    G9_ROOT="$TEST_ROOT/g9-rootfs"
    _g9_build_rootfs
    if command -v unshare >/dev/null 2>&1 && command -v chroot >/dev/null 2>&1 \
       && command -v mount >/dev/null 2>&1 \
       && unshare --user --map-root-user --mount true >/dev/null 2>&1; then
        echo "    isolation: unprivileged user namespace (unshare --user --map-root-user --mount + chroot)"
        expect_rc 11 "gpu-quality exit 11 (isolated rootfs, unshare+chroot)" -- _g9_run_unshare
    elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        echo "    isolation: docker (run-owned imported rootfs, --network none)"
        expect_rc 11 "gpu-quality exit 11 (isolated rootfs, docker --network none)" -- _g9_run_docker
    else
        fail "gate[9]: no isolation available for a deterministic exit-11 test — need unprivileged user namespaces (unshare --user --map-root-user) or a working docker daemon. Refusing to SKIP GPU-quality coverage."
    fi
    grep -qiE 'sidecar|cudnn|gpu runtime' "$TEST_ROOT/last.out" ||
        fail "gate[9]: exit-11 path did not emit the expected GPU-runtime/cuDNN diagnostic"
else
    # Non-Linux: only a native empty-HOME path is available. Require a genuine 11.
    g9h="$TEST_ROOT/g9-home"
    mkdir -p "$g9h"
    g9rc=0
    env HOME="$g9h" SPARROW_ENGINE_PREFIX="$TEST_ROOT/p9" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        bash "$SH_INSTALLER" --cli --flavor gpu -y >"$TEST_ROOT/last.out" 2>&1 || g9rc=$?
    [[ "$g9rc" == 11 ]] ||
        fail "gate[9]: non-Linux host did not deterministically surface exit 11 via an empty HOME (got $g9rc); run this gate on Linux (unprivileged user namespaces or docker), or supply a container image without GPU runtime sidecars"
fi

echo "[10] cross-flavor install without --reprobe surfaces exit 12"
p10="$TEST_ROOT/p10"
mkdir -p "$p10"
printf '{\n  "flavor": "gpu",\n  "mode": "cli",\n  "version": "%s"\n}\n' "$cli_version" >"$p10/installed.json"
expect_rc 12 "cross-flavor refusal exit 12" -- \
    env HOME="$TEST_ROOT/h10" SPARROW_ENGINE_PREFIX="$p10" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        SPARROW_ENGINE_RELEASE_BASE="$FILE_BASE" SPARROW_ENGINE_VERSION="$cli_version" \
        bash "$SH_INSTALLER" --cli --flavor cpu -y
grep -q '"flavor"' "$p10/installed.json" && grep -q 'gpu' "$p10/installed.json" ||
    fail "cross-flavor refusal must not rewrite the existing state file"

echo "[11] manual rc-file edit surfaces exit 13 (local release fixture; run-owned HOME)"
h11="$TEST_ROOT/h11"
mkdir -p "$h11"
printf '%s\nexport PATH="/manually/edited/path:$PATH"\n%s\n' \
    '# >>> sparrow-engine >>>' '# <<< sparrow-engine <<<' >"$h11/.bashrc"
expect_rc 13 "rc-edit exit 13" -- \
    env HOME="$h11" SPARROW_ENGINE_PREFIX="$TEST_ROOT/p11" \
        SPARROW_ENGINE_RELEASE_BASE="$FILE_BASE" SPARROW_ENGINE_VERSION="$cli_version" \
        bash "$SH_INSTALLER" --cli --flavor cpu -y
grep -q '/manually/edited/path' "$h11/.bashrc" ||
    fail "exit-13 path must not overwrite the manually edited rc block"

echo "[12] same-flavor reinstall + cross-flavor reprobe idempotency (local release fixture)"
h12="$TEST_ROOT/h12"; p12="$TEST_ROOT/p12"
mkdir -p "$h12"
# Fresh CPU install via the local fixture (NO_MODIFY_PATH keeps rc files untouched).
expect_rc 0 "fresh cpu install (file:// fixture)" -- \
    env HOME="$h12" SPARROW_ENGINE_PREFIX="$p12" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        SPARROW_ENGINE_RELEASE_BASE="$FILE_BASE" SPARROW_ENGINE_VERSION="$cli_version" \
        bash "$SH_INSTALLER" --cli --flavor cpu -y
[[ -x "$p12/bin/spe" ]] || fail "fresh install did not place bin/spe"
grep -q 'cpu' "$p12/installed.json" || fail "fresh install did not record cpu state"
# Same-flavor re-invocation with no --reinstall = soft no-op (exit 0, no download).
expect_rc 0 "same-flavor no-op (no --reinstall)" -- \
    env HOME="$h12" SPARROW_ENGINE_PREFIX="$p12" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        SPARROW_ENGINE_RELEASE_BASE="$FILE_BASE" SPARROW_ENGINE_VERSION="$cli_version" \
        bash "$SH_INSTALLER" --cli --flavor cpu -y
grep -qi 'already installed' "$TEST_ROOT/last.out" ||
    fail "same-flavor no-op message missing"
# Same-flavor --reinstall force-overwrites (exit 0).
expect_rc 0 "same-flavor --reinstall" -- \
    env HOME="$h12" SPARROW_ENGINE_PREFIX="$p12" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        SPARROW_ENGINE_RELEASE_BASE="$FILE_BASE" SPARROW_ENGINE_VERSION="$cli_version" \
        bash "$SH_INSTALLER" --cli --flavor cpu --reinstall -y
# Cross-flavor switch WITH --reprobe (gpu state -> cpu; cpu path skips GPU quality).
p12b="$TEST_ROOT/p12b"; h12b="$TEST_ROOT/h12b"
mkdir -p "$h12b" "$p12b"
printf '{\n  "flavor": "gpu",\n  "mode": "cli",\n  "version": "%s"\n}\n' "$cli_version" >"$p12b/installed.json"
expect_rc 0 "cross-flavor reprobe switch (gpu->cpu)" -- \
    env HOME="$h12b" SPARROW_ENGINE_PREFIX="$p12b" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        SPARROW_ENGINE_RELEASE_BASE="$FILE_BASE" SPARROW_ENGINE_VERSION="$cli_version" \
        bash "$SH_INSTALLER" --cli --flavor cpu --reprobe -y
grep -q 'cpu' "$p12b/installed.json" || fail "reprobe did not switch state to cpu"
# Idempotency: re-running the now-current flavor is a no-op (exit 0).
expect_rc 0 "reprobe idempotency (no-op)" -- \
    env HOME="$h12b" SPARROW_ENGINE_PREFIX="$p12b" SPARROW_ENGINE_NO_MODIFY_PATH=1 \
        SPARROW_ENGINE_RELEASE_BASE="$FILE_BASE" SPARROW_ENGINE_VERSION="$cli_version" \
        bash "$SH_INSTALLER" --cli --flavor cpu -y
grep -qi 'already installed' "$TEST_ROOT/last.out" ||
    fail "reprobe idempotency no-op message missing"

echo "PASS: installer version, repository, mode, truncation, flavor-conflict (exit 3, shell + PowerShell), required-tool (8), GPU-quality (11), cross-flavor (12), rc-edit (13), and reinstall/reprobe idempotency gates"
