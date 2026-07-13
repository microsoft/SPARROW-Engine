#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOADER="$SCRIPT_DIR/download_models.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

CATALOG="$TEST_ROOT/catalog.toml"
FAKE_BIN="$TEST_ROOT/bin"
FAKE_CURL_LOG="$TEST_ROOT/curl.log"
FAKE_ZIP="$TEST_ROOT/test-model.zip"
FAKE_ZIP_TWO="$TEST_ROOT/test-model-two.zip"
ZIP_NAME="camera_trap__classifier__test-model.zip"
ZIP_NAME_TWO="camera_trap__classifier__test-model-two.zip"
API_URL_FRAGMENT="/api/records/999"
FILE_URL_FRAGMENT="/records/999/files/$ZIP_NAME"
FILE_URL_FRAGMENT_TWO="/records/999/files/$ZIP_NAME_TWO"
mkdir -p "$FAKE_BIN" "$TEST_ROOT/home"

cat > "$CATALOG" <<'EOF'
schema_version = "1.1"

[zenodo]
record = "999"
concept_doi = "10.5281/zenodo.999"
version = "test"

[[model]]
id = "test-model"
domain = "camera_trap"
task = "classifier"
format = "onnx"
license = "MIT"
zip = "camera_trap__classifier__test-model.zip"

[[model]]
id = "test-model-two"
domain = "camera_trap"
task = "classifier"
format = "onnx"
license = "MIT"
zip = "camera_trap__classifier__test-model-two.zip"
EOF

python3 - "$FAKE_ZIP" "$FAKE_ZIP_TWO" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1], "w") as archive:
    archive.writestr("test-model/manifest.toml", 'id = "test-model"\n')
with zipfile.ZipFile(sys.argv[2], "w") as archive:
    archive.writestr("test-model-two/manifest.toml", 'id = "test-model-two"\n')
PY
EXPECTED_MD5="$(md5sum "$FAKE_ZIP" | awk '{print $1}')"
EXPECTED_MD5_TWO="$(md5sum "$FAKE_ZIP_TWO" | awk '{print $1}')"

cat > "$FAKE_BIN/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

url=""
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)
      output="$2"
      shift 2
      ;;
    http://*|https://*)
      url="$1"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

printf '%s\n' "$url" >> "$FAKE_CURL_LOG"
if [[ "$url" == *"/api/records/"* ]]; then
  case "$FAKE_API_MODE" in
    fail)
      exit 22
      ;;
    empty)
      printf '{"files":[]}\n' > "$output"
      ;;
    blank)
      printf '{"files":[{"key":"%s","checksum":""}]}\n' \
        "$FAKE_ZIP_NAME" > "$output"
      ;;
    missing)
      printf '{"files":[{"key":"other.zip","checksum":"md5:%s"}]}\n' \
        "$FAKE_API_MD5" > "$output"
      ;;
    valid)
      printf '{"files":[{"key":"%s","checksum":"md5:%s"}]}\n' \
        "$FAKE_ZIP_NAME" "$FAKE_API_MD5" > "$output"
      ;;
    valid_both)
      printf '{"files":[{"key":"%s","checksum":"md5:%s"},{"key":"%s","checksum":"md5:%s"}]}\n' \
        "$FAKE_ZIP_NAME" "$FAKE_API_MD5" \
        "$FAKE_ZIP_NAME_TWO" "$FAKE_API_MD5_TWO" > "$output"
      ;;
    mismatch)
      printf '{"files":[{"key":"%s","checksum":"md5:00000000000000000000000000000000"}]}\n' \
        "$FAKE_ZIP_NAME" > "$output"
      ;;
    *)
      echo "unknown FAKE_API_MODE: $FAKE_API_MODE" >&2
      exit 24
      ;;
  esac
  exit 0
fi
if [[ -z "$output" || "$url" != *"/records/"*"/files/"* ]]; then
  exit 23
fi
case "$url" in
  *"/$FAKE_ZIP_NAME")
    cp "$FAKE_ZIP" "$output"
    ;;
  *"/$FAKE_ZIP_NAME_TWO")
    cp "$FAKE_ZIP_TWO" "$output"
    ;;
  *)
    exit 25
    ;;
esac
EOF
chmod +x "$FAKE_BIN/curl"

run_downloader() {
  local api_mode="$1"
  shift
  env \
    -u ZENODO_RECORD \
    -u SPARROW_ALL \
    -u SPARROW_ENGINE_MODEL_DIR \
    HOME="$TEST_ROOT/home" \
    PATH="$FAKE_BIN:$PATH" \
    FAKE_API_MODE="$api_mode" \
    FAKE_API_MD5="$EXPECTED_MD5" \
    FAKE_API_MD5_TWO="$EXPECTED_MD5_TWO" \
    FAKE_CURL_LOG="$FAKE_CURL_LOG" \
    FAKE_ZIP="$FAKE_ZIP" \
    FAKE_ZIP_TWO="$FAKE_ZIP_TWO" \
    FAKE_ZIP_NAME="$ZIP_NAME" \
    FAKE_ZIP_NAME_TWO="$ZIP_NAME_TWO" \
    SPARROW_CATALOG="$CATALOG" \
    bash "$DOWNLOADER" "$@"
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

assert_absent() {
  local needle="$1"
  local file="$2"
  if grep -Fq "$needle" "$file"; then
    fail "unexpected '$needle' in $file"
  fi
}

assert_file_absent() {
  [[ ! -e "$1" ]] || fail "unexpected path exists: $1"
}

expect_failure() {
  local output="$1"
  local api_mode="$2"
  shift 2
  if run_downloader "$api_mode" "$@" > "$output" 2>&1; then
    fail "expected downloader failure: $*"
  fi
}

reset_log() {
  : > "$FAKE_CURL_LOG"
}

echo "[1] unavailable checksum API reuses an existing unstamped model"
installed="$TEST_ROOT/installed"
mkdir -p "$installed/test-model"
printf 'id = "test-model"\n' > "$installed/test-model/manifest.toml"
reset_log
run_downloader fail --dest "$installed" test-model > "$TEST_ROOT/existing.out" 2>&1
assert_contains "failed to fetch record API JSON" "$TEST_ROOT/existing.out"
assert_contains "Existing models may be reused" "$TEST_ROOT/existing.out"
assert_contains "already present; skipping (checksum not verified" "$TEST_ROOT/existing.out"
assert_contains "$API_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_absent "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"

echo "[2] unavailable checksum API blocks a new verified download"
fresh="$TEST_ROOT/fresh"
reset_log
expect_failure "$TEST_ROOT/fresh.out" fail --dest "$fresh" test-model
assert_contains "refusing to download unverified bytes" "$TEST_ROOT/fresh.out"
assert_absent "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_file_absent "$fresh/test-model"
assert_file_absent "$fresh/$ZIP_NAME"

echo "[3] empty checksum response remains distinct from transport failure"
empty="$TEST_ROOT/empty"
reset_log
expect_failure "$TEST_ROOT/empty.out" empty --dest "$empty" test-model
assert_contains "record API returned no file checksums" "$TEST_ROOT/empty.out"
assert_contains "refusing to download unverified bytes" "$TEST_ROOT/empty.out"
assert_absent "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"

echo "[4] --no-verify is the explicit opt-out"
unverified="$TEST_ROOT/unverified"
reset_log
run_downloader fail --no-verify --dest "$unverified" test-model \
  > "$TEST_ROOT/unverified.out" 2>&1
[[ -f "$unverified/test-model/manifest.toml" ]]
assert_absent "$API_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_contains "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_file_absent "$unverified/test-model/.sparrow_zip_md5"

echo "[5] valid checksum installs and stamps a fresh model"
verified="$TEST_ROOT/verified"
reset_log
run_downloader valid --dest "$verified" test-model > "$TEST_ROOT/verified.out" 2>&1
assert_contains "[OK] MD5 verified" "$TEST_ROOT/verified.out"
assert_contains "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
[[ -f "$verified/test-model/manifest.toml" ]] || fail "verified model missing"
[[ "$(cat "$verified/test-model/.sparrow_zip_md5")" == "$EXPECTED_MD5" ]] \
  || fail "verified install stamp does not match downloaded ZIP"

echo "[6] per-file missing checksum fails before model transfer"
missing="$TEST_ROOT/missing"
reset_log
expect_failure "$TEST_ROOT/missing.out" missing --dest "$missing" test-model
assert_contains "no MD5 available for $ZIP_NAME" "$TEST_ROOT/missing.out"
assert_absent "record API returned no file checksums" "$TEST_ROOT/missing.out"
assert_absent "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_file_absent "$missing/test-model"

echo "[7] per-file missing checksum still reuses an existing model"
missing_existing="$TEST_ROOT/missing-existing"
mkdir -p "$missing_existing/test-model"
printf 'id = "test-model"\n' > "$missing_existing/test-model/manifest.toml"
reset_log
run_downloader missing --dest "$missing_existing" test-model \
  > "$TEST_ROOT/missing-existing.out" 2>&1
assert_contains "already present; skipping (checksum not verified" \
  "$TEST_ROOT/missing-existing.out"
assert_absent "record API returned no file checksums" \
  "$TEST_ROOT/missing-existing.out"
assert_contains "$API_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_absent "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"

echo "[8] matching install stamp skips an up-to-date model"
current="$TEST_ROOT/current"
mkdir -p "$current/test-model"
printf 'id = "test-model"\n' > "$current/test-model/manifest.toml"
printf '%s\n' "$EXPECTED_MD5" > "$current/test-model/.sparrow_zip_md5"
reset_log
run_downloader valid --dest "$current" test-model > "$TEST_ROOT/current.out" 2>&1
assert_contains "checksum up to date" "$TEST_ROOT/current.out"
assert_absent "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"

echo "[9] older unstamped installs retain the documented skip behavior"
older="$TEST_ROOT/older"
mkdir -p "$older/test-model"
printf 'id = "test-model"\n' > "$older/test-model/manifest.toml"
reset_log
run_downloader valid --dest "$older" test-model > "$TEST_ROOT/older.out" 2>&1
assert_contains "not checksum-stamped" "$TEST_ROOT/older.out"
assert_absent "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"

echo "[10] stale install stamp triggers verified replacement"
stale="$TEST_ROOT/stale"
mkdir -p "$stale/test-model"
printf 'id = "test-model"\n' > "$stale/test-model/manifest.toml"
printf 'stale-md5\n' > "$stale/test-model/.sparrow_zip_md5"
reset_log
run_downloader valid --dest "$stale" test-model > "$TEST_ROOT/stale.out" 2>&1
assert_contains "local copy is STALE" "$TEST_ROOT/stale.out"
assert_contains "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
[[ "$(cat "$stale/test-model/.sparrow_zip_md5")" == "$EXPECTED_MD5" ]] \
  || fail "stale model was not re-stamped"

echo "[11] checksum mismatch aborts and removes downloaded artifacts"
mismatch="$TEST_ROOT/mismatch"
reset_log
expect_failure "$TEST_ROOT/mismatch.out" mismatch --dest "$mismatch" test-model
assert_contains "MD5 mismatch" "$TEST_ROOT/mismatch.out"
assert_contains "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_file_absent "$mismatch/$ZIP_NAME"
assert_file_absent "$mismatch/test-model"
if compgen -G "$mismatch/.stage.*" > /dev/null; then
  fail "checksum mismatch left a staging directory"
fi

echo "[12] --force re-downloads an up-to-date verified model"
forced="$TEST_ROOT/forced"
mkdir -p "$forced/test-model"
printf 'id = "test-model"\n' > "$forced/test-model/manifest.toml"
printf '%s\n' "$EXPECTED_MD5" > "$forced/test-model/.sparrow_zip_md5"
reset_log
run_downloader valid --force --dest "$forced" test-model > "$TEST_ROOT/forced.out" 2>&1
assert_contains "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_contains "[OK] MD5 verified" "$TEST_ROOT/forced.out"

echo "[13] --force still refuses transfer when checksums are unavailable"
force_offline="$TEST_ROOT/force-offline"
mkdir -p "$force_offline/test-model"
printf 'id = "test-model"\n' > "$force_offline/test-model/manifest.toml"
printf '%s\n' "$EXPECTED_MD5" > "$force_offline/test-model/.sparrow_zip_md5"
reset_log
expect_failure "$TEST_ROOT/force-offline.out" fail \
  --force --dest "$force_offline" test-model
assert_contains "refusing to download unverified bytes" "$TEST_ROOT/force-offline.out"
assert_absent "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
[[ -f "$force_offline/test-model/manifest.toml" ]] \
  || fail "force refusal damaged the existing install"

echo "[14] multi-model run installs verified models before refusing a later gap"
multi="$TEST_ROOT/multi"
reset_log
expect_failure "$TEST_ROOT/multi.out" valid \
  --dest "$multi" test-model test-model-two
[[ -f "$multi/test-model/manifest.toml" ]] \
  || fail "first verified model was not installed"
assert_file_absent "$multi/test-model-two"
assert_contains "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_absent "$FILE_URL_FRAGMENT_TWO" "$FAKE_CURL_LOG"
assert_contains "no MD5 available for $ZIP_NAME_TWO" "$TEST_ROOT/multi.out"

echo "[15] checksum-less API entries are treated as no usable checksums"
blank="$TEST_ROOT/blank"
reset_log
expect_failure "$TEST_ROOT/blank.out" blank --dest "$blank" test-model
assert_contains "record API returned no file checksums" "$TEST_ROOT/blank.out"
assert_contains "refusing to download unverified bytes" "$TEST_ROOT/blank.out"
assert_absent "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"

echo "[16] fully verified multi-model run installs and stamps every model"
multi_success="$TEST_ROOT/multi-success"
reset_log
run_downloader valid_both --dest "$multi_success" test-model test-model-two \
  > "$TEST_ROOT/multi-success.out" 2>&1
[[ -f "$multi_success/test-model/manifest.toml" ]] \
  || fail "first model missing after successful multi-model run"
[[ -f "$multi_success/test-model-two/manifest.toml" ]] \
  || fail "second model missing after successful multi-model run"
[[ "$(cat "$multi_success/test-model/.sparrow_zip_md5")" == "$EXPECTED_MD5" ]] \
  || fail "first model stamp mismatch after multi-model run"
[[ "$(cat "$multi_success/test-model-two/.sparrow_zip_md5")" == "$EXPECTED_MD5_TWO" ]] \
  || fail "second model stamp mismatch after multi-model run"
assert_contains "$FILE_URL_FRAGMENT" "$FAKE_CURL_LOG"
assert_contains "$FILE_URL_FRAGMENT_TWO" "$FAKE_CURL_LOG"
assert_contains "Downloaded 2 model(s)" "$TEST_ROOT/multi-success.out"

echo "download_models fail-closed tests: PASS"
