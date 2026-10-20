"""Offline tests for `sparrow_engine.zoo` (model-zoo download).

HTTP is replaced by an in-memory fake serving a small catalog, the record API
JSON, and per-model ZIPs, so these tests mirror the behaviors pinned for
`scripts/download_models.sh` in `scripts/tests/test_download_models.sh`.
"""
from __future__ import annotations

import hashlib
import io
import json
import tomllib
import zipfile
from pathlib import Path

import pytest
import sparrow_engine
from sparrow_engine import zoo

RECORD = "999"

CATALOG = f"""
[zenodo]
record = "{RECORD}"
version = "9.9.9"

[[model]]
id = "det-a"
domain = "camera_trap"
task = "detector"
format = "onnx"
license = "MIT"
status = "active"
hosting_status = "hosted"
alias = ["Old_Det_A"]
zip = "camera_trap__detector__det-a.zip"

[[model]]
id = "det-a-fp16"
domain = "camera_trap"
task = "detector"
format = "onnx"
flavor = "onnx-fp16"
license = "MIT"
status = "active"
hosting_status = "hosted"
zip = "camera_trap__detector__det-a-fp16.zip"

[[model]]
id = "cls-b"
domain = "camera_trap"
task = "classifier"
format = "tflite"
license = "MIT"
status = "active"
hosting_status = "hosted_restricted"
zip = "camera_trap__classifier__cls-b.zip"

[[model]]
id = "linked"
domain = "camera_trap"
task = "detector"
format = "onnx"
license = "AGPL-3.0"
status = "active"
hosting_status = "link_only"
original_source_url = "https://example.org/linked"
zip = "camera_trap__detector__linked.zip"

[[model]]
id = "cand"
domain = "camera_trap"
task = "detector"
format = "onnx"
license = "MIT"
status = "candidate"
hosting_status = "hosted"
zip = "camera_trap__detector__cand.zip"
""".encode()


def _model_zip(model_id: str, payload: bytes = b"weights") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{model_id}/manifest.toml", f'id = "{model_id}"\n')
        zf.writestr(f"{model_id}/model.onnx", payload)
    return buf.getvalue()


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324


class FakeZenodo:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {"catalog.toml": CATALOG}
        for mid in ("det-a", "det-a-fp16", "cls-b"):
            m = next(
                x for x in tomllib.loads(CATALOG.decode())["model"] if x["id"] == mid
            )
            self.files[m["zip"]] = _model_zip(mid)
        self.checksums = {k: _md5(v) for k, v in self.files.items()}
        self.requests: list[str] = []

    def urlopen(self, url: str):
        self.requests.append(url)
        if url == f"https://zenodo.org/api/records/{RECORD}":
            files = [{"key": k, "checksum": f"md5:{v}"} for k, v in self.checksums.items()]
            return io.BytesIO(json.dumps({"files": files}).encode())
        prefix = f"https://zenodo.org/records/{RECORD}/files/"
        if url.startswith(prefix) and url[len(prefix):] in self.files:
            return io.BytesIO(self.files[url[len(prefix):]])
        raise OSError(f"404 {url}")

    def downloads(self) -> list[str]:
        return [u.rsplit("/", 1)[1] for u in self.requests if u.endswith(".zip")]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeZenodo:
    z = FakeZenodo()
    monkeypatch.setattr(zoo, "_urlopen", z.urlopen)
    monkeypatch.delenv("SPARROW_CATALOG", raising=False)
    monkeypatch.delenv("ZENODO_RECORD", raising=False)
    monkeypatch.setattr(zoo, "ZENODO_RECORD", RECORD)
    return z


def _ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


def test_pinned_record_matches_repo_catalog() -> None:
    catalog = Path(__file__).resolve().parents[2] / "scripts" / "catalog.toml"
    if not catalog.is_file():
        pytest.skip("repo catalog not available (installed-wheel test run)")
    record = tomllib.loads(catalog.read_text())["zenodo"]["record"]
    assert zoo.ZENODO_RECORD == record, (
        "bump sparrow_engine.zoo.ZENODO_RECORD to match scripts/catalog.toml"
    )


def test_public_api_exported() -> None:
    for name in ("zoo_models", "download_models", "download_model", "ZooModel", "ZooError"):
        assert name in sparrow_engine.__all__
        assert hasattr(sparrow_engine, name)


def test_zoo_models_flags(fake: FakeZenodo) -> None:
    models = {m.id: m for m in sparrow_engine.zoo_models()}
    assert set(models) == {"det-a", "det-a-fp16", "cls-b", "linked", "cand"}
    assert [k for k, m in models.items() if m.default] == ["det-a"]
    assert {k for k, m in models.items() if m.downloadable} == {"det-a", "det-a-fp16", "cls-b"}
    assert models["linked"].source_url == "https://example.org/linked"


def test_default_selection_installs_and_stamps(fake: FakeZenodo, tmp_path: Path) -> None:
    out = sparrow_engine.download_models(dest=tmp_path, quiet=True)
    assert _ids(out) == ["det-a"]
    assert (tmp_path / "det-a" / "model.onnx").read_bytes() == b"weights"
    stamp = (tmp_path / "det-a" / ".sparrow_zip_md5").read_text().strip()
    assert stamp == fake.checksums["camera_trap__detector__det-a.zip"]
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".stage.")]


def test_all_models_selects_every_hosted_package(fake: FakeZenodo, tmp_path: Path) -> None:
    out = sparrow_engine.download_models(dest=tmp_path, all_models=True, quiet=True)
    assert sorted(_ids(out)) == ["cls-b", "det-a", "det-a-fp16"]


def test_alias_resolves_and_dedupes(fake: FakeZenodo, tmp_path: Path) -> None:
    out = sparrow_engine.download_models(["Old_Det_A", "det-a"], dest=tmp_path, quiet=True)
    assert _ids(out) == ["det-a"]
    assert fake.downloads() == ["camera_trap__detector__det-a.zip"]


@pytest.mark.parametrize(
    ("model", "message"),
    [("nope", "unknown model id"), ("linked", "link_only"), ("cand", "candidate")],
)
def test_refused_selections(fake: FakeZenodo, tmp_path: Path, model: str, message: str) -> None:
    with pytest.raises(sparrow_engine.ZooError, match=message):
        sparrow_engine.download_model(model, dest=tmp_path, force=True, verify=False, quiet=True)
    assert fake.downloads() == []


def test_md5_mismatch_leaves_nothing(fake: FakeZenodo, tmp_path: Path) -> None:
    fake.checksums["camera_trap__detector__det-a.zip"] = "0" * 32
    with pytest.raises(sparrow_engine.ZooError, match="MD5 mismatch"):
        sparrow_engine.download_model("det-a", dest=tmp_path, quiet=True)
    assert list(tmp_path.iterdir()) == []


def test_missing_checksum_refuses_unless_verify_false(fake: FakeZenodo, tmp_path: Path) -> None:
    del fake.checksums["camera_trap__detector__det-a.zip"]
    with pytest.raises(sparrow_engine.ZooError, match="no MD5 published"):
        sparrow_engine.download_model("det-a", dest=tmp_path, quiet=True)
    path = sparrow_engine.download_model("det-a", dest=tmp_path, verify=False, quiet=True)
    assert (path / "model.onnx").is_file()
    assert not (path / ".sparrow_zip_md5").exists()


def test_reinstall_rules(fake: FakeZenodo, tmp_path: Path) -> None:
    zip_name = "camera_trap__detector__det-a.zip"
    sparrow_engine.download_model("det-a", dest=tmp_path, quiet=True)
    sparrow_engine.download_model("det-a", dest=tmp_path, quiet=True)
    assert fake.downloads() == [zip_name], "current stamp must skip"

    fake.files[zip_name] = _model_zip("det-a", b"republished")
    fake.checksums[zip_name] = _md5(fake.files[zip_name])
    sparrow_engine.download_model("det-a", dest=tmp_path, quiet=True)
    assert (tmp_path / "det-a" / "model.onnx").read_bytes() == b"republished"
    assert len(fake.downloads()) == 2, "stale stamp must re-download"

    (tmp_path / "det-a" / ".sparrow_zip_md5").unlink()
    sparrow_engine.download_model("det-a", dest=tmp_path, quiet=True)
    assert len(fake.downloads()) == 2, "unstamped install is kept"
    sparrow_engine.download_model("det-a", dest=tmp_path, force=True, quiet=True)
    assert len(fake.downloads()) == 3, "force re-downloads"


def test_tampered_remote_catalog_rejected(fake: FakeZenodo, tmp_path: Path) -> None:
    fake.checksums["catalog.toml"] = "0" * 32
    with pytest.raises(sparrow_engine.ZooError, match="catalog.toml"):
        sparrow_engine.download_models(dest=tmp_path, quiet=True)


def test_local_catalog_and_env_dest(
    fake: FakeZenodo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cat = tmp_path / "catalog.toml"
    cat.write_bytes(CATALOG)
    models_dir = tmp_path / "models"
    monkeypatch.setenv("SPARROW_CATALOG", str(cat))
    monkeypatch.setenv("SPARROW_ENGINE_MODEL_DIR", str(models_dir))
    out = sparrow_engine.download_models(["cls-b"], quiet=True)
    assert out == [models_dir / "cls-b"]
    assert not any(u.endswith("catalog.toml") for u in fake.requests)


def test_zip_without_model_dir_rejected(fake: FakeZenodo, tmp_path: Path) -> None:
    zip_name = "camera_trap__detector__det-a.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("other/manifest.toml", "")
    fake.files[zip_name] = buf.getvalue()
    fake.checksums[zip_name] = _md5(fake.files[zip_name])
    with pytest.raises(sparrow_engine.ZooError, match="expected 'det-a/'"):
        sparrow_engine.download_model("det-a", dest=tmp_path, quiet=True)
    assert list(tmp_path.iterdir()) == []
