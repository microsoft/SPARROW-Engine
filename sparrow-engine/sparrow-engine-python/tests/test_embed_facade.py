from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import sparrow_engine


class FakeEngine:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def embed(self, paths, model, progress_callback=None):
        self.calls.append((paths, model))
        if progress_callback is not None:
            for i, path in enumerate(paths):
                progress_callback(i, len(paths), path)
        return self.results


class PerFileFakeEngine:
    def __init__(self, by_path):
        self.by_path = by_path
        self.calls = []

    def embed(self, paths, model, progress_callback=None):
        self.calls.append((paths, model))
        assert len(paths) == 1
        value = self.by_path[paths[0]]
        if isinstance(value, Exception):
            raise value
        return [value]


def fake_result(values, *, model_id="encoder"):
    return SimpleNamespace(
        vector=np.array(values, dtype=np.float32),
        dim=len(values),
        normalized=True,
        metric="cosine",
        model_id=model_id,
        embedding_version="v1",
        model_hash="abc123",
        embed_schema_version="1.0",
        image_width=10,
        image_height=20,
        processing_time_ms=1.5,
    )


def _case_dir(name: str) -> Path:
    path = Path.cwd() / "target" / "pytest-embed-facade" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_embed_single_returns_owned_writable_vector(monkeypatch):
    case_dir = _case_dir("single")
    image = case_dir / "a.jpg"
    image.write_bytes(b"not-real-image")
    engine = FakeEngine([fake_result([1.0, 2.0, 3.0])])
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    arr = sparrow_engine.embed(image, "encoder")

    assert arr.shape == (3,)
    assert arr.dtype == np.float32
    assert arr.flags.owndata
    assert arr.flags.writeable
    assert engine.calls == [([str(image)], "encoder")]


def test_embed_batch_returns_matrix(monkeypatch):
    case_dir = _case_dir("batch")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = PerFileFakeEngine({
        str(image_a): fake_result([1.0, 0.0]),
        str(image_b): fake_result([0.0, 1.0]),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    arr = sparrow_engine.embed([image_b, image_a], "encoder")

    assert arr.shape == (2, 2)
    assert arr.dtype == np.float32
    assert arr.flags.owndata
    assert arr.flags.writeable
    assert [call[0][0] for call in engine.calls] == sorted(
        [str(image_a), str(image_b)]
    )


def test_embed_empty_batch_returns_empty_matrix(monkeypatch):
    engine = FakeEngine([])
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    arr = sparrow_engine.embed([], "encoder")

    assert arr.shape == (0, 0)
    assert arr.dtype == np.float32
    assert arr.flags.owndata
    assert arr.flags.writeable


def test_embed_with_meta_preserves_identity(monkeypatch):
    case_dir = _case_dir("meta")
    image = case_dir / "a.jpg"
    image.write_bytes(b"a")
    result = fake_result([1.0, 2.0], model_id="encoder")
    engine = FakeEngine([result])
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    out = sparrow_engine.embed_with_meta(image, "encoder")

    assert out.model_id == "encoder"
    assert out.embedding_version == "v1"
    assert out.model_hash == "abc123"
    assert out.embed_schema_version == "1.0"


def test_embed_with_meta_single_missing_input_raises(monkeypatch):
    engine = FakeEngine([])
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    with pytest.raises(sparrow_engine.SparrowEngineError):
        sparrow_engine.embed_with_meta("missing.jpg", "encoder")


def test_embed_directory_string_returns_matrix(monkeypatch):
    case_dir = _case_dir("directory")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = PerFileFakeEngine({
        str(image_a): fake_result([1.0, 0.0]),
        str(image_b): fake_result([0.0, 1.0]),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    arr = sparrow_engine.embed(str(case_dir), "encoder")
    meta = sparrow_engine.embed_with_meta(str(case_dir), "encoder")

    assert arr.shape == (2, 2)
    assert isinstance(meta, list)
    assert len(meta) == 2


def test_embed_with_meta_skips_bad_files_and_reports_progress(monkeypatch):
    case_dir = _case_dir("skip")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = PerFileFakeEngine({
        str(image_a): fake_result([1.0, 0.0]),
        str(image_b): sparrow_engine.SparrowEngineError("bad image"),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)
    progress = []

    out = sparrow_engine.embed_with_meta(
        [image_b, image_a],
        "encoder",
        progress_callback=lambda i, total, path: progress.append((i, total, path)),
    )

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].vector.tolist() == [1.0, 0.0]
    assert [call[0][0] for call in engine.calls] == sorted(
        [str(image_a), str(image_b)]
    )
    assert progress == [
        (0, 2, str(image_a)),
        (1, 2, str(image_b)),
    ]


def test_embed_with_meta_raises_only_when_all_files_fail(monkeypatch):
    case_dir = _case_dir("all-fail")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = PerFileFakeEngine({
        str(image_a): sparrow_engine.SparrowEngineError("bad a"),
        str(image_b): sparrow_engine.SparrowEngineError("bad b"),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    with pytest.raises(sparrow_engine.SparrowEngineError, match="All files failed"):
        sparrow_engine.embed_with_meta([image_b, image_a], "encoder")
