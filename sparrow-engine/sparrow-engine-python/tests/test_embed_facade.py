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
    engine = FakeEngine([fake_result([1.0, 0.0]), fake_result([0.0, 1.0])])
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    arr = sparrow_engine.embed([image_b, image_a], "encoder")

    assert arr.shape == (2, 2)
    assert arr.dtype == np.float32
    assert arr.flags.owndata
    assert arr.flags.writeable
    assert engine.calls[0][0] == sorted([str(image_a), str(image_b)])


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
    engine = FakeEngine([fake_result([1.0, 0.0]), fake_result([0.0, 1.0])])
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    arr = sparrow_engine.embed(str(case_dir), "encoder")
    meta = sparrow_engine.embed_with_meta(str(case_dir), "encoder")

    assert arr.shape == (2, 2)
    assert isinstance(meta, list)
    assert len(meta) == 2
