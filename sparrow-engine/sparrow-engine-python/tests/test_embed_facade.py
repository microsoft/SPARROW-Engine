from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import sparrow_engine


class BatchFakeEngine:
    """Batched double mimicking the native ``PyEngine`` embed / embed_aligned contract.

    Keyed by resolved path string; a mapped ``Exception`` marks that file as a
    failure. Both methods receive the FULL path list in a single call (batched,
    in caller order) and fire ``progress_callback`` once per path — in order,
    regardless of success or failure — matching the native engine.

    - ``embed`` is fail-closed: raises ``EmbedAllFailedError`` when every file
      fails, ``EmbedPartialFailureError`` on any partial failure, else returns
      ``list[result]`` in caller order.
    - ``embed_aligned`` returns one slot per path (``None`` for failures) and
      raises ``EmbedAllFailedError`` only when every file fails.
    """

    def __init__(self, by_path):
        self.by_path = by_path
        self.embed_calls = []
        self.embed_aligned_calls = []

    def _slots(self, paths, progress_callback):
        slots = []
        failures = 0
        total = len(paths)
        for i, path in enumerate(paths):
            value = self.by_path[path]
            if isinstance(value, Exception):
                slots.append(None)
                failures += 1
            else:
                slots.append(value)
            if progress_callback is not None:
                progress_callback(i, total, path)
        return slots, failures, total

    def embed(self, paths, model, progress_callback=None):
        self.embed_calls.append(list(paths))
        slots, failures, total = self._slots(paths, progress_callback)
        if failures and failures == total and total > 0:
            raise sparrow_engine.EmbedAllFailedError("All files failed processing.")
        if failures:
            raise sparrow_engine.EmbedPartialFailureError(
                f"{failures} of {total} files failed"
            )
        return slots

    def embed_aligned(self, paths, model, progress_callback=None):
        self.embed_aligned_calls.append(list(paths))
        slots, failures, total = self._slots(paths, progress_callback)
        if failures and failures == total and total > 0:
            raise sparrow_engine.EmbedAllFailedError("All files failed processing.")
        return slots


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


def test_aligned_embedding_errors_are_public() -> None:
    assert issubclass(
        sparrow_engine.EmbedPartialFailureError,
        sparrow_engine.SparrowEngineError,
    )
    assert issubclass(
        sparrow_engine.EmbedAllFailedError,
        sparrow_engine.SparrowEngineError,
    )
    assert hasattr(sparrow_engine.PyEngine, "embed_aligned")
    # The new facade aligned variants are public and exported.
    assert callable(sparrow_engine.embed_aligned)
    assert callable(sparrow_engine.embed_aligned_with_meta)
    assert "embed_aligned" in sparrow_engine.__all__
    assert "embed_aligned_with_meta" in sparrow_engine.__all__


# --- Strict (fail-closed, order-preserving) contract -------------------------


def test_embed_single_returns_owned_writable_vector(monkeypatch):
    case_dir = _case_dir("single")
    image = case_dir / "a.jpg"
    image.write_bytes(b"not-real-image")
    engine = BatchFakeEngine({str(image): fake_result([1.0, 2.0, 3.0])})
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    arr = sparrow_engine.embed(image, "encoder")

    assert arr.shape == (3,)
    assert arr.dtype == np.float32
    assert arr.flags.owndata
    assert arr.flags.writeable
    # The facade now issues ONE batched native call (not one call per file).
    assert engine.embed_calls == [[str(image)]]


def test_embed_batch_returns_matrix_in_caller_order(monkeypatch):
    case_dir = _case_dir("batch")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = BatchFakeEngine({
        str(image_a): fake_result([1.0, 0.0]),
        str(image_b): fake_result([0.0, 1.0]),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    # Caller order is [b, a]; result rows MUST follow caller order, not sorted.
    arr = sparrow_engine.embed([image_b, image_a], "encoder")

    assert arr.shape == (2, 2)
    assert arr.dtype == np.float32
    assert arr.flags.owndata
    assert arr.flags.writeable
    assert arr.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    # One batched call, paths in caller order.
    assert engine.embed_calls == [[str(image_b), str(image_a)]]


def test_embed_empty_batch_returns_empty_matrix(monkeypatch):
    engine = BatchFakeEngine({})
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    arr = sparrow_engine.embed([], "encoder")

    assert arr.shape == (0, 0)
    assert arr.dtype == np.float32
    assert arr.flags.owndata
    assert arr.flags.writeable
    assert engine.embed_calls == []


@pytest.mark.parametrize("shape", ["empty_list", "empty_dir", "no_match_glob_in_list"])
def test_embed_empty_inputs_do_not_invoke_engine(monkeypatch, shape):
    # Empty resolutions must be a no-op: return an empty result WITHOUT invoking
    # the native engine (which loads the model before its chunk loop, so an empty
    # request would otherwise raise when the model is unavailable). The recorded
    # call list is the only signal that proves the engine was never touched.
    engine = BatchFakeEngine({})
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)
    case_dir = _case_dir(f"empty-strict-{shape}")
    if shape == "empty_list":
        input = []
    elif shape == "empty_dir":
        input = str(case_dir)
    else:  # no_match_glob_in_list
        input = [str(case_dir / "*.jpg")]

    arr = sparrow_engine.embed(input, "encoder")
    meta = sparrow_engine.embed_with_meta(input, "encoder")

    assert arr.shape == (0, 0)
    assert arr.dtype == np.float32
    assert meta == []
    assert engine.embed_calls == []


@pytest.mark.parametrize("shape", ["empty_list", "empty_dir", "no_match_glob_in_list"])
def test_embed_aligned_empty_inputs_do_not_invoke_engine(monkeypatch, shape):
    # Same no-op contract for the aligned variants: an empty positional list and
    # NO native call.
    engine = BatchFakeEngine({})
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)
    case_dir = _case_dir(f"empty-aligned-{shape}")
    if shape == "empty_list":
        input = []
    elif shape == "empty_dir":
        input = str(case_dir)
    else:  # no_match_glob_in_list
        input = [str(case_dir / "*.jpg")]

    bare = sparrow_engine.embed_aligned(input, "encoder")
    meta = sparrow_engine.embed_aligned_with_meta(input, "encoder")

    assert bare == []
    assert meta == []
    assert engine.embed_aligned_calls == []


def test_embed_single_no_match_glob_raises_without_engine(monkeypatch):
    # A SINGLE-STRING no-match glob is fail-closed: it raises BEFORE the engine is
    # touched (guarding that the REV-004 short-circuit did not weaken this path).
    engine = BatchFakeEngine({})
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)
    case_dir = _case_dir("single-no-match-glob")

    with pytest.raises(sparrow_engine.SparrowEngineError):
        sparrow_engine.embed_with_meta(str(case_dir / "*.jpg"), "encoder")

    assert engine.embed_calls == []


def test_embed_with_meta_preserves_identity(monkeypatch):
    case_dir = _case_dir("meta")
    image = case_dir / "a.jpg"
    image.write_bytes(b"a")
    engine = BatchFakeEngine({str(image): fake_result([1.0, 2.0], model_id="encoder")})
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    out = sparrow_engine.embed_with_meta(image, "encoder")

    assert out.model_id == "encoder"
    assert out.embedding_version == "v1"
    assert out.model_hash == "abc123"
    assert out.embed_schema_version == "1.0"


def test_embed_with_meta_single_missing_input_raises(monkeypatch):
    engine = BatchFakeEngine({})
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    with pytest.raises(sparrow_engine.SparrowEngineError):
        sparrow_engine.embed_with_meta("missing.jpg", "encoder")


def test_embed_directory_string_returns_matrix(monkeypatch):
    case_dir = _case_dir("directory")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = BatchFakeEngine({
        str(image_a): fake_result([1.0, 0.0]),
        str(image_b): fake_result([0.0, 1.0]),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    arr = sparrow_engine.embed(str(case_dir), "encoder")
    meta = sparrow_engine.embed_with_meta(str(case_dir), "encoder")

    assert arr.shape == (2, 2)
    assert isinstance(meta, list)
    assert len(meta) == 2
    # A directory expansion is deterministically sorted within the expansion.
    assert engine.embed_calls == [
        [str(image_a), str(image_b)],
        [str(image_a), str(image_b)],
    ]


def test_embed_with_meta_raises_on_partial_failure_and_reports_progress(monkeypatch):
    case_dir = _case_dir("partial-fail")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = BatchFakeEngine({
        str(image_a): fake_result([1.0, 0.0]),
        str(image_b): sparrow_engine.SparrowEngineError("bad image"),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)
    progress = []

    # A partial failure now RAISES (fail-closed) rather than compacting to len==1.
    with pytest.raises(sparrow_engine.EmbedPartialFailureError):
        sparrow_engine.embed_with_meta(
            [image_b, image_a],
            "encoder",
            progress_callback=lambda i, total, path: progress.append((i, total, path)),
        )

    # Caller order [b, a]; progress fires once per file, even on the failed one.
    assert engine.embed_calls == [[str(image_b), str(image_a)]]
    assert progress == [
        (0, 2, str(image_b)),
        (1, 2, str(image_a)),
    ]


def test_embed_with_meta_raises_all_failed(monkeypatch):
    case_dir = _case_dir("all-fail")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = BatchFakeEngine({
        str(image_a): sparrow_engine.SparrowEngineError("bad a"),
        str(image_b): sparrow_engine.SparrowEngineError("bad b"),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    with pytest.raises(sparrow_engine.EmbedAllFailedError, match="All files failed"):
        sparrow_engine.embed_with_meta([image_b, image_a], "encoder")


# --- Aligned facade variants (one Optional slot per requested input) ----------


def test_embed_aligned_with_meta_preserves_positions_and_none_slots(monkeypatch):
    case_dir = _case_dir("aligned-meta")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = BatchFakeEngine({
        str(image_a): sparrow_engine.SparrowEngineError("bad a"),
        str(image_b): fake_result([0.0, 1.0]),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)
    progress = []

    out = sparrow_engine.embed_aligned_with_meta(
        [image_b, image_a],
        "encoder",
        progress_callback=lambda i, total, path: progress.append((i, total, path)),
    )

    # One slot per input, caller order [b, a]; the failed input -> None.
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0] is not None and out[0].model_id == "encoder"
    assert out[1] is None
    assert engine.embed_aligned_calls == [[str(image_b), str(image_a)]]
    assert progress == [
        (0, 2, str(image_b)),
        (1, 2, str(image_a)),
    ]


def test_embed_aligned_returns_ndarray_slots_with_none(monkeypatch):
    case_dir = _case_dir("aligned-bare")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = BatchFakeEngine({
        str(image_a): fake_result([1.0, 0.0]),
        str(image_b): sparrow_engine.SparrowEngineError("bad b"),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    out = sparrow_engine.embed_aligned([image_a, image_b], "encoder")

    assert isinstance(out, list)
    assert len(out) == 2
    assert isinstance(out[0], np.ndarray)
    assert out[0].tolist() == [1.0, 0.0]
    assert out[0].dtype == np.float32
    assert out[0].flags.owndata
    assert out[0].flags.writeable
    assert out[1] is None


def test_embed_aligned_single_input_returns_one_element_list(monkeypatch):
    case_dir = _case_dir("aligned-single")
    image = case_dir / "a.jpg"
    image.write_bytes(b"a")
    engine = BatchFakeEngine({str(image): fake_result([1.0, 2.0])})
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    out = sparrow_engine.embed_aligned_with_meta(image, "encoder")

    # Aligned variants never unwrap a single input -> always a positional list.
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].model_id == "encoder"


def test_embed_aligned_all_failed_raises(monkeypatch):
    case_dir = _case_dir("aligned-all-fail")
    image_a = case_dir / "a.jpg"
    image_b = case_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    engine = BatchFakeEngine({
        str(image_a): sparrow_engine.SparrowEngineError("bad a"),
        str(image_b): sparrow_engine.SparrowEngineError("bad b"),
    })
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    with pytest.raises(sparrow_engine.EmbedAllFailedError, match="All files failed"):
        sparrow_engine.embed_aligned([image_b, image_a], "encoder")
