"""Per-file results carry ``path`` and list inputs keep the caller's order.

The native batch functions skip files that fail to decode, so the result list
can be shorter than the input list; ``zip(paths, results)`` then pairs the
wrong image with a result. Each result names its file in ``.path`` instead.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import sparrow_engine


class RecordingEngine:
    def __init__(self):
        self.paths = None

    def detect(self, paths, model, threshold=None, max_detections=None, progress_callback=None):
        self.paths = list(paths)
        return []


def test_list_input_keeps_caller_order(monkeypatch, tmp_path):
    names = ["c.jpg", "a.jpg", "b.jpg"]
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    engine = RecordingEngine()
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    sparrow_engine.detect([str(tmp_path / n) for n in names], model="m")

    assert [Path(p).name for p in engine.paths] == names


def test_directory_input_is_sorted(monkeypatch, tmp_path):
    for n in ["c.jpg", "a.jpg", "b.jpg"]:
        (tmp_path / n).write_bytes(b"x")
    engine = RecordingEngine()
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)

    sparrow_engine.detect(str(tmp_path), model="m")

    assert [Path(p).name for p in engine.paths] == ["a.jpg", "b.jpg", "c.jpg"]


def test_pairs_accepts_results_and_tuples():
    r = SimpleNamespace(path="/x/a.jpg")
    assert sparrow_engine._pairs([r]) == [("/x/a.jpg", r)]
    assert sparrow_engine._pairs([(Path("/x/b.jpg"), r)]) == [("/x/b.jpg", r)]


MODEL = "MDV6-yolov10-c"


def _model_dir() -> Path:
    return Path(sparrow_engine._default_model_dir())


@pytest.fixture
def images(tmp_path):
    if not (_model_dir() / MODEL / "manifest.toml").exists():
        pytest.skip(f"{MODEL} not in {_model_dir()}")
    np = pytest.importorskip("numpy")
    pil = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    paths = []
    for name in ["z.jpg", "m.jpg", "a.jpg"]:
        p = tmp_path / name
        pil.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    # bad file in the middle: z, bad, m, a
    return [paths[0], bad, paths[1], paths[2]]


def test_detect_results_carry_paths_and_skip_failures(images, tmp_path):
    results = sparrow_engine.detect([str(p) for p in images], model=MODEL)

    good = [str(p) for p in images if p.name != "bad.jpg"]
    assert [r.path for r in results] == good

    out = tmp_path / "out.json"
    sparrow_engine.export(results, "megadet", str(out), model_id=MODEL)
    files = [img["file"] for img in json.loads(out.read_text())["images"]]
    assert [os.path.basename(f) for f in files] == ["z.jpg", "m.jpg", "a.jpg"]
