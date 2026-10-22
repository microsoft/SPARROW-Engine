"""Tests for sparrow_engine.video. The engine is replaced by fakes; no model is needed."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import sparrow_engine as spe  # noqa: E402
from sparrow_engine import video  # noqa: E402

N_FRAMES = 20
FPS = 10.0


def _shade(i: int) -> int:
    return i * 10


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    """A 20-frame, 10 fps clip; frame i is a uniform grey of value 10*i."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    assert writer.isOpened()
    for i in range(N_FRAMES):
        writer.write(np.full((48, 64, 3), _shade(i), dtype=np.uint8))
    writer.release()
    return path


class FakeEngine:
    """Records calls and returns one result per path carrying the frame's grey value."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict]] = []

    def _results(self, paths):
        out = []
        for p in paths:
            img = cv2.imread(p)
            out.append(SimpleNamespace(shade=float(img.mean()), detections=[]))
        return out

    def detect(self, paths, **kw):
        self.calls.append(("detect", list(paths), kw))
        return self._results(paths)

    def pipeline(self, paths, **kw):
        self.calls.append(("pipeline", list(paths), kw))
        return self._results(paths)


@pytest.fixture()
def fake(monkeypatch) -> FakeEngine:
    eng = FakeEngine()
    monkeypatch.setattr(spe, "detect", eng.detect)
    monkeypatch.setattr(spe, "pipeline", eng.pipeline)
    monkeypatch.setattr(spe, "_resolve_default_model", lambda *a: "DEFAULT-DET")
    return eng


def _assert_aligned(frames):
    for fr in frames:
        assert abs(fr.result.shade - _shade(fr.index)) < 4, fr


def test_exports():
    for name in ("FrameResult", "detect_video", "iter_video", "detect_webcam", "draw_detections"):
        assert name in spe.__all__
        assert getattr(spe, name) is getattr(video, name)


def test_every_frame_in_order_and_batched(clip, fake):
    frames = spe.detect_video(clip, "M", batch_size=8, threshold=0.3)
    assert [f.index for f in frames] == list(range(N_FRAMES))
    assert frames[3].timestamp_s == pytest.approx(0.3)
    assert all(f.frame is None for f in frames)
    _assert_aligned(frames)
    assert [len(c[1]) for c in fake.calls] == [8, 8, 4]
    assert fake.calls[0][2] == {"model": "M", "threshold": 0.3, "max_detections": None}


def test_temp_frames_removed(clip, fake):
    spe.detect_video(clip, batch_size=5)
    dirs = {Path(p).parent for _, paths, _ in fake.calls for p in paths}
    assert len(dirs) == 1
    assert not next(iter(dirs)).exists()


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"every_n": 5}, [0, 5, 10, 15]),
        ({"fps": 2.0}, [0, 5, 10, 15]),
        ({"every_n": 3, "max_frames": 3}, [0, 3, 6]),
        ({"fps": 100.0}, list(range(N_FRAMES))),
    ],
)
def test_sampling(clip, fake, kwargs, expected):
    frames = spe.detect_video(clip, **kwargs)
    assert [f.index for f in frames] == expected
    _assert_aligned(frames)


def test_keep_frames(clip, fake):
    frames = spe.detect_video(clip, every_n=10, keep_frames=True)
    assert frames[1].frame.shape == (48, 64, 3)


def test_iter_video_streams_by_batch(clip, fake):
    it = spe.iter_video(clip, batch_size=4)
    first = next(it)
    assert first.index == 0 and len(fake.calls) == 1
    it.close()


def test_classifier_routes_to_pipeline_with_default_detector(clip, fake):
    spe.detect_video(clip, classifier="C", every_n=10, top_k=3)
    kind, _, kw = fake.calls[0]
    assert kind == "pipeline"
    assert kw["detector"] == "DEFAULT-DET" and kw["classifier"] == "C" and kw["top_k"] == 3


def test_pipeline_id_keeps_detector_unset(clip, fake):
    spe.detect_video(clip, pipeline_id="P", every_n=10)
    kw = fake.calls[0][2]
    assert kw["pipeline_id"] == "P" and kw["detector"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"every_n": 2, "fps": 1.0},
        {"every_n": 0},
        {"fps": 0},
        {"batch_size": 0},
        {"max_frames": 0},
        {"classifier": "C", "pipeline_id": "P"},
        {"model": "D", "pipeline_id": "P"},
        {"classifier": "C", "max_detections": 5},
        {"pipeline_id": "P", "max_detections": 5},
    ],
)
def test_bad_arguments(clip, fake, kwargs):
    with pytest.raises(ValueError):
        spe.detect_video(clip, **kwargs)


class _SpyCapture:
    opened: list["_SpyCapture"] = []

    def __init__(self, *a):
        self.released = False
        _SpyCapture.opened.append(self)

    def isOpened(self):
        return False

    def release(self):
        self.released = True


@pytest.mark.parametrize("kwargs", [{"every_n": 0}, {"model": "D", "pipeline_id": "P"}])
def test_bad_arguments_open_no_capture(clip, fake, monkeypatch, kwargs):
    _SpyCapture.opened = []
    monkeypatch.setattr(cv2, "VideoCapture", _SpyCapture)
    with pytest.raises(ValueError):
        spe.detect_video(clip, **kwargs)
    with pytest.raises(ValueError):
        next(spe.detect_webcam(0, **kwargs))
    assert _SpyCapture.opened == []


def test_unopenable_capture_is_released(clip, fake, monkeypatch):
    _SpyCapture.opened = []
    monkeypatch.setattr(cv2, "VideoCapture", _SpyCapture)
    with pytest.raises(RuntimeError):
        spe.detect_video(clip)
    with pytest.raises(RuntimeError):
        next(spe.detect_webcam(0))
    assert len(_SpyCapture.opened) == 2 and all(c.released for c in _SpyCapture.opened)


def test_default_detector_resolved_once(clip, fake, monkeypatch):
    lookups = []
    monkeypatch.setattr(
        spe, "_resolve_default_model", lambda *a: lookups.append(a) or "DEFAULT-DET"
    )
    frames = list(spe.detect_webcam(str(clip), classifier="C", max_frames=5))
    assert len(frames) == 5 and len(fake.calls) == 5
    assert len(lookups) == 1


def test_missing_file(tmp_path, fake):
    with pytest.raises(FileNotFoundError):
        spe.detect_video(tmp_path / "nope.mp4")


def test_result_count_mismatch_raises(clip, monkeypatch):
    monkeypatch.setattr(spe, "detect", lambda paths, **kw: [])
    with pytest.raises(RuntimeError, match="0 results for"):
        spe.detect_video(clip)


def test_webcam_with_stream_source(clip, fake):
    # Any OpenCV capture source works; a file stands in for a camera here.
    frames = list(spe.detect_webcam(str(clip), every_n=4, max_frames=3))
    assert [f.index for f in frames] == [0, 4, 8]
    assert all(f.frame is not None for f in frames)
    assert all(len(c[1]) == 1 for c in fake.calls)
    ts = [f.timestamp_s for f in frames]
    assert ts == sorted(ts) and ts[0] >= 0
    _assert_aligned(frames)


def test_webcam_unopenable_source(fake):
    with pytest.raises(RuntimeError, match="could not open camera"):
        next(spe.detect_webcam("/nonexistent/stream.avi"))


def _box(x0, y0, x1, y1):
    def to_pixels(w, h):
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

    return SimpleNamespace(to_pixels=to_pixels)


def test_draw_detections_detect_and_pipeline():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = SimpleNamespace(label="animal", confidence=0.9, bbox=_box(0.1, 0.1, 0.5, 0.5))
    out = spe.draw_detections(frame, SimpleNamespace(detections=[det]), show_labels=False)
    assert frame.sum() == 0  # input untouched
    assert tuple(out[10, 30]) == (0, 255, 0)  # top edge of the box
    assert out[30, 30].sum() == 0  # interior empty

    pdet = SimpleNamespace(
        detection=det, classification=SimpleNamespace(label="deer", confidence=0.8)
    )
    out2 = spe.draw_detections(frame, SimpleNamespace(detections=[pdet]), color=(0, 0, 255))
    assert tuple(out2[50, 30]) == (0, 0, 255)  # bottom edge of the box
