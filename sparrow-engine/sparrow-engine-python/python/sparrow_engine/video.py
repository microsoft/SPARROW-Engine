"""Video-file and live-camera inference on top of :func:`sparrow_engine.detect`.

Frames are decoded with OpenCV (optional extra: ``pip install 'sparrow-engine[video]'``),
written as JPEGs to a private temporary directory in batches, and run through the
same ``detect`` / ``pipeline`` calls used for still images, so results, thresholds
and model defaults are identical to image inference.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    import numpy as np

    from sparrow_engine._sparrow_engine_core import DetectResult, PipelineResult

__all__ = [
    "FrameResult",
    "detect_video",
    "iter_video",
    "detect_webcam",
    "draw_detections",
]

_INSTALL_HINT = (
    "Video and webcam inference need OpenCV. Install it with "
    "`pip install 'sparrow-engine[video]'` (or any package that provides `cv2`)."
)
_JPEG_QUALITY = 95


@dataclass(frozen=True)
class FrameResult:
    """Inference result for one sampled frame.

    ``index`` is the 0-based frame number in the source stream. ``timestamp_s`` is
    the position in the video (file input) or seconds since capture started
    (camera input). ``result.path`` is ``"<source>#frame=<index>"`` (for example
    ``"clip.mp4#frame=42"`` or ``"camera:0#frame=42"``); the temporary JPEG used
    for inference is already deleted. ``frame`` holds the decoded BGR image only
    when ``keep_frames=True``.
    """

    index: int
    timestamp_s: float
    result: Union["DetectResult", "PipelineResult"]
    frame: Optional["np.ndarray"] = None


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without OpenCV
        raise ImportError(_INSTALL_HINT) from exc
    return cv2


def _frame_step(every_n: Optional[int], fps: Optional[float], source_fps: float) -> int:
    if every_n is not None and fps is not None:
        raise ValueError("pass at most one of every_n and fps")
    if every_n is not None:
        if every_n < 1:
            raise ValueError("every_n must be >= 1")
        return every_n
    if fps is not None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if source_fps <= 0:
            return 1
        return max(1, round(source_fps / fps))
    return 1


def _source_label(source: Union[int, str]) -> str:
    """Label for camera results: ``camera:N`` for a device, else the URL without credentials.

    Stream URLs such as ``rtsp://user:password@host/stream`` often carry a password, and the
    label ends up in every result's ``path``, its repr and exported files.
    """
    if isinstance(source, int):
        return f"camera:{source}"
    parts = urlsplit(source)
    if parts.username is None and parts.password is None:
        return source
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit(parts._replace(netloc=host))


class _Runner:
    """Batch frames to temporary JPEGs and run one detect/pipeline call per batch."""

    def __init__(
        self,
        source: str,
        model: Optional[str],
        classifier: Optional[str],
        pipeline_id: Optional[str],
        threshold: Optional[float],
        max_detections: Optional[int],
        top_k: int,
    ) -> None:
        if pipeline_id is not None and classifier is not None:
            raise ValueError("pass either pipeline_id or classifier, not both")
        if pipeline_id is not None and model is not None:
            raise ValueError(
                "pass either pipeline_id or model: a named pipeline already sets its detector"
            )
        self.use_pipeline = classifier is not None or pipeline_id is not None
        if self.use_pipeline and max_detections is not None:
            raise ValueError("max_detections applies to detection only, not to pipelines")
        if self.use_pipeline and pipeline_id is None and model is None:
            import sparrow_engine as spe

            # Resolved once here, not per batch: the lookup lists the model catalog.
            model = spe._resolve_default_model(
                None, spe._DETECTION_MODEL_TYPES, spe._DEFAULT_DETECT_MODEL
            )
        self.source = source
        self.model = model
        self.classifier = classifier
        self.pipeline_id = pipeline_id
        self.threshold = threshold
        self.max_detections = max_detections
        self.top_k = top_k
        self._tmp = tempfile.TemporaryDirectory(prefix="sparrow-video-")
        self._dir = Path(self._tmp.name)

    def close(self) -> None:
        self._tmp.cleanup()

    def run(
        self, batch: list[tuple[int, float, "np.ndarray"]], keep_frames: bool
    ) -> list[FrameResult]:
        import sparrow_engine as spe

        cv2 = _cv2()
        paths = []
        for index, _, frame in batch:
            # Zero-padded names keep detect()'s sorted path order equal to frame order.
            path = self._dir / f"{index:012d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]):
                raise RuntimeError(f"could not encode frame {index}")
            paths.append(str(path))
        try:
            if self.use_pipeline:
                results = spe.pipeline(
                    paths,
                    detector=self.model,
                    classifier=self.classifier,
                    threshold=self.threshold,
                    top_k=self.top_k,
                    pipeline_id=self.pipeline_id,
                )
            else:
                results = spe.detect(
                    paths,
                    model=self.model,
                    threshold=self.threshold,
                    max_detections=self.max_detections,
                )
        finally:
            for p in paths:
                Path(p).unlink(missing_ok=True)
        if len(results) != len(batch):
            raise RuntimeError(
                f"engine returned {len(results)} results for {len(batch)} frames"
            )
        return [
            FrameResult(
                index,
                ts,
                res.with_path(f"{self.source}#frame={index}"),
                frame if keep_frames else None,
            )
            for (index, ts, frame), res in zip(batch, results, strict=True)
        ]


def iter_video(
    path: Union[str, Path],
    model: Optional[str] = None,
    *,
    classifier: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    top_k: int = 5,
    every_n: Optional[int] = None,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    batch_size: int = 16,
    keep_frames: bool = False,
) -> Iterator[FrameResult]:
    """Stream :class:`FrameResult` objects for a video file, one batch at a time.

    Sampling: ``every_n`` keeps every n-th frame; ``fps`` keeps about that many frames
    per second of video; neither keeps every frame. ``max_frames`` caps the number of
    frames run through the model. Without ``classifier`` / ``pipeline_id`` each frame
    goes through :func:`sparrow_engine.detect`; with either, through
    :func:`sparrow_engine.pipeline`. With ``classifier``, ``model`` is the detector
    (default: the catalog default detector). With ``pipeline_id``, leave ``model``
    unset; the named pipeline sets its own detector. ``max_detections`` applies to
    detection only. Invalid arguments raise ``ValueError`` before the video is opened.
    """
    cv2 = _cv2()
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    _frame_step(every_n, fps, 0.0)  # validate sampling arguments before opening anything
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    runner = _Runner(str(path), model, classifier, pipeline_id, threshold, max_detections, top_k)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        step = _frame_step(every_n, fps, source_fps)
        batch: list[tuple[int, float, Any]] = []
        index = taken = 0
        while max_frames is None or taken < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                ts = index / source_fps if source_fps > 0 else cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3
                batch.append((index, float(ts), frame))
                taken += 1
                if len(batch) == batch_size:
                    yield from runner.run(batch, keep_frames)
                    batch = []
            index += 1
        if batch:
            yield from runner.run(batch, keep_frames)
    finally:
        cap.release()
        runner.close()


def detect_video(
    path: Union[str, Path], model: Optional[str] = None, **kwargs: Any
) -> list[FrameResult]:
    """Run detection (or a pipeline) on a video file and return all frame results.

    Takes the same keyword arguments as :func:`iter_video`. Use ``iter_video`` for
    long videos when results should be processed as they arrive.
    """
    return list(iter_video(path, model, **kwargs))


def detect_webcam(
    source: Union[int, str] = 0,
    model: Optional[str] = None,
    *,
    classifier: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    top_k: int = 5,
    every_n: int = 1,
    max_frames: Optional[int] = None,
    keep_frames: bool = True,
) -> Iterator[FrameResult]:
    """Yield a :class:`FrameResult` per sampled frame from a live camera.

    ``source`` is an OpenCV capture source: a device index (``0`` = default webcam)
    or a stream URL such as ``rtsp://...``. Frames are processed one at a time for
    the lowest latency. The generator stops after ``max_frames`` inferred frames or
    when the stream ends; stop it early with ``break`` (the camera is released).
    ``keep_frames`` defaults to ``True`` so callers can display annotated frames
    with :func:`draw_detections`.
    """
    cv2 = _cv2()
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    label = _source_label(source)
    runner = _Runner(label, model, classifier, pipeline_id, threshold, max_detections, top_k)
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open camera source: {label!r}")
        start = time.monotonic()
        index = taken = 0
        while max_frames is None or taken < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % every_n == 0:
                ts = time.monotonic() - start
                yield from runner.run([(index, ts, frame)], keep_frames)
                taken += 1
            index += 1
    finally:
        cap.release()
        runner.close()


def draw_detections(
    frame: "np.ndarray",
    result: Union["DetectResult", "PipelineResult"],
    *,
    show_labels: bool = True,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> "np.ndarray":
    """Return a copy of a BGR ``frame`` with the result's boxes drawn on it.

    For pipeline results the label is the classification when present, else the
    detection label.
    """
    cv2 = _cv2()
    out = frame.copy()
    height, width = out.shape[:2]
    for det in result.detections:
        inner = getattr(det, "detection", det)
        label, conf = inner.label, inner.confidence
        cls = getattr(det, "classification", None)
        if cls is not None:
            label, conf = cls.label, cls.confidence
        x1, y1, x2, y2 = inner.bbox.to_pixels(width, height)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if show_labels:
            cv2.putText(
                out,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return out
