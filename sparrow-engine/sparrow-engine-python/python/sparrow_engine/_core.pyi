"""Type stubs for sparrow_engine._sparrow_engine_core native module."""
from typing import Callable, Literal, Optional, TypedDict

import numpy as np

# Per-file progress callback: (index_0_based, total, filename) -> None.
# Invoked after each file's inference attempt resolves. See `detect` etc.
_ProgressCallback = Callable[[int, int, str], None]

class SparrowEngineError(Exception): ...
class TrtUnsupportedHardware(SparrowEngineError): ...
class EmbedPartialFailureError(SparrowEngineError): ...
class EmbedAllFailedError(SparrowEngineError): ...

class TrtStateInfo(TypedDict):
    state: Literal[
        "not_loaded",
        "cuda_ready",
        "trt_warming",
        "trt_ready",
        "trt_error",
        "unsupported",
        "unknown",
    ]
    detail: Optional[str]

class TrtWarmupOutcome(TypedDict):
    outcome: Literal["started", "already_ready"]

class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]: ...

class Detection:
    label: str
    label_id: int
    confidence: float
    bbox: BBox

class DetectResult:
    model_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    detections: list[Detection]
    def __len__(self) -> int: ...

class Classification:
    label: str
    label_id: int
    confidence: float

class ClassifyResult:
    model_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    classifications: list[Classification]
    # First classification (highest-confidence) if any; None on empty result.
    # Convenience for the common `result.top1.label` idiom.
    top1: Optional[Classification]
    def __len__(self) -> int: ...

class EmbedResult:
    vector: np.ndarray
    dim: int
    normalized: bool
    metric: str
    model_id: str
    embedding_version: str
    model_hash: str
    embed_schema_version: str
    image_width: int
    image_height: int
    processing_time_ms: float
    def __len__(self) -> int: ...

class PipelineCropRegion:
    bbox: BBox
    width_px: int
    height_px: int
    coordinate_source: str

class PipelineFailure:
    stage: str
    code: str
    model_id: Optional[str]
    message: str

class PipelineStageProvenance:
    model_id: str
    model_version: Optional[str]
    model_hash: Optional[str]

class PipelineProvenance:
    detector: PipelineStageProvenance
    classifier: Optional[PipelineStageProvenance]

class PipelineDetection:
    detection: Detection
    classification: Optional[Classification]
    crop: Optional[PipelineCropRegion]
    failure: Optional[PipelineFailure]

class PipelineResult:
    pipeline_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    detections: list[PipelineDetection]
    stage_provenance: PipelineProvenance
    def __len__(self) -> int: ...

class AudioClass:
    class_idx: int
    label: Optional[str]
    probability: float

class AudioSegment:
    start_time_s: float
    end_time_s: float
    confidence: float
    classes: list[AudioClass]

class AudioResult:
    model_id: str
    duration_s: float
    sample_rate: int
    window_s: float
    stride_s: float
    processing_time_ms: float
    segments: list[AudioSegment]
    def __len__(self) -> int: ...

class AudioEvent:
    start_time_s: float
    end_time_s: float
    low_freq_hz: float
    high_freq_hz: float
    peak_time_s: float
    peak_freq_hz: float
    confidence: float
    classes: list[AudioClass]

class AudioEventResult:
    model_id: str
    duration_s: float
    analyzed_duration_s: float
    sample_rate: int
    clip_duration_s: float
    clip_stride_s: float
    processing_time_ms: float
    events: list[AudioEvent]
    def __len__(self) -> int: ...

class ModelInfo:
    id: str
    model_type: str
    # Manifest [model].subtype: "standard" for normal detectors, "overhead"
    # for top-down / drone-imagery detectors (HerdNet, OWL). Derived from
    # model_type when the native ModelInfo only carries the broader type.
    subtype: str
    default: bool
    version: Optional[str]
    description: Optional[str]
    onnx_sha256: Optional[str]
    onnx_size_bytes: Optional[int]
    embedding_version: Optional[str]
    embedding_dim: Optional[int]
    normalized: Optional[bool]
    metric: Optional[str]

class PyEngine:
    def __init__(self, device: str, model_dir: str) -> None: ...
    def load_model(
        self,
        id: str,
        trt_warmup: bool = False,
    ) -> None: ...
    def trt_warmup(
        self,
        id: str,
        wait: bool = True,
    ) -> TrtStateInfo | TrtWarmupOutcome: ...
    def trt_state(self, id: str) -> TrtStateInfo: ...
    def detect(
        self,
        paths: list[str],
        model: str,
        threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[DetectResult]: ...
    def classify(
        self,
        paths: list[str],
        model: str,
        top_k: Optional[int] = None,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[ClassifyResult]: ...
    def embed(
        self,
        paths: list[str],
        model: str,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[EmbedResult]: ...
    def embed_aligned(
        self,
        paths: list[str],
        model: str,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[Optional[EmbedResult]]: ...
    def detect_audio(
        self,
        paths: list[str],
        model: str,
        threshold: Optional[float] = None,
        stride_s: Optional[float] = None,
        segment_duration_s: Optional[float] = None,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[AudioResult]: ...
    def detect_audio_events(
        self,
        paths: list[str],
        model: str,
        detection_threshold: Optional[float] = None,
        classification_threshold: Optional[float] = None,
        max_events: Optional[int] = None,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[AudioEventResult]: ...
    def pipeline(
        self,
        paths: list[str],
        detector: Optional[str] = None,
        classifier: Optional[str] = None,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        progress_callback: Optional[_ProgressCallback] = None,
        pipeline_id: Optional[str] = None,
    ) -> list[PipelineResult]: ...
    def list_models(self) -> list[ModelInfo]: ...
    def model_info(self, model_id: str) -> ModelInfo: ...
    def active_device(self) -> str: ...
    def hash_file(self, path: str) -> str: ...
    def day_night(self, path: str) -> dict: ...
    def verify_model(self, model_id: str) -> dict: ...


def hash_file(path: str) -> str: ...


def day_night(path: str) -> dict: ...


def verify_model(model_dir: str, model_id: str) -> dict: ...


def summarize(results: list[DetectResult]) -> dict: ...


def visualize(
    items: list[tuple[str, DetectResult | ClassifyResult | PipelineResult]],
    output_dir: Optional[str] = None,
    show_labels: bool = False,
) -> list[bytes]: ...


def visualize_audio(
    engine: PyEngine,
    items: list[tuple[str, AudioResult]],
    output_dir: Optional[str] = None,
    smooth: bool = False,
    show_windows: bool = False,
    show_ranges: bool = True,
) -> list[list[bytes]]: ...


def export_results(
    items: list[tuple[str, DetectResult | PipelineResult]],
    format: str,
    output: Optional[str] = None,
    model_id: Optional[str] = None,
) -> str: ...
