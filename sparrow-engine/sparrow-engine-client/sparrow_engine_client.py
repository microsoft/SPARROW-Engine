"""sparrow-engine-client: Python SDK for sparrow-engine-server HTTP API.

Install: pip install httpx  (only runtime dependency)
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx


# ---------------------------------------------------------------------------
# Dataclasses — field names match sparrow-engine-server JSON exactly
# ---------------------------------------------------------------------------


@dataclass
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized [0,1] bbox to pixel coordinates."""
        return (
            round(self.x_min * width),
            round(self.y_min * height),
            round(self.x_max * width),
            round(self.y_max * height),
        )


@dataclass
class Detection:
    label: str
    label_id: int
    confidence: float
    bbox: BBox


@dataclass
class Classification:
    label: str
    label_id: int
    confidence: float


@dataclass
class PipelineDetection:
    detection: Detection
    classification: Optional[Classification]
    crop: Optional["PipelineCropRegion"] = None
    failure: Optional["PipelineFailure"] = None


@dataclass
class PipelineCropRegion:
    bbox: BBox
    width_px: int
    height_px: int
    coordinate_source: str


@dataclass
class PipelineFailure:
    stage: str
    code: str
    message: str
    model_id: Optional[str] = None


@dataclass
class PipelineStageProvenance:
    model_id: str
    model_version: Optional[str] = None
    model_hash: Optional[str] = None


@dataclass
class PipelineProvenance:
    detector: PipelineStageProvenance
    classifier: Optional[PipelineStageProvenance] = None


@dataclass
class DetectResult:
    model_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    detections: list[Detection]


@dataclass
class ClassifyResult:
    model_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    classifications: list[Classification]


@dataclass
class PipelineResult:
    pipeline_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    detections: list[PipelineDetection]
    stage_provenance: Optional[PipelineProvenance] = None


@dataclass
class AudioClass:
    """One class score inside a multiclass audio segment.

    ``label`` is optional: the server omits it when the manifest provides no
    label for ``class_idx``.
    """

    class_idx: int
    probability: float
    label: Optional[str] = None


@dataclass
class AudioSegment:
    start_time_s: float
    end_time_s: float
    confidence: float
    # Populated only for multiclass audio models (server emits `classes` when
    # a segment carries more than one class). None for binary/single-class
    # models, preserving the original binary AudioSegment shape.
    classes: Optional[list[AudioClass]] = None


@dataclass
class AudioResult:
    model_id: str
    duration_s: float
    sample_rate: int
    processing_time_ms: float
    segments: list[AudioSegment]


@dataclass
class ModelInfo:
    id: str
    model_type: str
    default: bool = False
    version: Optional[str] = None
    description: Optional[str] = None
    onnx_sha256: Optional[str] = None
    onnx_size_bytes: Optional[int] = None
    embedding_version: Optional[str] = None
    embedding_dim: Optional[int] = None
    normalized: Optional[bool] = None
    metric: Optional[str] = None


@dataclass
class EmbedResult:
    """Single-image embedding (POST /v1/embed)."""

    model_id: str
    embedding_version: str
    model_hash: str
    embedding_dim: int
    normalized: bool
    metric: str
    image_size: tuple[int, int]
    processing_time_ms: float
    embedding: list[float]
    embed_schema_version: str


@dataclass
class EmbedBatchItem:
    index: int
    image_size: tuple[int, int]
    processing_time_ms: float
    embedding: list[float]


@dataclass
class EmbedBatchResult:
    """Batch embedding (POST /v1/embed/batch)."""

    model_id: str
    embedding_version: str
    model_hash: str
    embedding_dim: int
    normalized: bool
    metric: str
    count: int
    processing_time_ms: float
    results: list[EmbedBatchItem]
    embed_schema_version: str


@dataclass
class CatalogEntry:
    """One entry from GET /v1/catalog (a model or a named pipeline).

    The core fields are typed; open-ended model-zoo metadata (display_name,
    family, geo_scope, provenance, …) is preserved verbatim in ``raw`` so the
    client stays forward-compatible as the server adds catalog fields.
    """

    model_id: str
    model_type: str
    framework: str
    loaded: bool
    trt_state: str
    trt_detail: Optional[str] = None
    embedding_dim: Optional[int] = None
    embedding_version: Optional[str] = None
    normalized: Optional[bool] = None
    metric: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class PipelineStep:
    role: str
    model_id: str


@dataclass
class PipelineInfo:
    """A named pipeline alias and its steps (GET /v1/pipelines, create/load)."""

    id: str
    steps: list[PipelineStep]


@dataclass
class TrtWarmupResult:
    """Result of POST /v1/models/{id}/trt-warmup.

    ``started`` is True when the server accepted a new warm-up build (HTTP 202)
    and False when the model was already TensorRT-ready (HTTP 200). ``poll``
    carries the server's suggested follow-up probe on a 202, else None.
    """

    trt_state: str
    started: bool
    poll: Optional[dict] = None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class SparrowEngineClientError(Exception):
    """Error returned by sparrow-engine-server."""

    def __init__(self, code: str, message: str, status: int) -> None:
        self.code = code
        self.message = message
        self.status = status
        super().__init__(f"[{status}] {code}: {message}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


class SparrowEngineClient:
    """Synchronous Python client for sparrow-engine-server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 60.0,
        management_token: Optional[str] = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._management_token = management_token

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SparrowEngineClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # --- Private helpers ---

    def _auth_kwargs(self) -> dict[str, Any]:
        """Request kwargs carrying the management bearer token, if configured.

        Returns ``{"headers": {"Authorization": "Bearer <token>"}}`` when a
        management token is set, else ``{}``. Only the management endpoints
        (models*, pipelines*) pass this; inference, catalog, and health stay
        open. Returning an empty dict when no token is configured keeps the
        underlying httpx call identical to the pre-auth client for the common
        no-token case.
        """
        if self._management_token:
            return {"headers": {"Authorization": f"Bearer {self._management_token}"}}
        return {}

    def _check(self, resp: httpx.Response) -> dict:
        """Parse response; raise SparrowEngineClientError on 4xx/5xx."""
        self._raise_for_error(resp)
        return resp.json()

    def _check_list(self, resp: httpx.Response) -> list:
        """Like ``_check`` but for endpoints whose success body is a JSON array
        (e.g. GET /v1/catalog). Error bodies remain the ``{"error": {...}}``
        object contract, so the same error handling applies."""
        self._raise_for_error(resp)
        return resp.json()

    @staticmethod
    def _raise_for_error(resp: httpx.Response) -> None:
        """Raise SparrowEngineClientError on 4xx/5xx, falling back to
        httpx.HTTPStatusError when the body is not the JSON error contract."""
        if resp.status_code >= 400:
            try:
                err = resp.json()["error"]
                raise SparrowEngineClientError(
                    err["code"], err["message"], resp.status_code
                )
            except (KeyError, ValueError, TypeError):
                resp.raise_for_status()

    @staticmethod
    def _add_store_params(
        params: dict[str, Any], store: bool, halt_on_store_failure: bool
    ) -> None:
        if store:
            params["store"] = True
        if halt_on_store_failure:
            params["halt_on_store_failure"] = True

    @staticmethod
    def _parse_detection(det: dict) -> Detection:
        """Parse a detection JSON dict into a Detection dataclass."""
        return Detection(
            label=det["label"],
            label_id=det["label_id"],
            confidence=det["confidence"],
            bbox=BBox(**det["bbox"]),
        )

    @staticmethod
    def _parse_audio_segment(seg: dict) -> AudioSegment:
        """Parse an audio segment, tolerating the optional multiclass `classes`.

        Binary / single-class models omit `classes` entirely (parsed as None,
        preserving the original binary shape). Multiclass models emit a
        `classes` array; each entry's `label` is optional.
        """
        raw_classes = seg.get("classes")
        classes = (
            [
                AudioClass(
                    class_idx=c["class_idx"],
                    probability=c["probability"],
                    label=c.get("label"),
                )
                for c in raw_classes
            ]
            if raw_classes is not None
            else None
        )
        return AudioSegment(
            start_time_s=seg["start_time_s"],
            end_time_s=seg["end_time_s"],
            confidence=seg["confidence"],
            classes=classes,
        )

    @staticmethod
    def _image_file(image: Any) -> tuple[str, bytes, str]:
        """Normalize image input to (filename, bytes, mimetype)."""
        if isinstance(image, (str, Path)):
            p = Path(image)
            mime = _MIME_MAP.get(p.suffix.lower(), "application/octet-stream")
            return (p.name, p.read_bytes(), mime)
        if isinstance(image, bytes):
            return ("image.jpg", image, "image/jpeg")
        if hasattr(image, "read"):
            return ("image.jpg", image.read(), "image/jpeg")
        # Assume PIL Image
        if hasattr(image, "mode") and image.mode == "RGBA":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        return ("image.jpg", buf.getvalue(), "image/jpeg")

    # --- Inference ---

    def detect(
        self,
        image: Any,
        model: str,
        threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> DetectResult:
        """Run single-image detection."""
        params: dict[str, Any] = {"model": model}
        if threshold is not None:
            params["threshold"] = threshold
        if max_detections is not None:
            params["max_detections"] = max_detections
        self._add_store_params(params, store, halt_on_store_failure)
        fname, data, mime = self._image_file(image)
        resp = self._client.post(
            "/v1/detect",
            params=params,
            files={"image": (fname, data, mime)},
        )
        d = self._check(resp)
        return DetectResult(
            model_id=d["model_id"],
            image_size=tuple(d["image_size"]),
            processing_time_ms=d["processing_time_ms"],
            detections=[self._parse_detection(det) for det in d["detections"]],
        )

    def classify(
        self,
        image: Any,
        model: str,
        top_k: int = 5,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> ClassifyResult:
        """Run single-image classification."""
        params: dict[str, Any] = {"model": model, "top_k": top_k}
        self._add_store_params(params, store, halt_on_store_failure)
        fname, data, mime = self._image_file(image)
        resp = self._client.post(
            "/v1/classify",
            params=params,
            files={"image": (fname, data, mime)},
        )
        d = self._check(resp)
        return ClassifyResult(
            model_id=d["model_id"],
            image_size=tuple(d["image_size"]),
            processing_time_ms=d["processing_time_ms"],
            classifications=[Classification(**c) for c in d["classifications"]],
        )

    def pipeline(
        self,
        image: Any,
        pipeline: Optional[str] = None,
        threshold: Optional[float] = None,
        top_k: int = 5,
        max_detections: Optional[int] = None,
        store: bool = False,
        halt_on_store_failure: bool = False,
        *,
        detector: Optional[str] = None,
        classifier: Optional[str] = None,
    ) -> PipelineResult:
        """Run a detect+classify pipeline.

        Two mutually exclusive selection shapes, matching the server:

        - Named alias:  ``pipeline("img.jpg", pipeline="my-pipeline")``
        - Ad-hoc pair:  ``pipeline("img.jpg", detector="MDV6-yolov10-e",
          classifier="SpeciesNet-Crop")``

        Exactly one shape must be given; supplying both, neither, or only one
        of ``detector``/``classifier`` raises ``ValueError`` before any request
        is sent.
        """
        named = pipeline is not None
        adhoc = detector is not None or classifier is not None
        if named and adhoc:
            raise ValueError(
                "specify either `pipeline` OR `detector`+`classifier`, not both"
            )
        if not named and not adhoc:
            raise ValueError(
                "one of `pipeline` or `detector`+`classifier` is required"
            )
        if adhoc and not (detector is not None and classifier is not None):
            raise ValueError(
                "ad-hoc pipeline requires both `detector` and `classifier`"
            )

        params: dict[str, Any] = {}
        if named:
            params["pipeline"] = pipeline
        else:
            params["detector"] = detector
            params["classifier"] = classifier
        params["top_k"] = top_k
        if threshold is not None:
            params["threshold"] = threshold
        if max_detections is not None:
            params["max_detections"] = max_detections
        self._add_store_params(params, store, halt_on_store_failure)
        fname, data, mime = self._image_file(image)
        resp = self._client.post(
            "/v1/pipeline",
            params=params,
            files={"image": (fname, data, mime)},
        )
        d = self._check(resp)
        stage_data = d.get("stage_provenance")
        stage_provenance = None
        if stage_data:
            stage_provenance = PipelineProvenance(
                detector=PipelineStageProvenance(**stage_data["detector"]),
                classifier=(
                    PipelineStageProvenance(**stage_data["classifier"])
                    if stage_data.get("classifier")
                    else None
                ),
            )
        return PipelineResult(
            pipeline_id=d["pipeline_id"],
            image_size=tuple(d["image_size"]),
            processing_time_ms=d["processing_time_ms"],
            detections=[
                PipelineDetection(
                    detection=self._parse_detection(det),
                    classification=(
                        Classification(**det["classification"])
                        if det.get("classification")
                        else None
                    ),
                    crop=(
                        PipelineCropRegion(
                            bbox=BBox(**det["crop"]["bbox"]),
                            width_px=det["crop"]["width_px"],
                            height_px=det["crop"]["height_px"],
                            coordinate_source=det["crop"]["coordinate_source"],
                        )
                        if det.get("crop")
                        else None
                    ),
                    failure=(
                        PipelineFailure(**det["failure"])
                        if det.get("failure")
                        else None
                    ),
                )
                for det in d["detections"]
            ],
            stage_provenance=stage_provenance,
        )

    def detect_audio(
        self,
        audio: Any,
        model: str,
        threshold: Optional[float] = None,
        segment_duration: Optional[float] = None,
        stride: Optional[float] = None,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> AudioResult:
        """Run audio detection."""
        params: dict[str, Any] = {"model": model}
        if threshold is not None:
            params["threshold"] = threshold
        if segment_duration is not None:
            params["segment_duration"] = segment_duration
        if stride is not None:
            params["stride"] = stride
        self._add_store_params(params, store, halt_on_store_failure)
        if isinstance(audio, (str, Path)):
            p = Path(audio)
            audio_data = p.read_bytes()
            fname = p.name
        elif isinstance(audio, bytes):
            audio_data = audio
            fname = "audio.wav"
        else:
            audio_data = audio.read()
            fname = "audio.wav"
        resp = self._client.post(
            "/v1/audio/detect",
            params=params,
            files={"audio": (fname, audio_data, "audio/wav")},
        )
        d = self._check(resp)
        return AudioResult(
            model_id=d["model_id"],
            duration_s=d["duration_s"],
            sample_rate=d["sample_rate"],
            processing_time_ms=d["processing_time_ms"],
            segments=[self._parse_audio_segment(s) for s in d["segments"]],
        )

    def detect_batch(
        self,
        images: list[Any],
        model: str,
        threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> list[DetectResult]:
        """Run batch detection on multiple images."""
        params: dict[str, Any] = {"model": model}
        if threshold is not None:
            params["threshold"] = threshold
        if max_detections is not None:
            params["max_detections"] = max_detections
        self._add_store_params(params, store, halt_on_store_failure)
        files = [
            ("images", self._image_file(img)) for img in images
        ]
        resp = self._client.post("/v1/detect/batch", params=params, files=files)
        d = self._check(resp)
        model_id = d["model_id"]
        processing_time_ms = d["processing_time_ms"]
        return [
            DetectResult(
                model_id=model_id,
                image_size=tuple(item["image_size"]),
                processing_time_ms=processing_time_ms,
                detections=[
                    self._parse_detection(det) for det in item["detections"]
                ],
            )
            for item in d["results"]
        ]

    # --- Embeddings ---

    def embed(
        self,
        image: Any,
        model: str,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> EmbedResult:
        """Compute a single-image embedding (POST /v1/embed)."""
        params: dict[str, Any] = {"model": model}
        self._add_store_params(params, store, halt_on_store_failure)
        fname, data, mime = self._image_file(image)
        resp = self._client.post(
            "/v1/embed",
            params=params,
            files={"image": (fname, data, mime)},
        )
        d = self._check(resp)
        return EmbedResult(
            model_id=d["model_id"],
            embedding_version=d["embedding_version"],
            model_hash=d["model_hash"],
            embedding_dim=d["embedding_dim"],
            normalized=d["normalized"],
            metric=d["metric"],
            image_size=tuple(d["image_size"]),
            processing_time_ms=d["processing_time_ms"],
            embedding=d["embedding"],
            embed_schema_version=d["embed_schema_version"],
        )

    def embed_batch(
        self,
        images: list[Any],
        model: str,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> EmbedBatchResult:
        """Compute embeddings for multiple images (POST /v1/embed/batch)."""
        params: dict[str, Any] = {"model": model}
        self._add_store_params(params, store, halt_on_store_failure)
        files = [("images", self._image_file(img)) for img in images]
        resp = self._client.post("/v1/embed/batch", params=params, files=files)
        d = self._check(resp)
        return EmbedBatchResult(
            model_id=d["model_id"],
            embedding_version=d["embedding_version"],
            model_hash=d["model_hash"],
            embedding_dim=d["embedding_dim"],
            normalized=d["normalized"],
            metric=d["metric"],
            count=d["count"],
            processing_time_ms=d["processing_time_ms"],
            results=[
                EmbedBatchItem(
                    index=item["index"],
                    image_size=tuple(item["image_size"]),
                    processing_time_ms=item["processing_time_ms"],
                    embedding=item["embedding"],
                )
                for item in d["results"]
            ],
            embed_schema_version=d["embed_schema_version"],
        )

    # --- Catalog (open, no auth) ---

    def catalog(self) -> list[CatalogEntry]:
        """List discovered models and named pipelines (GET /v1/catalog).

        Read-only and unauthenticated. Returns a typed core view; open-ended
        model-zoo metadata is preserved in each entry's ``raw`` dict.
        """
        resp = self._client.get("/v1/catalog")
        entries = self._check_list(resp)
        return [
            CatalogEntry(
                model_id=e["model_id"],
                model_type=e["model_type"],
                framework=e["framework"],
                loaded=e["loaded"],
                trt_state=e["trt_state"],
                trt_detail=e.get("trt_detail"),
                embedding_dim=e.get("embedding_dim"),
                embedding_version=e.get("embedding_version"),
                normalized=e.get("normalized"),
                metric=e.get("metric"),
                raw=e,
            )
            for e in entries
        ]

    # --- Model management (bearer-token protected) ---

    def list_models(self) -> list[ModelInfo]:
        resp = self._client.get("/v1/models", **self._auth_kwargs())
        d = self._check(resp)
        return [ModelInfo(**m) for m in d["models"]]

    def load_model(self, model_id: str) -> ModelInfo:
        resp = self._client.post(
            "/v1/models/load", json={"model_id": model_id}, **self._auth_kwargs()
        )
        d = self._check(resp)
        return ModelInfo(**d)

    def unload_model(self, model_id: str) -> None:
        resp = self._client.delete(
            f"/v1/models/{quote(model_id, safe='')}", **self._auth_kwargs()
        )
        if resp.status_code == 204:
            return
        self._check(resp)

    def trt_warmup(self, model_id: str) -> TrtWarmupResult:
        """Kick an explicit TensorRT warm-up (POST /v1/models/{id}/trt-warmup).

        HTTP 202 → a build started (``started=True``); HTTP 200 → the model was
        already TensorRT-ready (``started=False``). A CPU build returns 422
        (``TRT_UNSUPPORTED_HARDWARE``) and an unknown model 404, both surfaced
        as ``SparrowEngineClientError``.
        """
        resp = self._client.post(
            f"/v1/models/{quote(model_id, safe='')}/trt-warmup",
            **self._auth_kwargs(),
        )
        d = self._check(resp)
        return TrtWarmupResult(
            trt_state=d["trt_state"],
            started=resp.status_code == 202,
            poll=d.get("poll"),
        )

    # --- Pipeline management (bearer-token protected) ---

    def list_pipelines(self) -> list[PipelineInfo]:
        """List loaded named pipelines (GET /v1/pipelines)."""
        resp = self._client.get("/v1/pipelines", **self._auth_kwargs())
        d = self._check(resp)
        return [self._parse_pipeline_info(p) for p in d["pipelines"]]

    def create_pipeline(
        self,
        pipeline_id: str,
        detector: str,
        classifier: str,
        replace: bool = False,
        persist: bool = False,
    ) -> PipelineInfo:
        """Create (or update) a named detector+classifier pipeline alias.

        POST /v1/pipelines. The server returns HTTP 201 for a newly created
        alias and 200 when an identical alias already existed; both parse to a
        ``PipelineInfo``. Pass ``replace=True`` to overwrite an alias that
        exists with a different definition (otherwise the server returns 409),
        and ``persist=True`` to write the alias to disk.
        """
        body: dict[str, Any] = {
            "id": pipeline_id,
            "detector": detector,
            "classifier": classifier,
            "replace": replace,
            "persist": persist,
        }
        resp = self._client.post(
            "/v1/pipelines", json=body, **self._auth_kwargs()
        )
        d = self._check(resp)
        return self._parse_pipeline_info(d)

    def load_pipeline(self, pipeline_id: str) -> PipelineInfo:
        """Load a persisted pipeline alias by ID (POST /v1/pipelines/load)."""
        resp = self._client.post(
            "/v1/pipelines/load",
            json={"pipeline_id": pipeline_id},
            **self._auth_kwargs(),
        )
        d = self._check(resp)
        return self._parse_pipeline_info(d)

    def delete_pipeline(self, pipeline_id: str) -> None:
        """Delete a pipeline alias (DELETE /v1/pipelines/{id}).

        Returns None on success (HTTP 204); a missing alias raises
        ``SparrowEngineClientError`` (HTTP 404).
        """
        resp = self._client.delete(
            f"/v1/pipelines/{quote(pipeline_id, safe='')}", **self._auth_kwargs()
        )
        if resp.status_code == 204:
            return
        self._check(resp)

    @staticmethod
    def _parse_pipeline_info(p: dict) -> PipelineInfo:
        return PipelineInfo(
            id=p["id"],
            steps=[
                PipelineStep(role=s["role"], model_id=s["model_id"])
                for s in p["steps"]
            ],
        )

    # --- Health (open, no auth) ---

    def health(self) -> dict:
        resp = self._client.get("/v1/health")
        return self._check(resp)

    def healthz(self) -> dict:
        """Liveness probe (GET /healthz). Returns ``{"alive": true}``."""
        resp = self._client.get("/healthz")
        return self._check(resp)

    def is_ready(self) -> bool:
        try:
            h = self.health()
            return h["status"] in ("ready", "no_models")
        except Exception:
            return False

    def wait_ready(self, timeout: float = 60.0, interval: float = 1.0) -> None:
        """Block until server is ready or timeout.

        Each health probe uses a short per-request timeout capped to
        ``min(interval, remaining)`` so the overall wall-clock time
        stays bounded by *timeout* even when the server is unreachable.
        """
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                req_timeout = min(interval, remaining)
                resp = self._client.get("/v1/health", timeout=req_timeout)
                if resp.status_code < 400:
                    data = resp.json()
                    if data["status"] in ("ready", "no_models"):
                        return
            except httpx.TransportError:
                pass
            remaining = deadline - time.time()
            if remaining > interval:
                time.sleep(interval)
        raise TimeoutError(f"sparrow-engine-server not ready after {timeout}s")
