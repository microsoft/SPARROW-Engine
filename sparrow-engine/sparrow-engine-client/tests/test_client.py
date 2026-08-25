"""Tests for sparrow_engine_client using pytest-httpserver to mock sparrow-engine-server."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import httpx
import pytest

# Add parent dir so sparrow_engine_client is importable without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sparrow_engine_client import (
    AudioClass,
    AudioResult,
    AudioSegment,
    BBox,
    CatalogEntry,
    EmbedBatchItem,
    EmbedBatchResult,
    EmbedResult,
    PipelineInfo,
    PipelineStep,
    SparrowEngineClient,
    SparrowEngineClientError,
    ClassifyResult,
    DetectResult,
    ModelInfo,
    PipelineResult,
    TrtWarmupResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(httpserver):
    """SparrowEngineClient pointed at the pytest-httpserver."""
    with SparrowEngineClient(base_url=httpserver.url_for(""), timeout=5.0) as c:
        yield c


# ---------------------------------------------------------------------------
# Sample response payloads (match sparrow-engine-server JSON exactly)
# ---------------------------------------------------------------------------

DETECT_RESPONSE = {
    "model_id": "megadetector_v6",
    "image_size": [1920, 1080],
    "processing_time_ms": 42.5,
    "detections": [
        {
            "label": "animal",
            "label_id": 1,
            "confidence": 0.95,
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6},
        }
    ],
}

CLASSIFY_RESPONSE = {
    "model_id": "speciesnet",
    "image_size": [224, 224],
    "processing_time_ms": 15.3,
    "classifications": [
        {"label": "deer", "label_id": 3, "confidence": 0.88},
        {"label": "elk", "label_id": 7, "confidence": 0.05},
    ],
}

AUDIO_RESPONSE = {
    "model_id": "md_audiobirds_v1",
    "duration_s": 10.0,
    "sample_rate": 48000,
    "processing_time_ms": 120.0,
    "segments": [
        {"start_time_s": 1.5, "end_time_s": 3.2, "confidence": 0.92},
    ],
}

BATCH_DETECT_RESPONSE = {
    "model_id": "megadetector_v6",
    "count": 2,
    "processing_time_ms": 85.0,
    "results": [
        {
            "index": 0,
            "image_size": [1920, 1080],
            "detections": [
                {
                    "label": "animal",
                    "label_id": 1,
                    "confidence": 0.93,
                    "bbox": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.6, "y_max": 0.7},
                }
            ],
        },
        {
            "index": 1,
            "image_size": [640, 480],
            "detections": [],
        },
    ],
}

PIPELINE_RESPONSE = {
    "pipeline_id": "md_speciesnet",
    "model_id": None,
    "image_size": [1920, 1080],
    "processing_time_ms": 65.0,
    "detections": [
        {
            "label": "animal",
            "label_id": 1,
            "confidence": 0.95,
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6},
            "classification": {"label": "deer", "label_id": 3, "confidence": 0.88},
        },
        {
            "label": "animal",
            "label_id": 1,
            "confidence": 0.70,
            "bbox": {"x_min": 0.6, "y_min": 0.1, "x_max": 0.9, "y_max": 0.4},
            "classification": None,
        },
    ],
}

MODELS_RESPONSE = {
    "models": [
        {
            "id": "megadetector_v6",
            "model_type": "detector",
            "default": True,
            "version": "1.0.0",
            "description": "MegaDetector v6 general wildlife detector",
            "onnx_sha256": "abcdef0123456789",
            "onnx_size_bytes": 123456789,
        },
        {"id": "speciesnet", "model_type": "classifier", "default": False},
    ]
}

LOAD_MODEL_RESPONSE = {
    "id": "megadetector_v6",
    "model_type": "detector",
    "default": True,
    "version": "1.0.0",
    "description": "MegaDetector v6 general wildlife detector",
    "onnx_sha256": "abcdef0123456789",
    "onnx_size_bytes": 123456789,
}

HEALTH_RESPONSE = {
    "status": "ready",
    "models_loaded": 2,
    "pipelines_loaded": 1,
    "version": "0.1.0",
}

ERROR_RESPONSE = {
    "error": {
        "code": "MODEL_NOT_LOADED",
        "message": "Model 'foo' is not loaded.",
        "status": 499,
    }
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect(httpserver, client):
    httpserver.expect_request(
        "/v1/detect",
        method="POST",
        query_string="model=megadetector_v6&threshold=0.4&max_detections=10&store=true&halt_on_store_failure=true",
    ).respond_with_json(DETECT_RESPONSE)
    result = client.detect(
        image=b"\xff\xd8fake-jpeg",
        model="megadetector_v6",
        threshold=0.4,
        max_detections=10,
        store=True,
        halt_on_store_failure=True,
    )
    assert isinstance(result, DetectResult)
    assert result.model_id == "megadetector_v6"
    assert result.image_size == (1920, 1080)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.label == "animal"
    assert det.label_id == 1
    assert det.confidence == pytest.approx(0.95)
    assert det.bbox.x_min == pytest.approx(0.1)
    assert det.bbox.y_max == pytest.approx(0.6)


def test_classify(httpserver, client):
    httpserver.expect_request(
        "/v1/classify",
        method="POST",
        query_string="model=speciesnet&top_k=3&store=true&halt_on_store_failure=true",
    ).respond_with_json(CLASSIFY_RESPONSE)
    result = client.classify(
        image=b"\xff\xd8fake-jpeg",
        model="speciesnet",
        top_k=3,
        store=True,
        halt_on_store_failure=True,
    )
    assert isinstance(result, ClassifyResult)
    assert result.model_id == "speciesnet"
    assert len(result.classifications) == 2
    assert result.classifications[0].label == "deer"
    assert result.classifications[0].confidence == pytest.approx(0.88)


def test_detect_audio(httpserver, client):
    httpserver.expect_request(
        "/v1/audio/detect",
        method="POST",
        query_string="model=md_audiobirds_v1&threshold=0.7&segment_duration=1.5&stride=0.5&store=true&halt_on_store_failure=true",
    ).respond_with_json(AUDIO_RESPONSE)
    result = client.detect_audio(
        audio=b"fake-wav-data",
        model="md_audiobirds_v1",
        threshold=0.7,
        segment_duration=1.5,
        stride=0.5,
        store=True,
        halt_on_store_failure=True,
    )
    assert isinstance(result, AudioResult)
    assert result.model_id == "md_audiobirds_v1"
    assert result.duration_s == pytest.approx(10.0)
    assert result.sample_rate == 48000
    assert len(result.segments) == 1
    assert result.segments[0].start_time_s == pytest.approx(1.5)


def test_detect_batch(httpserver, client):
    httpserver.expect_request(
        "/v1/detect/batch",
        method="POST",
        query_string="model=megadetector_v6&threshold=0.5&max_detections=2&store=true&halt_on_store_failure=true",
    ).respond_with_json(BATCH_DETECT_RESPONSE)
    results = client.detect_batch(
        images=[b"\xff\xd8img1", b"\xff\xd8img2"],
        model="megadetector_v6",
        threshold=0.5,
        max_detections=2,
        store=True,
        halt_on_store_failure=True,
    )
    assert len(results) == 2
    assert results[0].model_id == "megadetector_v6"
    assert results[0].image_size == (1920, 1080)
    assert len(results[0].detections) == 1
    assert results[1].image_size == (640, 480)
    assert len(results[1].detections) == 0


def test_pipeline(httpserver, client):
    httpserver.expect_request(
        "/v1/pipeline",
        method="POST",
        query_string="pipeline=md_speciesnet&top_k=2&threshold=0.3&max_detections=5&store=true&halt_on_store_failure=true",
    ).respond_with_json(PIPELINE_RESPONSE)
    result = client.pipeline(
        image=b"\xff\xd8fake",
        pipeline="md_speciesnet",
        threshold=0.3,
        top_k=2,
        max_detections=5,
        store=True,
        halt_on_store_failure=True,
    )
    assert isinstance(result, PipelineResult)
    assert result.pipeline_id == "md_speciesnet"
    assert len(result.detections) == 2
    # First detection has classification
    pd0 = result.detections[0]
    assert pd0.detection.label == "animal"
    assert pd0.classification is not None
    assert pd0.classification.label == "deer"
    # Second detection has no classification
    pd1 = result.detections[1]
    assert pd1.classification is None


def test_list_models(httpserver, client):
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        MODELS_RESPONSE
    )
    models = client.list_models()
    assert len(models) == 2
    assert isinstance(models[0], ModelInfo)
    assert models[0].id == "megadetector_v6"
    assert models[0].default is True
    assert models[0].version == "1.0.0"
    assert models[0].description == "MegaDetector v6 general wildlife detector"
    assert models[0].onnx_sha256 == "abcdef0123456789"
    assert models[0].onnx_size_bytes == 123456789
    assert models[1].model_type == "classifier"
    assert models[1].default is False
    assert models[1].version is None
    assert models[1].onnx_sha256 is None


def test_load_model(httpserver, client):
    httpserver.expect_request("/v1/models/load", method="POST").respond_with_json(
        LOAD_MODEL_RESPONSE
    )
    info = client.load_model("megadetector_v6")
    assert isinstance(info, ModelInfo)
    assert info.id == "megadetector_v6"
    assert info.model_type == "detector"
    assert info.default is True
    assert info.version == "1.0.0"
    assert info.onnx_sha256 == "abcdef0123456789"


def test_list_models_legacy_wire_format(httpserver, client):
    legacy_response = {
        "models": [
            {"id": "megadetector_v6", "model_type": "detector"},
            {"id": "speciesnet", "model_type": "classifier"},
        ]
    }
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        legacy_response
    )
    models = client.list_models()
    assert len(models) == 2
    assert models[0].id == "megadetector_v6"
    assert models[0].default is False
    assert models[0].version is None
    assert models[0].description is None
    assert models[0].onnx_sha256 is None
    assert models[0].onnx_size_bytes is None


def test_unload_model(httpserver, client):
    httpserver.expect_request(
        "/v1/models/megadetector_v6", method="DELETE"
    ).respond_with_data("", status=204)
    client.unload_model("megadetector_v6")  # should not raise


def test_unload_model_escapes_model_id():
    class StubClient:
        path: str | None = None

        def delete(self, path: str) -> httpx.Response:
            self.path = path
            return httpx.Response(204)

        def close(self) -> None:
            pass

    client = SparrowEngineClient()
    client._client.close()
    stub = StubClient()
    client._client = stub  # type: ignore[assignment]

    client.unload_model("model/with space")

    assert stub.path == "/v1/models/model%2Fwith%20space"


def test_health(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        HEALTH_RESPONSE
    )
    h = client.health()
    assert h["status"] == "ready"
    assert h["models_loaded"] == 2
    assert h["version"] == "0.1.0"


def test_error_handling(httpserver, client):
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        ERROR_RESPONSE, status=404
    )
    with pytest.raises(SparrowEngineClientError) as exc_info:
        client.list_models()
    err = exc_info.value
    assert err.code == "MODEL_NOT_LOADED"
    assert err.status == 404
    assert "foo" in err.message


def test_bbox_to_pixels():
    bbox = BBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6)
    px = bbox.to_pixels(1920, 1080)
    assert px == (192, 216, 960, 648)


def test_image_file_path(tmp_path):
    img_path = tmp_path / "photo.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    name, data, mime = SparrowEngineClient._image_file(img_path)
    assert name == "photo.jpg"
    assert data == b"\xff\xd8\xff\xe0fake-jpeg"
    assert mime == "image/jpeg"


def test_image_file_bytes():
    raw = b"\xff\xd8\xff\xe0raw-bytes"
    name, data, mime = SparrowEngineClient._image_file(raw)
    assert name == "image.jpg"
    assert data == raw
    assert mime == "image/jpeg"


def test_image_file_fileobj():
    raw = b"\xff\xd8\xff\xe0fileobj-data"
    name, data, mime = SparrowEngineClient._image_file(io.BytesIO(raw))
    assert name == "image.jpg"
    assert data == raw
    assert mime == "image/jpeg"


# ---------------------------------------------------------------------------
# wait_ready / is_ready regression tests (BUG-01 coverage)
# ---------------------------------------------------------------------------


def test_is_ready_true(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        HEALTH_RESPONSE
    )
    assert client.is_ready() is True


def test_is_ready_false_non_ready_status(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        {"status": "loading", "models_loaded": 0, "pipelines_loaded": 0, "version": "0.1.0"}
    )
    assert client.is_ready() is False


def test_is_ready_false_connection_error():
    """is_ready returns False when server is unreachable."""
    c = SparrowEngineClient(base_url="http://127.0.0.1:1", timeout=0.1)
    try:
        assert c.is_ready() is False
    finally:
        c.close()


def test_wait_ready_success(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        HEALTH_RESPONSE
    )
    client.wait_ready(timeout=2.0, interval=0.1)  # should not raise


def test_wait_ready_timeout(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        {"status": "loading", "models_loaded": 0, "pipelines_loaded": 0, "version": "0.1.0"}
    )
    with pytest.raises(TimeoutError, match="not ready after"):
        client.wait_ready(timeout=0.3, interval=0.1)


def test_wait_ready_surfaces_malformed_success(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_data(
        "not-json", status=200, content_type="application/json"
    )
    with pytest.raises(ValueError):
        client.wait_ready(timeout=0.3, interval=0.1)


# ---------------------------------------------------------------------------
# Error edge cases
# ---------------------------------------------------------------------------


def test_error_non_json_response(httpserver, client):
    """Non-JSON error (e.g., reverse proxy HTML 502) raises HTTPStatusError, not SparrowEngineClientError."""
    httpserver.expect_request("/v1/models", method="GET").respond_with_data(
        "<html>502 Bad Gateway</html>", status=502
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.list_models()


# ---------------------------------------------------------------------------
# Management authentication (bearer token)
# ---------------------------------------------------------------------------

MGMT_TOKEN = "s3cret-token"


@pytest.fixture()
def auth_client(httpserver):
    """Client configured with a management bearer token."""
    with SparrowEngineClient(
        base_url=httpserver.url_for(""), timeout=5.0, management_token=MGMT_TOKEN
    ) as c:
        yield c


def _last_request(httpserver):
    """The most recently handled werkzeug request from the mock server log."""
    assert httpserver.log, "no request was recorded by the mock server"
    return httpserver.log[-1][0]


def test_management_sends_bearer_token_when_configured(httpserver, auth_client):
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        MODELS_RESPONSE
    )
    auth_client.list_models()
    req = _last_request(httpserver)
    assert req.headers.get("Authorization") == f"Bearer {MGMT_TOKEN}"


def test_management_omits_auth_header_without_token(httpserver, client):
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        MODELS_RESPONSE
    )
    client.list_models()
    req = _last_request(httpserver)
    assert req.headers.get("Authorization") is None


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.list_models(),
        lambda c: c.load_model("m"),
        lambda c: c.unload_model("m"),
        lambda c: c.trt_warmup("m"),
        lambda c: c.list_pipelines(),
        lambda c: c.create_pipeline("p", "d", "k"),
        lambda c: c.load_pipeline("p"),
        lambda c: c.delete_pipeline("p"),
    ],
)
def test_all_management_methods_send_bearer_token(httpserver, auth_client, call):
    # A permissive catch-all handler: every management verb/path returns a
    # trivially-parseable body/status so the call completes and we can inspect
    # the recorded Authorization header.
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        MODELS_RESPONSE
    )
    httpserver.expect_request("/v1/models/load", method="POST").respond_with_json(
        LOAD_MODEL_RESPONSE
    )
    httpserver.expect_request("/v1/models/m", method="DELETE").respond_with_data(
        "", status=204
    )
    httpserver.expect_request(
        "/v1/models/m/trt-warmup", method="POST"
    ).respond_with_json({"trt_state": "trt_ready"})
    httpserver.expect_request("/v1/pipelines", method="GET").respond_with_json(
        {"pipelines": []}
    )
    httpserver.expect_request("/v1/pipelines", method="POST").respond_with_json(
        PIPELINE_INFO_RESPONSE, status=201
    )
    httpserver.expect_request("/v1/pipelines/load", method="POST").respond_with_json(
        PIPELINE_INFO_RESPONSE
    )
    httpserver.expect_request("/v1/pipelines/p", method="DELETE").respond_with_data(
        "", status=204
    )
    call(auth_client)
    req = _last_request(httpserver)
    assert req.headers.get("Authorization") == f"Bearer {MGMT_TOKEN}"


@pytest.mark.parametrize(
    ("path", "method", "call"),
    [
        ("/v1/detect", "POST", lambda c: c.detect(b"\xff\xd8x", model="m")),
        ("/v1/catalog", "GET", lambda c: c.catalog()),
        ("/v1/health", "GET", lambda c: c.health()),
        ("/healthz", "GET", lambda c: c.healthz()),
    ],
)
def test_open_endpoints_never_send_auth_header(
    httpserver, auth_client, path, method, call
):
    # Even with a token configured, inference / catalog / health stay open and
    # must NOT carry the Authorization header.
    responses = {
        "/v1/detect": DETECT_RESPONSE,
        "/v1/catalog": CATALOG_RESPONSE,
        "/v1/health": HEALTH_RESPONSE,
        "/healthz": {"alive": True},
    }
    httpserver.expect_request(path, method=method).respond_with_json(responses[path])
    call(auth_client)
    req = _last_request(httpserver)
    assert req.headers.get("Authorization") is None


def test_management_unauthorized_raises(httpserver, client):
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        {
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Missing bearer credential for the management API.",
                "status": 401,
            }
        },
        status=401,
    )
    with pytest.raises(SparrowEngineClientError) as exc_info:
        client.list_models()
    assert exc_info.value.code == "UNAUTHORIZED"
    assert exc_info.value.status == 401


# ---------------------------------------------------------------------------
# Multiclass audio
# ---------------------------------------------------------------------------

AUDIO_MULTICLASS_RESPONSE = {
    "model_id": "md_audiobirds_v1",
    "duration_s": 5.0,
    "sample_rate": 48000,
    "processing_time_ms": 90.0,
    "segments": [
        {
            "start_time_s": 0.0,
            "end_time_s": 1.0,
            "confidence": 0.9,
            "classes": [
                {"class_idx": 0, "label": "sparrow", "probability": 0.7},
                {"class_idx": 1, "probability": 0.2},
                {"class_idx": 2, "label": "thrush", "probability": 0.1},
            ],
        }
    ],
}


def test_detect_audio_multiclass(httpserver, client):
    httpserver.expect_request(
        "/v1/audio/detect", method="POST"
    ).respond_with_json(AUDIO_MULTICLASS_RESPONSE)
    result = client.detect_audio(audio=b"fake-wav", model="md_audiobirds_v1")
    seg = result.segments[0]
    assert seg.classes is not None
    assert len(seg.classes) == 3
    assert isinstance(seg.classes[0], AudioClass)
    assert seg.classes[0].class_idx == 0
    assert seg.classes[0].label == "sparrow"
    assert seg.classes[0].probability == pytest.approx(0.7)
    # Optional label omitted by the server → None, not a crash.
    assert seg.classes[1].label is None
    assert seg.classes[1].class_idx == 1
    assert seg.classes[2].label == "thrush"


def test_detect_audio_binary_preserves_no_classes(httpserver, client):
    # AUDIO_RESPONSE has no `classes` key (binary/single-class path).
    httpserver.expect_request("/v1/audio/detect", method="POST").respond_with_json(
        AUDIO_RESPONSE
    )
    result = client.detect_audio(audio=b"fake-wav", model="md_audiobirds_v1")
    assert isinstance(result, AudioResult)
    assert result.segments[0].classes is None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBED_RESPONSE = {
    "embed_schema_version": "1.0",
    "model_id": "encoder-a",
    "embedding_version": "encoder-space-1",
    "model_hash": "abc123",
    "embedding_dim": 3,
    "normalized": True,
    "metric": "cosine",
    "image_size": [640, 480],
    "processing_time_ms": 1.25,
    "embedding": [0.1, 0.2, 0.3],
}

EMBED_BATCH_RESPONSE = {
    "embed_schema_version": "1.0",
    "model_id": "encoder-a",
    "embedding_version": "encoder-space-1",
    "model_hash": "abc123",
    "embedding_dim": 3,
    "normalized": True,
    "metric": "cosine",
    "count": 2,
    "processing_time_ms": 2.5,
    "results": [
        {
            "index": 0,
            "image_size": [640, 480],
            "processing_time_ms": 1.0,
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "index": 1,
            "image_size": [320, 240],
            "processing_time_ms": 1.5,
            "embedding": [0.4, 0.5, 0.6],
        },
    ],
}


def test_embed(httpserver, client):
    httpserver.expect_request(
        "/v1/embed", method="POST", query_string="model=encoder-a"
    ).respond_with_json(EMBED_RESPONSE)
    result = client.embed(image=b"\xff\xd8img", model="encoder-a")
    assert isinstance(result, EmbedResult)
    assert result.model_id == "encoder-a"
    assert result.embedding_version == "encoder-space-1"
    assert result.model_hash == "abc123"
    assert result.embedding_dim == 3
    assert result.normalized is True
    assert result.metric == "cosine"
    assert result.image_size == (640, 480)
    assert result.embedding == [0.1, 0.2, 0.3]
    assert result.embed_schema_version == "1.0"


def test_embed_store_flags(httpserver, client):
    httpserver.expect_request(
        "/v1/embed",
        method="POST",
        query_string={
            "model": "encoder-a",
            "store": "true",
            "halt_on_store_failure": "true",
        },
    ).respond_with_json(EMBED_RESPONSE)
    result = client.embed(
        image=b"\xff\xd8img",
        model="encoder-a",
        store=True,
        halt_on_store_failure=True,
    )
    assert isinstance(result, EmbedResult)


def test_embed_batch(httpserver, client):
    httpserver.expect_request(
        "/v1/embed/batch", method="POST", query_string="model=encoder-a"
    ).respond_with_json(EMBED_BATCH_RESPONSE)
    result = client.embed_batch(
        images=[b"\xff\xd8a", b"\xff\xd8b"], model="encoder-a"
    )
    assert isinstance(result, EmbedBatchResult)
    assert result.count == 2
    assert len(result.results) == 2
    assert isinstance(result.results[0], EmbedBatchItem)
    assert result.results[0].index == 0
    assert result.results[0].image_size == (640, 480)
    assert result.results[1].embedding == [0.4, 0.5, 0.6]
    assert result.embed_schema_version == "1.0"


def test_embed_wrong_model_type_error(httpserver, client):
    httpserver.expect_request("/v1/embed", method="POST").respond_with_json(
        {
            "error": {
                "code": "WRONG_MODEL_TYPE",
                "message": "model is not an encoder",
                "status": 400,
            }
        },
        status=400,
    )
    with pytest.raises(SparrowEngineClientError) as exc_info:
        client.embed(image=b"\xff\xd8x", model="not-an-encoder")
    assert exc_info.value.code == "WRONG_MODEL_TYPE"
    assert exc_info.value.status == 400


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

CATALOG_RESPONSE = [
    {
        "model_id": "megadetector_v6",
        "model_type": "detector",
        "framework": "onnx",
        "loaded": True,
        "trt_state": "cuda_ready",
        "display_name": "MegaDetector v6",
        "family": ["megadetector"],
    },
    {
        "model_id": "encoder-a",
        "model_type": "image_encoder",
        "framework": "onnx",
        "loaded": False,
        "trt_state": "not_loaded",
        "embedding_dim": 512,
        "embedding_version": "encoder-space-1",
        "normalized": True,
        "metric": "cosine",
    },
    {
        "model_id": "md_speciesnet",
        "model_type": "cascade",
        "framework": "cascade",
        "loaded": False,
        "trt_state": "unsupported",
    },
]


def test_catalog(httpserver, client):
    httpserver.expect_request("/v1/catalog", method="GET").respond_with_json(
        CATALOG_RESPONSE
    )
    entries = client.catalog()
    assert len(entries) == 3
    assert all(isinstance(e, CatalogEntry) for e in entries)
    det = entries[0]
    assert det.model_id == "megadetector_v6"
    assert det.model_type == "detector"
    assert det.framework == "onnx"
    assert det.loaded is True
    assert det.trt_state == "cuda_ready"
    # Open-ended model-zoo metadata preserved in raw.
    assert det.raw["display_name"] == "MegaDetector v6"
    assert det.raw["family"] == ["megadetector"]
    enc = entries[1]
    assert enc.embedding_dim == 512
    assert enc.embedding_version == "encoder-space-1"
    assert enc.normalized is True
    assert enc.metric == "cosine"
    # Missing optionals default to None.
    assert det.embedding_dim is None
    assert det.trt_detail is None


# ---------------------------------------------------------------------------
# TensorRT warm-up
# ---------------------------------------------------------------------------


def test_trt_warmup_started(httpserver, auth_client):
    httpserver.expect_request(
        "/v1/models/megadetector_v6/trt-warmup", method="POST"
    ).respond_with_json(
        {
            "trt_state": "trt_warming",
            "poll": {"method": "GET", "path": "/v1/catalog"},
        },
        status=202,
    )
    result = auth_client.trt_warmup("megadetector_v6")
    assert isinstance(result, TrtWarmupResult)
    assert result.trt_state == "trt_warming"
    assert result.started is True
    assert result.poll == {"method": "GET", "path": "/v1/catalog"}


def test_trt_warmup_already_ready(httpserver, auth_client):
    httpserver.expect_request(
        "/v1/models/megadetector_v6/trt-warmup", method="POST"
    ).respond_with_json({"trt_state": "trt_ready"}, status=200)
    result = auth_client.trt_warmup("megadetector_v6")
    assert result.trt_state == "trt_ready"
    assert result.started is False
    assert result.poll is None


def test_trt_warmup_unsupported_hardware(httpserver, auth_client):
    httpserver.expect_request(
        "/v1/models/megadetector_v6/trt-warmup", method="POST"
    ).respond_with_json(
        {
            "error": {
                "code": "TRT_UNSUPPORTED_HARDWARE",
                "message": "cpu build",
                "status": 422,
                "reason": "cpu_build",
            }
        },
        status=422,
    )
    with pytest.raises(SparrowEngineClientError) as exc_info:
        auth_client.trt_warmup("megadetector_v6")
    assert exc_info.value.code == "TRT_UNSUPPORTED_HARDWARE"
    assert exc_info.value.status == 422


def test_trt_warmup_escapes_model_id():
    class StubClient:
        path = None

        def post(self, path, **kwargs):
            self.path = path
            return httpx.Response(200, json={"trt_state": "trt_ready"})

        def close(self):
            pass

    client = SparrowEngineClient()
    client._client.close()
    stub = StubClient()
    client._client = stub  # type: ignore[assignment]
    client.trt_warmup("model/with space")
    assert stub.path == "/v1/models/model%2Fwith%20space/trt-warmup"


# ---------------------------------------------------------------------------
# Pipeline management
# ---------------------------------------------------------------------------

PIPELINE_INFO_RESPONSE = {
    "id": "md_speciesnet",
    "steps": [
        {"role": "detector", "model_id": "megadetector_v6"},
        {"role": "classifier", "model_id": "speciesnet"},
    ],
}

PIPELINES_LIST_RESPONSE = {"pipelines": [PIPELINE_INFO_RESPONSE]}


def test_list_pipelines(httpserver, auth_client):
    httpserver.expect_request("/v1/pipelines", method="GET").respond_with_json(
        PIPELINES_LIST_RESPONSE
    )
    pipelines = auth_client.list_pipelines()
    assert len(pipelines) == 1
    p = pipelines[0]
    assert isinstance(p, PipelineInfo)
    assert p.id == "md_speciesnet"
    assert len(p.steps) == 2
    assert isinstance(p.steps[0], PipelineStep)
    assert p.steps[0].role == "detector"
    assert p.steps[0].model_id == "megadetector_v6"
    assert p.steps[1].role == "classifier"


def test_create_pipeline_created(httpserver, auth_client):
    httpserver.expect_request(
        "/v1/pipelines",
        method="POST",
        json={
            "id": "md_speciesnet",
            "detector": "megadetector_v6",
            "classifier": "speciesnet",
            "replace": False,
            "persist": False,
        },
    ).respond_with_json(PIPELINE_INFO_RESPONSE, status=201)
    info = auth_client.create_pipeline(
        "md_speciesnet", detector="megadetector_v6", classifier="speciesnet"
    )
    assert isinstance(info, PipelineInfo)
    assert info.id == "md_speciesnet"
    assert info.steps[1].model_id == "speciesnet"


def test_create_pipeline_existing_returns_200(httpserver, auth_client):
    httpserver.expect_request(
        "/v1/pipelines",
        method="POST",
        json={
            "id": "md_speciesnet",
            "detector": "megadetector_v6",
            "classifier": "speciesnet",
            "replace": True,
            "persist": True,
        },
    ).respond_with_json(PIPELINE_INFO_RESPONSE, status=200)
    info = auth_client.create_pipeline(
        "md_speciesnet",
        detector="megadetector_v6",
        classifier="speciesnet",
        replace=True,
        persist=True,
    )
    assert info.id == "md_speciesnet"


def test_create_pipeline_conflict_raises(httpserver, auth_client):
    httpserver.expect_request("/v1/pipelines", method="POST").respond_with_json(
        {
            "error": {
                "code": "PIPELINE_ALIAS_CONFLICT",
                "message": "pipeline alias exists with a different definition",
                "status": 409,
            }
        },
        status=409,
    )
    with pytest.raises(SparrowEngineClientError) as exc_info:
        auth_client.create_pipeline("md_speciesnet", detector="d", classifier="k")
    assert exc_info.value.code == "PIPELINE_ALIAS_CONFLICT"
    assert exc_info.value.status == 409


def test_load_pipeline(httpserver, auth_client):
    httpserver.expect_request(
        "/v1/pipelines/load",
        method="POST",
        json={"pipeline_id": "md_speciesnet"},
    ).respond_with_json(PIPELINE_INFO_RESPONSE)
    info = auth_client.load_pipeline("md_speciesnet")
    assert isinstance(info, PipelineInfo)
    assert info.id == "md_speciesnet"


def test_delete_pipeline(httpserver, auth_client):
    httpserver.expect_request(
        "/v1/pipelines/md_speciesnet", method="DELETE"
    ).respond_with_data("", status=204)
    auth_client.delete_pipeline("md_speciesnet")  # should not raise


def test_delete_pipeline_not_found_raises(httpserver, auth_client):
    httpserver.expect_request(
        "/v1/pipelines/ghost", method="DELETE"
    ).respond_with_json(
        {
            "error": {
                "code": "PIPELINE_NOT_FOUND",
                "message": "no such pipeline",
                "status": 404,
            }
        },
        status=404,
    )
    with pytest.raises(SparrowEngineClientError) as exc_info:
        auth_client.delete_pipeline("ghost")
    assert exc_info.value.status == 404


def test_delete_pipeline_escapes_id():
    class StubClient:
        path = None

        def delete(self, path, **kwargs):
            self.path = path
            return httpx.Response(204)

        def close(self):
            pass

    client = SparrowEngineClient()
    client._client.close()
    stub = StubClient()
    client._client = stub  # type: ignore[assignment]
    client.delete_pipeline("alias/with space")
    assert stub.path == "/v1/pipelines/alias%2Fwith%20space"


# ---------------------------------------------------------------------------
# healthz liveness
# ---------------------------------------------------------------------------


def test_healthz(httpserver, client):
    httpserver.expect_request("/healthz", method="GET").respond_with_json(
        {"alive": True}
    )
    assert client.healthz() == {"alive": True}


# ---------------------------------------------------------------------------
# Ad-hoc pipeline (detector + classifier)
# ---------------------------------------------------------------------------


def test_pipeline_adhoc(httpserver, client):
    httpserver.expect_request(
        "/v1/pipeline",
        method="POST",
        query_string={
            "detector": "megadetector_v6",
            "classifier": "speciesnet",
            "top_k": "5",
        },
    ).respond_with_json(PIPELINE_RESPONSE)
    result = client.pipeline(
        image=b"\xff\xd8fake",
        detector="megadetector_v6",
        classifier="speciesnet",
    )
    assert isinstance(result, PipelineResult)
    assert result.pipeline_id == "md_speciesnet"
    assert len(result.detections) == 2


def test_pipeline_named_still_supported(httpserver, client):
    httpserver.expect_request(
        "/v1/pipeline",
        method="POST",
        query_string={"pipeline": "md_speciesnet", "top_k": "5"},
    ).respond_with_json(PIPELINE_RESPONSE)
    result = client.pipeline(image=b"\xff\xd8fake", pipeline="md_speciesnet")
    assert result.pipeline_id == "md_speciesnet"


def test_pipeline_rejects_both_named_and_adhoc(client):
    with pytest.raises(ValueError, match="not both"):
        client.pipeline(
            image=b"x",
            pipeline="md_speciesnet",
            detector="megadetector_v6",
            classifier="speciesnet",
        )


def test_pipeline_rejects_neither(client):
    with pytest.raises(ValueError, match="is required"):
        client.pipeline(image=b"x")


def test_pipeline_rejects_partial_adhoc(client):
    with pytest.raises(ValueError, match="both"):
        client.pipeline(image=b"x", detector="megadetector_v6")
    with pytest.raises(ValueError, match="both"):
        client.pipeline(image=b"x", classifier="speciesnet")
