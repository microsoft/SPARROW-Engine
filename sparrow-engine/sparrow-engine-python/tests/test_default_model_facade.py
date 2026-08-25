"""Facade tests for the ``model``-optional ``detect`` / ``detect_audio`` path.

``detect`` and ``detect_audio`` accept an omitted (``None``) model, mirroring
the ``spe detect`` / ``spe detect-audio`` CLI (Phase 2.5 functionality-
consistency rule). The resolver precedence is:

  1. an explicit ``model`` argument, used verbatim;
  2. the catalog default whose ``model_type`` matches the task family
     (``ModelInfo.default`` marks the default *for its type*); a default of the
     wrong type is ignored;
  3. the same stable per-task fallback id the CLI uses
     (``MDV6-yolov10-e`` for images, ``md-audiobirds-v1`` for audio).

These tests mock ``_get_engine`` (the same pattern as ``test_embed_facade``),
so they need no ONNX model or ORT runtime — only the native extension
importable for ``import sparrow_engine``.
"""
from __future__ import annotations

from types import SimpleNamespace

import sparrow_engine


class RecordingEngine:
    """Engine double that records the model id ``detect`` / ``detect_audio``
    resolve to, and serves a fixed catalog from ``list_models``.
    """

    def __init__(self, models):
        self._models = list(models)
        self.detect_model = None
        self.detect_audio_model = None

    def list_models(self):
        return list(self._models)

    def detect(
        self,
        paths,
        model,
        threshold=None,
        max_detections=None,
        progress_callback=None,
    ):
        self.detect_model = model
        return []

    def detect_audio(
        self,
        paths,
        model,
        threshold=None,
        stride_s=None,
        segment_duration_s=None,
        progress_callback=None,
    ):
        self.detect_audio_model = model
        return []


def _model(model_id: str, model_type: str, *, default: bool = False) -> SimpleNamespace:
    """Minimal ``ModelInfo`` stand-in carrying the fields the resolver reads."""
    return SimpleNamespace(id=model_id, model_type=model_type, default=default)


def _install(monkeypatch, engine: RecordingEngine) -> None:
    monkeypatch.setattr(sparrow_engine, "_get_engine", lambda: engine)


# --- Module wording (CPU-or-GPU flavor generalization) -----------------------


def test_module_docstring_is_flavor_neutral() -> None:
    doc = sparrow_engine.__doc__ or ""
    # The facade ships unchanged in BOTH the CPU and GPU wheels, so the old
    # "powered by sparrow-engine-cpu" claim was wrong for the GPU wheel.
    assert "sparrow-engine-cpu" not in doc
    assert "powered by sparrow-engine" in doc


# --- detect() precedence -----------------------------------------------------


def test_detect_explicit_model_used_verbatim(monkeypatch) -> None:
    engine = RecordingEngine([_model("some-detector", "detector", default=True)])
    _install(monkeypatch, engine)

    sparrow_engine.detect([], model="my-custom-detector")

    # An explicit id is never second-guessed, even when a catalog default exists.
    assert engine.detect_model == "my-custom-detector"


def test_detect_uses_catalog_default_detector(monkeypatch) -> None:
    engine = RecordingEngine([
        _model("catalog-detector", "detector", default=True),
        _model("speciesnet", "classifier", default=True),
    ])
    _install(monkeypatch, engine)

    sparrow_engine.detect([])

    assert engine.detect_model == "catalog-detector"


def test_detect_prefers_standard_detector_over_overhead(monkeypatch) -> None:
    engine = RecordingEngine([
        _model("herdnet", "overhead_detector", default=True),
        _model("standard-detector", "detector", default=True),
    ])
    _install(monkeypatch, engine)

    sparrow_engine.detect([])

    # Family order is (detector, overhead_detector): standard wins.
    assert engine.detect_model == "standard-detector"


def test_detect_uses_overhead_default_when_only_overhead(monkeypatch) -> None:
    engine = RecordingEngine([_model("herdnet", "overhead_detector", default=True)])
    _install(monkeypatch, engine)

    sparrow_engine.detect([])

    assert engine.detect_model == "herdnet"


def test_detect_rejects_wrong_type_default(monkeypatch) -> None:
    # Only audio / classifier defaults present: a detection command must NOT
    # borrow them — it falls back to the CLI's stable image id.
    engine = RecordingEngine([
        _model("md-audiobirds-v1", "audio_detector", default=True),
        _model("speciesnet", "classifier", default=True),
    ])
    _install(monkeypatch, engine)

    sparrow_engine.detect([])

    assert engine.detect_model == "MDV6-yolov10-e"


def test_detect_falls_back_when_no_default_flagged(monkeypatch) -> None:
    engine = RecordingEngine([_model("some-detector", "detector", default=False)])
    _install(monkeypatch, engine)

    sparrow_engine.detect([])

    assert engine.detect_model == "MDV6-yolov10-e"


def test_detect_falls_back_on_empty_catalog(monkeypatch) -> None:
    engine = RecordingEngine([])
    _install(monkeypatch, engine)

    sparrow_engine.detect([])

    assert engine.detect_model == "MDV6-yolov10-e"


# --- detect_audio() precedence ----------------------------------------------


def test_detect_audio_explicit_model_used_verbatim(monkeypatch) -> None:
    engine = RecordingEngine([_model("md-audiobirds-v1", "audio_detector", default=True)])
    _install(monkeypatch, engine)

    sparrow_engine.detect_audio([], model="perch-v2")

    assert engine.detect_audio_model == "perch-v2"


def test_detect_audio_uses_catalog_default(monkeypatch) -> None:
    engine = RecordingEngine([_model("catalog-audio", "audio_detector", default=True)])
    _install(monkeypatch, engine)

    sparrow_engine.detect_audio([])

    assert engine.detect_audio_model == "catalog-audio"


def test_detect_audio_prefers_detector_over_classifier(monkeypatch) -> None:
    engine = RecordingEngine([
        _model("perch-v2", "audio_classifier", default=True),
        _model("bird-detector", "audio_detector", default=True),
    ])
    _install(monkeypatch, engine)

    sparrow_engine.detect_audio([])

    # Family order is (audio_detector, audio_classifier): the detector wins.
    assert engine.detect_audio_model == "bird-detector"


def test_detect_audio_uses_classifier_default_when_only_classifier(monkeypatch) -> None:
    engine = RecordingEngine([_model("perch-v2", "audio_classifier", default=True)])
    _install(monkeypatch, engine)

    sparrow_engine.detect_audio([])

    assert engine.detect_audio_model == "perch-v2"


def test_detect_audio_rejects_wrong_type_default(monkeypatch) -> None:
    # Only an image-detector default present: audio must not borrow it.
    engine = RecordingEngine([_model("standard-detector", "detector", default=True)])
    _install(monkeypatch, engine)

    sparrow_engine.detect_audio([])

    assert engine.detect_audio_model == "md-audiobirds-v1"


def test_detect_audio_falls_back_when_no_default_flagged(monkeypatch) -> None:
    engine = RecordingEngine([_model("some-audio", "audio_detector", default=False)])
    _install(monkeypatch, engine)

    sparrow_engine.detect_audio([])

    assert engine.detect_audio_model == "md-audiobirds-v1"
