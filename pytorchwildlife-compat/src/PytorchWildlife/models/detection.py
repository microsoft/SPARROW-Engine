"""PytorchWildlife 1.x detection classes, running on sparrow_engine.

Each class maps to a Sparrow model zoo ID. Results keep the 1.x dict shape:
``{"img_id", "detections", "labels", "normalized_coords"}`` with ``detections.xyxy`` in
pixels.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import sparrow_engine as spe

from .. import _engine as _e

__all__ = [
    "MegaDetectorV6",
    "MegaDetectorV6MIT",
    "MegaDetectorV6Apache",
    "MegaDetectorV5",
    "DeepfauneDetector",
    "HerdNet",
    "OWL",
]


class _SparrowDetector:
    """Base class: a 1.x-style detector backed by one Sparrow zoo model."""

    CLASS_NAMES: dict[int, str] = {}
    VERSIONS: dict[str, str] = {}
    DEFAULT_VERSION: str | None = None

    def __init__(self, weights=None, device="cpu", pretrained=True, version=None):
        if weights is not None:
            raise NotImplementedError(
                "PytorchWildlife 2.0 does not load PyTorch weight files. Export the model to "
                "ONNX and run it with sparrow_engine, or use a Sparrow model zoo model."
            )
        version = version or self.DEFAULT_VERSION
        if version not in self.VERSIONS:
            raise ValueError(
                f"Unsupported version '{version}' for {type(self).__name__}; "
                f"choose one of {sorted(self.VERSIONS)}."
            )
        self.version = version
        self.device = device
        self.model_id = self.VERSIONS[version]
        path = _e.ensure_model(self.model_id)
        if not self.CLASS_NAMES:
            self.CLASS_NAMES = _e.read_labels(path)

    def _class_name(self, class_id: int) -> str:
        return self.CLASS_NAMES.get(int(class_id), str(class_id))

    def _to_result(self, res, img_id, id_strip, size=None) -> dict:
        width, height = size if size is not None else res.image_size
        xyxy, conf, cls = [], [], []
        for det in res.detections if res is not None else []:
            b = det.bbox
            xyxy.append([b.x_min * width, b.y_min * height, b.x_max * width, b.y_max * height])
            conf.append(det.confidence)
            cls.append(det.label_id)
        detections = _e.make_detections(xyxy, conf, cls)
        labels = [f"{self._class_name(c)} {p:0.2f}" for c, p in zip(cls, conf, strict=True)]
        normalized = [[x1 / width, y1 / height, x2 / width, y2 / height] for x1, y1, x2, y2 in xyxy]
        return {
            "img_id": _e.strip_id(img_id, id_strip),
            "detections": detections,
            "labels": labels,
            "normalized_coords": normalized,
        }

    def _detect_paths(self, paths, det_conf_thres) -> dict:
        """Run the engine on ``paths`` and return ``{realpath: DetectResult}``."""
        if not paths:
            return {}
        results = spe.detect(list(paths), model=self.model_id, threshold=det_conf_thres)
        return {os.path.realpath(r.path): r for r in results}

    def single_image_detection(self, img, img_path=None, det_conf_thres=0.2, id_strip=None):
        """Detect objects in one image (a path or an RGB ndarray)."""
        if isinstance(img, (str, os.PathLike)):
            path = str(img)
            if img_path is None:
                img_path = path
            by_path = self._detect_paths([path], det_conf_thres)
            res = by_path.get(os.path.realpath(path))
            if res is None:
                raise RuntimeError(f"sparrow_engine could not read image {path}")
            return self._to_result(res, img_path, id_strip)
        arr = np.asarray(img)
        with _e.temp_dir() as tmp:
            (path,) = _e.write_temp_images([arr], tmp)
            res = self._detect_paths([path], det_conf_thres)[os.path.realpath(path)]
        return self._to_result(res, img_path, id_strip, size=(arr.shape[1], arr.shape[0]))

    def batch_image_detection(self, data_source, batch_size=16, det_conf_thres=0.2, id_strip=None):
        """Detect objects in a folder (searched recursively) or a list of RGB ndarrays.

        ``batch_size`` is accepted for compatibility; the engine schedules its own batches.
        """
        if isinstance(data_source, (list, tuple, np.ndarray)):
            arrays = list(data_source)
            with _e.temp_dir() as tmp:
                paths = _e.write_temp_images(arrays, tmp)
                by_path = self._detect_paths(paths, det_conf_thres)
                return [
                    self._to_result(
                        by_path.get(os.path.realpath(p)), str(i), id_strip,
                        size=(np.asarray(a).shape[1], np.asarray(a).shape[0]),
                    )
                    for i, (p, a) in enumerate(zip(paths, arrays, strict=True))
                ]
        paths = _e.list_images(str(data_source))
        by_path = self._detect_paths(paths, det_conf_thres)
        results = []
        for p in paths:
            res = by_path.get(os.path.realpath(p))
            if res is None:
                warnings.warn(
                    f"sparrow_engine skipped {p}; it is missing from the results.", stacklevel=2
                )
                continue
            results.append(self._to_result(res, p, id_strip))
        return results


_MD_CLASSES = {0: "animal", 1: "person", 2: "vehicle"}


class MegaDetectorV6(_SparrowDetector):
    CLASS_NAMES = dict(_MD_CLASSES)
    VERSIONS = {v: v for v in (
        "MDV6-yolov9-c", "MDV6-yolov9-e", "MDV6-yolov10-c", "MDV6-yolov10-e", "MDV6-rtdetr-c",
    )}
    DEFAULT_VERSION = "MDV6-yolov9-c"


class MegaDetectorV6MIT(_SparrowDetector):
    CLASS_NAMES = dict(_MD_CLASSES)
    VERSIONS = {v: v for v in ("MDV6-mit-yolov9-c", "MDV6-mit-yolov9-e")}
    DEFAULT_VERSION = "MDV6-mit-yolov9-c"


class MegaDetectorV6Apache(_SparrowDetector):
    CLASS_NAMES = dict(_MD_CLASSES)
    VERSIONS = {v: v for v in ("MDV6-apa-rtdetr-c", "MDV6-apa-rtdetr-e")}
    DEFAULT_VERSION = "MDV6-apa-rtdetr-c"


class MegaDetectorV5(_SparrowDetector):
    """MegaDetector v5. The zoo lists MDV5a/MDV5b but may not host the files; install them
    into the sparrow-engine model directory if the download fails."""

    CLASS_NAMES = dict(_MD_CLASSES)
    VERSIONS = {"a": "MDV5a", "b": "MDV5b"}
    DEFAULT_VERSION = "a"


class DeepfauneDetector(_SparrowDetector):
    CLASS_NAMES = dict(_MD_CLASSES)
    VERSIONS = {"deepfaune-yolo8s": "deepfaune-yolo8s"}
    DEFAULT_VERSION = "deepfaune-yolo8s"

    def __init__(self, weights=None, device="cpu"):
        super().__init__(weights=weights, device=device)


class _Unsupported:
    ENGINE_HINT = ""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is not part of the PytorchWildlife 2.0 transition API. "
            f"{self.ENGINE_HINT}"
        )


class HerdNet(_Unsupported):
    ENGINE_HINT = (
        "Use sparrow_engine.detect(..., model='HerdNet_General_Dataset_2022'); it returns "
        "points, not boxes."
    )


class OWL(_Unsupported):
    ENGINE_HINT = "Use sparrow_engine.detect(..., model='OWL')."
