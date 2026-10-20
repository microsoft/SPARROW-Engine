"""PytorchWildlife 1.x classification classes, running on sparrow_engine.

Results keep the 1.x dict shape:
``{"img_id", "prediction", "class_id", "confidence", "all_confidences"}``.
"""

from __future__ import annotations

import os

import numpy as np
import sparrow_engine as spe
from PIL import Image

from .. import _engine as _e

__all__ = [
    "AI4GAmazonRainforest",
    "AI4GSnapshotSerengeti",
    "DeepfauneClassifier",
    "DFNE",
    "AI4GOpossum",
    "CustomWeights",
]

_ALL_CLASSES = 100_000


class _SparrowClassifier:
    """Base class: a 1.x-style classifier backed by one Sparrow zoo model."""

    MODEL_ID: str = ""

    def __init__(self, weights=None, device="cpu", **_ignored):
        if weights is not None:
            raise NotImplementedError(
                "PytorchWildlife 2.0 does not load PyTorch weight files. Export the model to "
                "ONNX and run it with sparrow_engine, or use a Sparrow model zoo model."
            )
        self.device = device
        self.model_id = self.MODEL_ID
        self.CLASS_NAMES = _e.read_labels(_e.ensure_model(self.model_id))
        self._name_to_id = {name: idx for idx, name in self.CLASS_NAMES.items()}

    def _to_result(self, res, img_id, id_strip) -> dict:
        scores = {c.label: float(c.confidence) for c in res.classifications}
        if self.CLASS_NAMES:
            names = [name for _, name in sorted(self.CLASS_NAMES.items())]
            all_conf = [[name, scores.get(name, 0.0)] for name in names]
        else:
            all_conf = [[name, conf] for name, conf in scores.items()]
        top = max(res.classifications, key=lambda c: c.confidence) if res.classifications else None
        prediction = top.label if top else None
        return {
            "img_id": _e.strip_id(img_id, id_strip),
            "prediction": prediction,
            "class_id": self._name_to_id.get(prediction, -1),
            "confidence": float(top.confidence) if top else 0.0,
            "all_confidences": all_conf,
        }

    def _classify_paths(self, paths) -> dict:
        if not paths:
            return {}
        results = spe.classify(list(paths), model=self.model_id, top_k=_ALL_CLASSES)
        return {os.path.realpath(r.path): r for r in results}

    def single_image_classification(self, img, img_id=None, id_strip=None):
        """Classify one image (a path or an RGB ndarray)."""
        if isinstance(img, (str, os.PathLike)):
            path = str(img)
            res = self._classify_paths([path]).get(os.path.realpath(path))
            if res is None:
                raise RuntimeError(f"sparrow_engine could not read image {path}")
            return self._to_result(res, img_id if img_id is not None else path, id_strip)
        with _e.temp_dir() as tmp:
            (path,) = _e.write_temp_images([np.asarray(img)], tmp)
            res = self._classify_paths([path])[os.path.realpath(path)]
        return self._to_result(res, img_id, id_strip)

    def batch_image_classification(self, data_path=None, det_results=None, id_strip=None,
                                   batch_size=32, num_workers=0, **kwargs):
        """Classify a folder of images, or the animal boxes (class 0) in ``det_results``.

        With ``det_results`` there is one result per animal box, in box order, and
        ``img_id`` is the source image. ``batch_size`` and ``num_workers`` are accepted for
        compatibility.
        """
        if data_path:
            paths = _e.list_images(str(data_path))
            by_path = self._classify_paths(paths)
            return [
                self._to_result(by_path[os.path.realpath(p)], p, id_strip)
                for p in paths if os.path.realpath(p) in by_path
            ]
        if det_results:
            return self._classify_crops(det_results, id_strip)
        raise ValueError("Need data for inference: pass data_path or det_results.")

    def _classify_crops(self, det_results, id_strip, animal_cls_id=0):
        crops = []
        with _e.temp_dir() as tmp:
            for det in det_results:
                dets = det["detections"]
                boxes = [
                    xyxy
                    for xyxy, cid in zip(dets.xyxy, dets.class_id, strict=True)
                    if int(cid) == animal_cls_id
                ]
                if not boxes:
                    continue
                with Image.open(det["img_id"]) as im:
                    im = im.convert("RGB")
                    for xyxy in boxes:
                        path = os.path.join(tmp, f"{len(crops):08d}.png")
                        im.crop(tuple(int(round(v)) for v in xyxy)).save(path)
                        crops.append((path, det["img_id"]))
            by_path = self._classify_paths([p for p, _ in crops])
            return [
                self._to_result(by_path[os.path.realpath(p)], src, id_strip)
                for p, src in crops if os.path.realpath(p) in by_path
            ]


class AI4GAmazonRainforest(_SparrowClassifier):
    MODEL_ID = "AI4G-Amazon-V2"

    def __init__(self, weights=None, device="cpu", pretrained=True, version="v2"):
        if version != "v2":
            raise ValueError(
                "Only version='v2' of AI4GAmazonRainforest is in the Sparrow model zoo."
            )
        super().__init__(weights=weights, device=device)


class AI4GSnapshotSerengeti(_SparrowClassifier):
    MODEL_ID = "AI4G-Serengeti"

    def __init__(self, weights=None, device="cpu", pretrained=True):
        super().__init__(weights=weights, device=device)


class DeepfauneClassifier(_SparrowClassifier):
    MODEL_ID = "Deepfaune-Europe"

    def __init__(self, weights=None, device="cpu", transform=None, class_name_lang="en"):
        super().__init__(weights=weights, device=device)


class DFNE(_SparrowClassifier):
    MODEL_ID = "Deepfaune-New-England"

    def __init__(self, weights=None, device="cpu", transform=None):
        super().__init__(weights=weights, device=device)


class _Unsupported:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is not in the Sparrow model zoo and is not part of "
            "PytorchWildlife 2.0. Fine-tuning moved to "
            "https://github.com/microsoft/MegaDetector-Classifier."
        )


class AI4GOpossum(_Unsupported):
    pass


class CustomWeights(_Unsupported):
    pass
