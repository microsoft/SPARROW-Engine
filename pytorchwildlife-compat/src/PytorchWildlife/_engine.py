"""Shared helpers between the PytorchWildlife 1.x API and sparrow_engine."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import sparrow_engine as spe
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

ZOO_URL = "https://github.com/microsoft/SPARROW-Playground"


class Detections:
    """Minimal stand-in for ``supervision.Detections`` when supervision is not installed.

    Exposes ``xyxy`` (pixels), ``confidence`` and ``class_id`` arrays, ``len()``, and
    iteration as ``(xyxy, mask, confidence, class_id, tracker_id, data)`` tuples, which is
    what PytorchWildlife 1.x code reads.
    """

    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.class_id = np.asarray(class_id, dtype=int)
        self.mask = None
        self.tracker_id = None
        self.data = {}

    def __len__(self):
        return len(self.xyxy)

    def __iter__(self):
        for i in range(len(self)):
            yield self.xyxy[i], None, self.confidence[i], self.class_id[i], None, {}

    def __repr__(self):
        return f"Detections(n={len(self)})"


def make_detections(xyxy, confidence, class_id):
    """Return ``supervision.Detections`` if supervision is installed, else :class:`Detections`."""
    try:
        import supervision as sv
    except ImportError:
        return Detections(xyxy, confidence, class_id)
    return sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
        confidence=np.asarray(confidence, dtype=np.float32),
        class_id=np.asarray(class_id, dtype=int),
    )


def model_dir(model_id: str) -> Path:
    default = Path.home() / ".sparrow-engine" / "models"
    root = os.environ.get("SPARROW_ENGINE_MODEL_DIR", str(default))
    return Path(root) / model_id


def ensure_model(model_id: str) -> Path:
    """Install ``model_id`` from the Sparrow model zoo if needed and return its directory."""
    download = getattr(spe, "download_model", None)
    if download is not None:
        try:
            return Path(download(model_id, quiet=True))
        except Exception as exc:  # noqa: BLE001 - re-raised with migration guidance
            local = model_dir(model_id)
            if (local / "manifest.toml").exists():
                return local
            raise RuntimeError(
                f"Could not install model '{model_id}' from the Sparrow model zoo: {exc}"
            ) from exc
    local = model_dir(model_id)
    if not (local / "manifest.toml").exists():
        raise RuntimeError(
            f"Model '{model_id}' is not installed in {local.parent}. Upgrade sparrow-engine "
            f"for automatic downloads, or install the model first (see {ZOO_URL})."
        )
    return local


def read_labels(path: Path) -> dict[int, str]:
    """Parse a zoo ``labels.txt`` (``name,index`` per line) into ``{index: name}``."""
    labels: dict[int, str] = {}
    labels_file = path / "labels.txt"
    if not labels_file.exists():
        return labels
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, idx = line.rpartition(",")
        if name and idx.strip().isdigit():
            labels[int(idx)] = name
        else:
            labels[len(labels)] = line
    return labels


def list_images(folder: str) -> list[str]:
    """Walk ``folder`` recursively like PytorchWildlife 1.x did and return sorted image paths."""
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def write_temp_images(arrays, tmpdir: str) -> list[str]:
    """Save RGB ndarrays as PNG files in ``tmpdir`` and return their paths."""
    paths = []
    for i, arr in enumerate(arrays):
        path = os.path.join(tmpdir, f"{i:08d}.png")
        Image.fromarray(np.asarray(arr).astype(np.uint8)).save(path)
        paths.append(path)
    return paths


def temp_dir():
    return tempfile.TemporaryDirectory(prefix="pytorchwildlife-")


def strip_id(img_id, id_strip):
    return str(img_id).strip(id_strip)
