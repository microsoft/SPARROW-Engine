"""PytorchWildlife 1.x post-processing helpers that work on 2.0 results.

The JSON writers keep the 1.x file format. The image writers draw with Pillow instead
of supervision/OpenCV, so box styling differs slightly from 1.x.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw

__all__ = [
    "save_detection_images",
    "save_crop_images",
    "save_detection_json",
    "save_detection_classification_json",
]


def _as_list(results):
    return results if isinstance(results, list) else [results]


def _target(entry, output_dir, input_dir, name):
    if input_dir:
        rel_dir = os.path.dirname(os.path.relpath(entry["img_id"], input_dir))
        out_dir = os.path.join(output_dir, rel_dir)
    else:
        out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def save_detection_images(results, output_dir, input_dir=None, overwrite=False):
    """Save copies of the input images with boxes and labels drawn on them."""
    os.makedirs(output_dir, exist_ok=True)
    for entry in _as_list(results):
        path = _target(entry, output_dir, input_dir, os.path.basename(entry["img_id"]))
        if os.path.exists(path) and not overwrite:
            continue
        with Image.open(entry["img_id"]) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            width = max(2, round(max(im.size) / 400))
            boxes = zip(entry["detections"].xyxy, entry["labels"], strict=True)
            for (x1, y1, x2, y2), label in boxes:
                draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=width)
                draw.text((x1 + width, y1 + width), label, fill=(255, 255, 255))
            im.save(path)


def save_crop_images(results, output_dir, input_dir=None, overwrite=False):
    """Save one image per detection box, named ``<class_id>_<index>_<image name>``."""
    os.makedirs(output_dir, exist_ok=True)
    for entry in _as_list(results):
        dets = entry["detections"]
        if len(dets) == 0:
            continue
        with Image.open(entry["img_id"]) as im:
            im = im.convert("RGB")
            for i, (xyxy, cat) in enumerate(zip(dets.xyxy, dets.class_id, strict=True)):
                name = f"{int(cat)}_{i}_{os.path.basename(entry['img_id'])}"
                path = _target(entry, output_dir, input_dir, name)
                if os.path.exists(path) and not overwrite:
                    continue
                im.crop(tuple(int(round(v)) for v in xyxy)).save(path)


def _strip_path(img_id, exclude_file_path):
    img_id = str(img_id)
    return img_id.replace(exclude_file_path + os.sep, "") if exclude_file_path else img_id


def save_detection_json(det_results, output_dir, categories=None, exclude_category_ids=(),
                        exclude_file_path=None):
    """Write detection results to the 1.x JSON format. ``output_dir`` is the file path."""
    json_results = {"annotations": [], "categories": categories}
    for det in det_results:
        category = np.asarray(det["detections"].class_id)
        keep = ~np.isin(category, list(exclude_category_ids))
        json_results["annotations"].append(
            {
                "img_id": _strip_path(det["img_id"], exclude_file_path),
                "bbox": np.asarray(det["detections"].xyxy).astype(int)[keep].tolist(),
                "category": category[keep].tolist(),
                "confidence": np.asarray(det["detections"].confidence)[keep].tolist(),
            }
        )
    with open(output_dir, "w") as f:
        json.dump(json_results, f, indent=4)


def save_detection_classification_json(det_results, clf_results, output_path, det_categories=None,
                                       clf_categories=None, exclude_file_path=None):
    """Write detections plus per-box classifications to the 1.x JSON format.

    ``clf_results`` must be in the order produced by
    ``batch_image_classification(det_results=...)``.
    """
    json_results = {"annotations": [], "det_categories": det_categories,
                    "clf_categories": clf_categories}
    counter = 0
    for det in det_results:
        clf_cats, clf_conf = [], []
        while counter < len(clf_results) and clf_results[counter]["img_id"] == det["img_id"]:
            clf_cats.append(int(clf_results[counter]["class_id"]))
            clf_conf.append(float(clf_results[counter]["confidence"]))
            counter += 1
        json_results["annotations"].append(
            {
                "img_id": _strip_path(det["img_id"], exclude_file_path),
                "bbox": np.asarray(det["detections"].xyxy).astype(int).tolist(),
                "det_category": [int(x) for x in det["detections"].class_id],
                "det_confidence": [float(x) for x in det["detections"].confidence],
                "clf_category": clf_cats,
                "clf_confidence": clf_conf,
            }
        )
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=4)
