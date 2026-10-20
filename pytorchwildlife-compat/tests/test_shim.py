import json
import os
import types
import warnings

import numpy as np
import pytest
import sparrow_engine as spe
from PIL import Image

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import PytorchWildlife
    from PytorchWildlife import utils as pw_utils
    from PytorchWildlife.models import classification as pw_classification
    from PytorchWildlife.models import detection as pw_detection

NS = types.SimpleNamespace


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    """Replace the engine with a fake: one animal box per image, two classifier labels."""
    models = tmp_path / "models"
    for model_id, labels in {"MDV6-yolov9-c": "animal,0\nperson,1\ncar,2\n",
                             "AI4G-Amazon-V2": "Dasyprocta,0\nBos,1\n"}.items():
        (models / model_id).mkdir(parents=True)
        (models / model_id / "manifest.toml").write_text("")
        (models / model_id / "labels.txt").write_text(labels)
    monkeypatch.setattr(spe, "download_model", lambda m, **k: models / m, raising=False)

    def detect(paths, model, threshold):
        out = []
        for p in paths:
            with Image.open(p) as im:
                size = im.size
            box = NS(x_min=0.25, y_min=0.5, x_max=0.75, y_max=1.0)
            out.append(NS(path=p, image_size=size,
                          detections=[NS(bbox=box, confidence=0.9, label_id=0, label="animal")]))
        return out[::-1]  # results out of order on purpose

    def classify(paths, model, top_k):
        return [NS(path=p, classifications=[NS(label="Bos", confidence=0.8),
                                            NS(label="Dasyprocta", confidence=0.2)])
                for p in paths]

    monkeypatch.setattr(spe, "detect", detect)
    monkeypatch.setattr(spe, "classify", classify)
    images = tmp_path / "images" / "sub"
    images.mkdir(parents=True)
    for name in ("b.jpg", "a.png"):
        Image.new("RGB", (200, 100)).save(images / name)
    (images / "notes.txt").write_text("x")
    return tmp_path


def test_import_warns_and_versions():
    assert PytorchWildlife.__version__ == "2.0.0"


def test_single_detection_matches_1x_shape(fake_engine):
    det = pw_detection.MegaDetectorV6()
    path = str(fake_engine / "images" / "sub" / "a.png")
    r = det.single_image_detection(path)
    assert r["img_id"] == path
    np.testing.assert_allclose(r["detections"].xyxy, [[50, 50, 150, 100]])
    assert list(r["detections"].class_id) == [0]
    assert r["labels"] == ["animal 0.90"]
    np.testing.assert_allclose(r["normalized_coords"], [[0.25, 0.5, 0.75, 1.0]])
    assert len(list(r["detections"])[0]) == 6


def test_ndarray_input_uses_array_size(fake_engine):
    det = pw_detection.MegaDetectorV6()
    r = det.single_image_detection(np.zeros((40, 80, 3), np.uint8), img_path="cam1/x.jpg")
    assert r["img_id"] == "cam1/x.jpg"
    np.testing.assert_allclose(r["detections"].xyxy, [[20, 20, 60, 40]])
    batch = det.batch_image_detection([np.zeros((10, 10, 3), np.uint8)] * 2)
    assert [b["img_id"] for b in batch] == ["0", "1"]


def test_batch_folder_is_recursive_sorted_and_filters_extensions(fake_engine):
    det = pw_detection.MegaDetectorV6()
    results = det.batch_image_detection(str(fake_engine / "images"))
    assert [os.path.basename(r["img_id"]) for r in results] == ["a.png", "b.jpg"]


def test_unsupported_inputs_raise(fake_engine):
    with pytest.raises(NotImplementedError):
        pw_detection.MegaDetectorV6(weights="md.pt")
    with pytest.raises(ValueError):
        pw_detection.MegaDetectorV6(version="MDV6-nope")
    with pytest.raises(NotImplementedError):
        pw_detection.HerdNet()
    with pytest.raises(NotImplementedError):
        pw_classification.AI4GOpossum()
    with pytest.raises(ValueError):
        pw_classification.AI4GAmazonRainforest(version="v1")


def test_classification_and_crop_pipeline_json(fake_engine, tmp_path):
    det = pw_detection.MegaDetectorV6()
    clf = pw_classification.AI4GAmazonRainforest()
    single = clf.single_image_classification(str(fake_engine / "images" / "sub" / "a.png"))
    assert (single["prediction"], single["class_id"]) == ("Bos", 1)
    assert single["all_confidences"] == [["Dasyprocta", 0.2], ["Bos", 0.8]]

    dets = det.batch_image_detection(str(fake_engine / "images"))
    crops = clf.batch_image_classification(det_results=dets)
    assert [c["img_id"] for c in crops] == [d["img_id"] for d in dets]

    out = tmp_path / "pipeline.json"
    pw_utils.save_detection_classification_json(dets, crops, str(out),
                                                det_categories=det.CLASS_NAMES,
                                                clf_categories=clf.CLASS_NAMES)
    ann = json.loads(out.read_text())["annotations"]
    assert ann[0]["bbox"] == [[50, 50, 150, 100]]
    assert ann[0]["clf_category"] == [1]

    out = tmp_path / "det.json"
    pw_utils.save_detection_json(dets, str(out), exclude_category_ids=[0])
    assert json.loads(out.read_text())["annotations"][0]["category"] == []


def test_image_writers(fake_engine, tmp_path):
    det = pw_detection.MegaDetectorV6()
    root = str(fake_engine / "images")
    dets = det.batch_image_detection(root)
    pw_utils.save_detection_images(dets, str(tmp_path / "vis"), input_dir=root)
    pw_utils.save_crop_images(dets, str(tmp_path / "crops"))
    assert (tmp_path / "vis" / "sub" / "a.png").exists()
    with Image.open(tmp_path / "crops" / "0_0_b.jpg") as im:
        assert im.size == (100, 50)


LIVE = os.environ.get("PW_SHIM_LIVE_IMAGES")


@pytest.mark.skipif(not LIVE, reason="set PW_SHIM_LIVE_IMAGES to a folder of images")
def test_live_detection_and_classification():
    det = pw_detection.MegaDetectorV6(version="MDV6-yolov10-c")
    results = det.batch_image_detection(LIVE)
    assert results and any(len(r["detections"]) for r in results)
    clf = pw_classification.AI4GAmazonRainforest()
    crops = clf.batch_image_classification(det_results=results)
    assert crops and all(c["prediction"] in clf.CLASS_NAMES.values() for c in crops)
