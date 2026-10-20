# PytorchWildlife 2.0 (transition package)

PytorchWildlife has moved to the Sparrow ecosystem:

- Inference runs on **[sparrow-engine](https://github.com/microsoft/SPARROW-Engine)**
  (ONNX Runtime, no PyTorch), with models from the Sparrow model zoo.
- Notebooks and examples live in **[SPARROW-Playground](https://github.com/microsoft/SPARROW-Playground)**.
- Fine-tuning moved to **[MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier)**.

`PytorchWildlife` 2.0 keeps the most-used 1.x inference calls working on top of
sparrow-engine so existing scripts keep running while you migrate. It emits a
`DeprecationWarning` on import. New code should call `sparrow_engine` directly.

## Install

```bash
pip install --upgrade PytorchWildlife   # pulls in sparrow-engine; no PyTorch
```

Models download from the Sparrow model zoo on first use into
`~/.sparrow-engine/models` (override with `SPARROW_ENGINE_MODEL_DIR`).

## What still works

```python
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils

detector = pw_detection.MegaDetectorV6(version="MDV6-yolov10-c")
result = detector.single_image_detection("image.jpg")
results = detector.batch_image_detection("images/", det_conf_thres=0.2)

classifier = pw_classification.AI4GAmazonRainforest()
crops = classifier.batch_image_classification(det_results=results)

pw_utils.save_detection_json(results, "detections.json",
                             categories=detector.CLASS_NAMES)
pw_utils.save_detection_classification_json(results, crops, "pipeline.json",
                                            det_categories=detector.CLASS_NAMES,
                                            clf_categories=classifier.CLASS_NAMES)
```

Results keep the 1.x shape: `img_id`, `detections` (`.xyxy` in pixels, `.confidence`,
`.class_id`), `labels` and `normalized_coords` for detection; `prediction`, `class_id`,
`confidence` and `all_confidences` for classification. `detections` is a
`supervision.Detections` when supervision is installed (`pip install
PytorchWildlife[supervision]`), otherwise a small object with the same fields.

| 1.x class | Sparrow zoo model |
|---|---|
| `MegaDetectorV6(version=...)` | `MDV6-yolov9-c`, `MDV6-yolov9-e`, `MDV6-yolov10-c`, `MDV6-yolov10-e`, `MDV6-rtdetr-c` |
| `MegaDetectorV6MIT(version=...)` | `MDV6-mit-yolov9-c`, `MDV6-mit-yolov9-e` |
| `MegaDetectorV6Apache(version=...)` | `MDV6-apa-rtdetr-c`, `MDV6-apa-rtdetr-e` |
| `MegaDetectorV5(version="a"/"b")` | `MDV5a`, `MDV5b` (install manually if the zoo download fails) |
| `DeepfauneDetector()` | `deepfaune-yolo8s` |
| `AI4GAmazonRainforest(version="v2")` | `AI4G-Amazon-V2` |
| `AI4GSnapshotSerengeti()` | `AI4G-Serengeti` |
| `DeepfauneClassifier()` | `Deepfaune-Europe` |
| `DFNE()` | `Deepfaune-New-England` |

Utilities: `save_detection_json`, `save_detection_classification_json`,
`save_detection_images` and `save_crop_images`.

## What changed

- `weights=` (PyTorch `.pth` files) raises `NotImplementedError`; use zoo models or export
  to ONNX and call sparrow-engine.
- `device=`, `batch_size=` and `num_workers=` are accepted and ignored; sparrow-engine picks
  the execution provider and batching.
- Removed: `HerdNet`, `OWL` (call `sparrow_engine.detect` with `HerdNet_General_Dataset_2022`
  or `OWL`), `AI4GOpossum`, `AI4GAmazonRainforest(version="v1")`, `CustomWeights`, the
  Timelapse/dots JSON writers, `detection_folder_separation`, video helpers, data
  loaders and all training code.
- Image writers draw with Pillow, so box styling differs from 1.x.
- Scores come from the ONNX exports and can differ slightly from the 1.x PyTorch models.

## Moving to sparrow-engine

| PytorchWildlife 2.0 | sparrow-engine |
|---|---|
| `MegaDetectorV6(...).batch_image_detection(folder)` | `sparrow_engine.detect(folder, model="MDV6-yolov9-c", recursive=True)` |
| `classifier.batch_image_classification(data_path=folder)` | `sparrow_engine.classify(folder, model="AI4G-Amazon-V2")` |
| detect, then `batch_image_classification(det_results=...)` | `sparrow_engine.pipeline(folder, detector=..., classifier=...)` |
| `save_detection_json(...)` | `sparrow_engine.export(results, format="json", output=...)` |

## Maintainers

- The `sparrow-engine==X.Y.Z` pin is co-bumped with every sparrow-engine release;
  `.github/workflows/release.yml` fails the release when it drifts.
- Tests: `pip install -e .[test] && pytest tests` (live tests skip unless
  `PW_SHIM_LIVE_IMAGES` points at a folder of camera-trap images and the models are
  installed).
- Publishing to PyPI needs the owner of the existing `PytorchWildlife` project
  (1.x releases up to 1.3.0); 2.0.0 must be uploaded under that project so
  `pip install --upgrade PytorchWildlife` picks it up.
