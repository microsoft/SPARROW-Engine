# Sparrow Engine — Model Zoo Catalogue

> Auto-generated from [`sparrow-engine/scripts/catalog.toml`](../sparrow-engine/scripts/catalog.toml) — the single source of truth. **61 models**, Zenodo record [`21273206`](https://zenodo.org/records/21273206) (v0.19.0, concept DOI [`10.5281/zenodo.20348978`](https://doi.org/10.5281/zenodo.20348978)).

This is a **multi-license bundle**: every model keeps its own upstream license. Download with `sparrow-engine/scripts/download_models.sh` (fetches ONNX models by default; TFLite / cascade artifacts on demand).

## Reading this catalogue

Models are grouped by **domain** (camera trap, acoustics, overhead, general) and **task** (detector, classifier, encoder, cascade). Each row carries:

- **Display name / ID** — human-readable label and the stable `id` the engine and `download_models.sh` resolve models by.
- **Family** — model lineage (e.g. MegaDetector, SpeciesNet, DeepFaune, AddaxAI, DCLDE-orca).
- **Format / version** — artifact type (`onnx` desktop, `tflite` mobile, `cascade` descriptor) with version / flavor.
- **Behavior** — what the model emits: a detector that gates on a class (e.g. animal) versus one that outputs species directly, a classifier, an encoder, or a cascade pipeline.
- **Geography / locality** — global, foundational, or the regions / locality a regional model targets.
- **Developer / owner** — the party that built the model and, where different, the data owner.
- **AI4G relationship** — first-party (built by Microsoft AI for Good Lab) versus third-party (external model onboarded into the zoo).

## Licensing at a glance

- **`Commercial use`**: ✅ = the license permits commercial use; ❌ = non-commercial only (CC-BY-NC-* licenses). This flag mirrors the machine-readable `commercial_use` field in the catalog.
- **Copyleft**: AGPL-3.0 / GPL-3.0 models *permit* commercial use but impose source-disclosure / copyleft obligations. Closed-source commercial use of YOLO-based (Ultralytics) detectors additionally requires an [Ultralytics Enterprise License](https://www.ultralytics.com/license).
- **No-derivatives**: `tropicam-ai` is CC-BY-NC-ND-4.0 — the no-derivatives clause may restrict redistribution of the converted ONNX; treat as non-commercial + review before redistribution.
- Not legal advice — confirm the upstream terms in each model's reference before commercial deployment.

## Camera Trap — Detector

| Display name | ID | Family | Format / version | Behavior | Geography / locality | Developer / owner | AI4G relationship | License | Commercial use |
|---|---|---|---|---|---|---|---|---|---|
| MDV5a | `MDV5a` | MegaDetector | onnx · v5a | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MDV6 yolov10 C | `MDV6-yolov10-c` | MegaDetector | onnx · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MDV6 yolov10 C Tflite | `MDV6-yolov10-c-tflite` | MegaDetector | tflite-fp16 · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MDV6 yolov10 E | `MDV6-yolov10-e` | MegaDetector | onnx · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| Deepfaune yolo8s | `deepfaune-yolo8s` | DeepFaune | onnx | Detector · gates on animal | Global | DeepFaune consortium (CNRS and OFB) | Third-party | AGPL-3.0 AND CC-BY-SA-4.0 | ✅ |
| MD European Mammals | `european_mammals` | MegaDetector | onnx | Detector · direct species output | Europe | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MD North American Mammals | `north_american_mammals` | MegaDetector | onnx | Detector · direct species output | North America | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MD Sub-Saharan Mammals | `sub_saharan` | MegaDetector | onnx | Detector · direct species output | Sub-Saharan Africa | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MDV5b | `MDV5b` | MegaDetector | onnx · v5b | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MD1000 Redwood | `MD1000-redwood` | MegaDetector | onnx · v1000-redwood | Detector · gates on animal | Global | Dan Morris / MegaDetector project | Third-party | AGPL-3.0 | ✅ |
| MD1000 Spruce | `MD1000-spruce` | MegaDetector | onnx · v1000-spruce | Detector · gates on animal | Global | Dan Morris / MegaDetector project | Third-party | AGPL-3.0 | ✅ |
| MD1000 Larch | `MD1000-larch` | MegaDetector | onnx · v1000-larch | Detector · gates on animal | Global | Dan Morris / MegaDetector project | Third-party | AGPL-3.0 | ✅ |
| MD1000 Cedar | `MD1000-cedar` | MegaDetector | onnx · v1000-cedar | Detector · gates on animal | Global | Dan Morris / MegaDetector project | Third-party | GPL-3.0 | ✅ |
| MD1000 Sorrel | `MD1000-sorrel` | MegaDetector | onnx · v1000-sorrel | Detector · gates on animal | Global | Dan Morris / MegaDetector project | Third-party | AGPL-3.0 | ✅ |
| MDV6 yolov9 C | `MDV6-yolov9-c` | MegaDetector | onnx · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MDV6 yolov9 E | `MDV6-yolov9-e` | MegaDetector | onnx · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MDV6 Rtdetr C | `MDV6-rtdetr-c` | MegaDetector | onnx · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | AGPL-3.0 | ✅ |
| MDV6 Mit yolov9 C | `MDV6-mit-yolov9-c` | MegaDetector | onnx · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| MDV6 Mit yolov9 E | `MDV6-mit-yolov9-e` | MegaDetector | onnx · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| MDV6 Apa Rtdetr C | `MDV6-apa-rtdetr-c` | MegaDetector | onnx · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | Apache-2.0 | ✅ |
| MDV6 Apa Rtdetr E | `MDV6-apa-rtdetr-e` | MegaDetector | onnx · v6 | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | Apache-2.0 | ✅ |

## Camera Trap — Classifier

| Display name | ID | Family | Format / version | Behavior | Geography / locality | Developer / owner | AI4G relationship | License | Commercial use |
|---|---|---|---|---|---|---|---|---|---|
| AI4G Amazon V2 | `AI4G-Amazon-V2` | AI4G | onnx | Classifier | South America — Amazon basin | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| AI4G Serengeti | `AI4G-Serengeti` | AI4G | onnx | Classifier | East Africa — Serengeti, Tanzania | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| Deepfaune Europe | `Deepfaune-Europe` | DeepFaune | onnx | Classifier | Europe | DeepFaune consortium (CNRS and OFB) | Third-party | CC-BY-SA-4.0 | ✅ |
| Deepfaune New England | `Deepfaune-New-England` | DeepFaune | onnx | Classifier | North America — New England, USA | Clarfeld et al. / USGS | Third-party | CC0-1.0 | ✅ |
| SpeciesNet Crop | `SpeciesNet-Crop` | SpeciesNet | onnx | Classifier | Global | Google Research / SpeciesNet | Third-party | Apache-2.0 | ✅ |
| southwest-usa-v3-SDZWA | `southwest-usa-v3` | AddaxAI | onnx | Classifier | North America — Southwest USA | Kyra Swanson / San Diego Zoo Wildlife Alliance | Third-party | MIT | ✅ |
| peruvian-andes-SDZWA | `peruvian-andes` | AddaxAI | onnx | Classifier | South America — Peruvian Andes | Kyra Swanson / San Diego Zoo Wildlife Alliance | Third-party | MIT | ✅ |
| sub-saharan-drylands-Addax | `sub-saharan-drylands` | AddaxAI | onnx | Classifier | Sub-Saharan Africa — eastern and southern drylands | Addax Data Science | Third-party | CC-BY-NC-SA-4.0 | ❌ |
| Manas Panthera | `manas-panthera` | AddaxAI | onnx | Classifier | Central Asia — Kyrgyzstan, Tian Shan | Hex Data<br>owner: OSI-Panthera | Third-party | CC-BY-NC-SA-4.0 | ❌ |
| gifu-japan-GifuUniversity | `gifu-japan` | AddaxAI | onnx | Classifier | East Asia — Gifu Prefecture, Japan | Gifu University<br>owner: Masaki Ando | Third-party | MIT | ✅ |
| hawaii-puaa-Addax | `hawaii-puaa` | AddaxAI, SpeciesNet | onnx | Classifier | North America — Hawaii, USA | Addax Data Science<br>owner: USDA Forest Service Institute of Pacific Islands Forestry and The Nature Conservancy | Third-party | CC-BY-NC-4.0 | ❌ |
| central-india-Addax | `central-india` | AddaxAI, SpeciesNet | onnx | Classifier | South Asia — Central India | Addax Data Science<br>owner: Wildlife Conservation Trust, India | Third-party | CC-BY-NC-4.0 | ❌ |
| top-end-savanna-Addax | `top-end-savanna` | AddaxAI, SpeciesNet | onnx | Classifier | Australia — Top End, Northern Territory | Addax Data Science<br>owner: Charles Darwin University, Territory Natural Resource Management, and Warddeken Land Management | Third-party | CC-BY-NC-4.0 | ❌ |
| parks-victoria-Addax | `parks-victoria` | AddaxAI, SpeciesNet | onnx | Classifier | Australia — Victoria | Addax Data Science<br>owner: Parks Victoria | Third-party | Apache-2.0 | ✅ |
| sw-borderlands-Addax | `sw-borderlands` | AddaxAI, SpeciesNet | onnx | Classifier | North America — USA-Mexico borderlands | Addax Data Science<br>owner: Tohono O'odham Nation and University of Arizona | Third-party | Apache-2.0 | ✅ |
| ahdrift-OSU-ColumbusZoo-Addax | `ahdrift` | AddaxAI, SpeciesNet | onnx | Classifier | North America — Midwest USA | Ohio State University, Columbus Zoo and Aquarium, and Addax Data Science | Third-party | Apache-2.0 | ✅ |
| deep-forest-vision-MNHN-OFVI | `deep-forest-vision` | AddaxAI | onnx | Classifier | Central Africa — African tropical forests | Hugo Magaldi / One Forest Vision initiative<br>owner: MNHN and One Forest Vision partners | Third-party | CC-BY-NC-SA-4.0 | ❌ |
| awc135-AWC | `awc135` | AddaxAI | onnx | Classifier | Australia | Australian Wildlife Conservancy | Third-party | CC-BY-NC-SA-4.0 | ❌ |
| namibian-Addax | `namibian` | AddaxAI | onnx | Classifier | Southern Africa — Namib Desert, Namibia | Addax Data Science<br>owner: Desert Lion Conservation and Smart Parks | Third-party | CC-BY-NC-SA-4.0 | ❌ |
| iran-Addax | `iran` | AddaxAI | onnx | Classifier | West Asia — Iran | Addax Data Science<br>owner: Iranian Cheetah Society | Third-party | CC-BY-NC-SA-4.0 | ❌ |
| nz-invasives-Addax | `nz-invasives` | AddaxAI | onnx | Classifier | New Zealand | Addax Data Science<br>owner: New Zealand Department of Conservation | Third-party | CC-BY-NC-SA-4.0 | ❌ |
| queensland-WildObs | `queensland` | AddaxAI, SpeciesNet | onnx | Classifier | Australia — Queensland Wet Tropics | Prakash Palanivelu Rajmohan and Renuka Sharma / WildObs | Third-party | CC-BY-4.0 | ✅ |
| nz-species-wekaResearch | `nz-species` | AddaxAI | onnx | Classifier | New Zealand | wekaResearch / Olly Powell<br>owner: New Zealand Department of Conservation | Third-party | CC-BY-NC-SA-4.0 | ❌ |
| tropicam-ai-MNCN-CSIC | `tropicam-ai` | AddaxAI | onnx | Classifier | Neotropics — Brazil, Peru, French Guiana, and Costa Rica | Andrea Zampetti / MNCN-CSIC | Third-party | CC-BY-NC-ND-4.0 | ❌ |
| Terai Nepal | `terai-nepal` | AddaxAI, MEWC | onnx | Classifier | South Asia — Terai lowlands, Nepal | Alexander Merdian-Tarko / TeraiNet | Third-party | MIT | ✅ |
| tasmanian-vertebrates-MEWC | `tasmanian-vertebrates` | AddaxAI, MEWC | onnx | Classifier | Australia — Tasmania | Barry Brook / MEWC / University of Tasmania | Third-party | CC-BY-4.0 | ✅ |
| peruvian-amazon-SDZWA | `peruvian-amazon-sdzwa` | AddaxAI | onnx | Classifier | South America — Peruvian Amazon | Mathias Tobler / San Diego Zoo Wildlife Alliance | Third-party | MIT | ✅ |

## Acoustics — Detector

| Display name | ID | Family | Format / version | Behavior | Geography / locality | Developer / owner | AI4G relationship | License | Commercial use |
|---|---|---|---|---|---|---|---|---|---|
| Md Audiobirds v1 | `md-audiobirds-v1` | — | onnx | Detector · gates on bird | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| Orca Detector dclde2026 v3 | `orca-detector-dclde2026-v3` | DCLDE-orca | onnx · v3 | Detector · gates on Orca | North Pacific — Pacific Northwest and Salish Sea | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| Orca Detector v3 fp16 Tflite | `orca-detector-v3-fp16-tflite` | DCLDE-orca | tflite-fp16 · v3 | Detector · gates on Orca | North Pacific — Pacific Northwest and Salish Sea | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| Orca Detector v3 int8 Tflite | `orca-detector-v3-int8-tflite` | DCLDE-orca | tflite-int8 · v3 | Detector · gates on Orca | North Pacific — Pacific Northwest and Salish Sea | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |

## Acoustics — Classifier

| Display name | ID | Family | Format / version | Behavior | Geography / locality | Developer / owner | AI4G relationship | License | Commercial use |
|---|---|---|---|---|---|---|---|---|---|
| Orca Ecotype dclde2026 v1 | `orca-ecotype-dclde2026-v1` | DCLDE-orca | onnx · v1 | Classifier | North Pacific — Pacific Northwest and Salish Sea | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| Orca Ecotype Melinput fp16 Tflite | `orca-ecotype-melinput-fp16-tflite` | DCLDE-orca | tflite-fp16 | Classifier | North Pacific — Pacific Northwest and Salish Sea | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| Orca Ecotype Melinput int8 Tflite | `orca-ecotype-melinput-int8-tflite` | DCLDE-orca | tflite-int8 | Classifier | North Pacific — Pacific Northwest and Salish Sea | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |
| Perch v2 | `perch-v2` | — | onnx | Classifier | Global | Google Research (Hamer et al.) | Third-party | Apache-2.0 | ✅ |

## Acoustics — Cascade

| Display name | ID | Family | Format / version | Behavior | Geography / locality | Developer / owner | AI4G relationship | License | Commercial use |
|---|---|---|---|---|---|---|---|---|---|
| Orca Cascade | `orca-cascade` | DCLDE-orca | cascade | Cascade · pipeline descriptor | North Pacific — Pacific Northwest and Salish Sea | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |

## Overhead — Detector

| Display name | ID | Family | Format / version | Behavior | Geography / locality | Developer / owner | AI4G relationship | License | Commercial use |
|---|---|---|---|---|---|---|---|---|---|
| HerdNet General Dataset 2022 | `HerdNet_General_Dataset_2022` | — | onnx | Detector · direct species output | Sub-Saharan Africa — aerial savanna | University of Liege, Gembloux Agro-Bio Tech (Delplanque et al.) | Third-party | CC-BY-NC-SA-4.0 | ❌ |
| OWL | `OWL` | — | onnx | Detector · gates on animal | Global | Microsoft AI for Good Lab (AI4G) | First-party (AI4G) | MIT | ✅ |

## General — Encoder

| Display name | ID | Family | Format / version | Behavior | Geography / locality | Developer / owner | AI4G relationship | License | Commercial use |
|---|---|---|---|---|---|---|---|---|---|
| Bioclip 2 | `bioclip-2` | BioCLIP | onnx · v2 | Encoder · embeddings | Foundational (global) | Imageomics Institute and Ohio State University | Third-party | MIT | ✅ |
| dinov3 vitl16 | `dinov3-vitl16` | DINOv3 | onnx · vitl16-lvd1689m | Encoder · embeddings | Foundational (global) | Meta AI | Third-party | DINOv3 License | ✅ |

## References & citations

Canonical source + citation per model (from the model-zoo metadata audit):

- **`MDV5a`** — MegaDetector v5 (MDv5a), Morris / Beery et al.; https://github.com/agentmorris/MegaDetector/releases/tag/v5.0
- **`MDV6-yolov10-c`** — MegaDetector V6 model zoo, Microsoft AI for Good; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md; weights DOI: https://doi.org/10.5281/zenodo.15398270
- **`MDV6-yolov10-c-tflite`** — MegaDetector V6 yolov10-c TFLite conversion; inherits MDV6-yolov10-c; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md
- **`MDV6-yolov10-e`** — MegaDetector V6 model zoo, Microsoft AI for Good; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md; weights DOI: https://doi.org/10.5281/zenodo.15398270
- **`deepfaune-yolo8s`** — DeepFaune detector (YOLOv8s), Rigoudy et al. 2023; https://plmlab.math.cnrs.fr/deepfaune/software; model parameters: https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.3/deepfaune-yolov8s_960.pt
- **`european_mammals`** — Microsoft AI for Good Lab regional YOLO detector (European mammals); canonical upstream URL UNVERIFIED; Sparrow redistribution DOI: https://doi.org/10.5281/zenodo.20563673
- **`north_american_mammals`** — Microsoft AI for Good Lab regional YOLO detector (North American mammals); canonical upstream URL UNVERIFIED; Sparrow redistribution DOI: https://doi.org/10.5281/zenodo.20563673
- **`sub_saharan`** — Microsoft AI for Good Lab regional YOLO detector (Sub-Saharan mammals); canonical upstream URL UNVERIFIED; Sparrow redistribution DOI: https://doi.org/10.5281/zenodo.20563673
- **`MDV5b`** — MegaDetector v5 (MDv5b), Morris / Beery et al.; https://github.com/agentmorris/MegaDetector/releases/tag/v5.0
- **`MD1000-redwood`** — MegaDetector v1000 redwood, Morris / MegaDetector team; https://github.com/agentmorris/MegaDetector/releases/tag/v1000.0; release notes: https://raw.githubusercontent.com/agentmorris/MegaDetector/main/docs/release-notes/mdv1000-release.md
- **`MD1000-spruce`** — MegaDetector v1000 spruce, Morris / MegaDetector team; https://github.com/agentmorris/MegaDetector/releases/tag/v1000.0; release notes: https://raw.githubusercontent.com/agentmorris/MegaDetector/main/docs/release-notes/mdv1000-release.md
- **`MD1000-larch`** — MegaDetector v1000 larch, Morris / MegaDetector team; https://github.com/agentmorris/MegaDetector/releases/tag/v1000.0; release notes: https://raw.githubusercontent.com/agentmorris/MegaDetector/main/docs/release-notes/mdv1000-release.md
- **`MD1000-cedar`** — MegaDetector v1000 cedar, Morris / MegaDetector team; https://github.com/agentmorris/MegaDetector/releases/tag/v1000.0; release notes: https://raw.githubusercontent.com/agentmorris/MegaDetector/main/docs/release-notes/mdv1000-release.md
- **`MD1000-sorrel`** — MegaDetector v1000 sorrel, Morris / MegaDetector team; https://github.com/agentmorris/MegaDetector/releases/tag/v1000.0; release notes: https://raw.githubusercontent.com/agentmorris/MegaDetector/main/docs/release-notes/mdv1000-release.md
- **`MDV6-yolov9-c`** — MegaDetector V6 model zoo, Microsoft AI for Good; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md; weights DOI: https://doi.org/10.5281/zenodo.15398270
- **`MDV6-yolov9-e`** — MegaDetector V6 model zoo, Microsoft AI for Good; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md; weights DOI: https://doi.org/10.5281/zenodo.15398270
- **`MDV6-rtdetr-c`** — MegaDetector V6 model zoo, Microsoft AI for Good; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md; weights DOI: https://doi.org/10.5281/zenodo.15398270
- **`MDV6-mit-yolov9-c`** — MegaDetector V6 MIT YOLO-MIT variant, Microsoft AI for Good; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md; weights DOI: https://doi.org/10.5281/zenodo.15398270
- **`MDV6-mit-yolov9-e`** — MegaDetector V6 MIT YOLO-MIT variant, Microsoft AI for Good; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md; weights DOI: https://doi.org/10.5281/zenodo.15398270
- **`MDV6-apa-rtdetr-c`** — MegaDetector V6 Apache RT-DETR variant, Microsoft AI for Good; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md; weights DOI: https://doi.org/10.5281/zenodo.15398270
- **`MDV6-apa-rtdetr-e`** — MegaDetector V6 Apache RT-DETR variant, Microsoft AI for Good; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md; weights DOI: https://doi.org/10.5281/zenodo.15398270
- **`AI4G-Amazon-V2`** — Microsoft AI for Good Lab, Biodiversity model-zoo classifier table (AI4G-Amazon-V2 v2): https://microsoft.github.io/Biodiversity/model_zoo/classifiers/
- **`AI4G-Serengeti`** — Microsoft AI for Good Lab, Biodiversity model-zoo classifier table (AI4G-Serengeti): https://microsoft.github.io/Biodiversity/model_zoo/classifiers/
- **`Deepfaune-Europe`** — Microsoft Biodiversity model-zoo classifier table (Deepfaune-classification v1.3) and Rigoudy et al. 2023, DOI 10.1007/s10344-023-01742-7: https://microsoft.github.io/Biodiversity/model_zoo/classifiers/
- **`Deepfaune-New-England`** — Clarfeld et al. 2025, DeepFaune New England, USGS software release, DOI 10.5066/P13T4EKE: https://code.usgs.gov/vtcfwru/deepfaune-new-england
- **`SpeciesNet-Crop`** — Google SpeciesNet / cameratrapai repository; Gadot et al. 2024, DOI 10.1049/cvi2.12318: https://github.com/google/cameratrapai
- **`southwest-usa-v3`** — Addax Data Science / San Diego Zoo Wildlife Alliance, Southwest USA v3 model card: https://huggingface.co/Addax-Data-Science/Southwest_USA_v3
- **`peruvian-andes`** — Addax Data Science / San Diego Zoo Wildlife Alliance, Peruvian Andes model card: https://huggingface.co/Addax-Data-Science/Peruvian_Andes
- **`sub-saharan-drylands`** — Addax Data Science, Sub-Saharan Drylands model card: https://huggingface.co/Addax-Data-Science/sub_saharan_drylands_v1.pt
- **`manas-panthera`** — Hex·Data / OSI-Panthera, Panthera model card: https://huggingface.co/Hex-Data/Panthera
- **`gifu-japan`** — Addax Data Science / Gifu University, Japan Gifu v0.2 model card: https://huggingface.co/Addax-Data-Science/Japan_Gifu_v0.2
- **`hawaii-puaa`** — Addax Data Science, Hawaiʻi AI Puaʻa v1.0 model card: https://huggingface.co/Addax-Data-Science/HWI-ADS-v1
- **`central-india`** — Addax Data Science / Wildlife Conservation Trust India, Central Indian Landscapes model card: https://huggingface.co/Addax-Data-Science/IND-ADS-v1
- **`top-end-savanna`** — Addax Data Science / Charles Darwin University / Territory Natural Resource Management / Warddeken Land Management, Top End Savanna model card: https://huggingface.co/Addax-Data-Science/ANT-ADS-v1
- **`parks-victoria`** — Addax Data Science. VIC-ADS-v1 (Parks Victoria classifier). Hugging Face model repository: https://huggingface.co/Addax-Data-Science/VIC-ADS-v1
- **`sw-borderlands`** — Addax Data Science. SBUSA-ADS-v1 (Southwestern Borderlands USA classifier). Hugging Face model repository: https://huggingface.co/Addax-Data-Science/SBUSA-ADS-v1
- **`ahdrift`** — Addax Data Science. AHDRIFT-v1 (AHDriFT-ID Midwest US classifier). Hugging Face model repository: https://huggingface.co/Addax-Data-Science/AHDRIFT-v1
- **`deep-forest-vision`** — MNHN-OFVI / Addax Data Science. AFR-DFV-v1 / DeepForestVision classifier. Hugging Face model repository: https://huggingface.co/Addax-Data-Science/AFR-DFV-v1; original project: https://github.com/MNHN-OFVI/DeepForestVision
- **`awc135`** — Australian Wildlife Conservancy / Addax Data Science. AWC135-AWC-v1 / Australian Wildlife Classifier. Hugging Face model repository: https://huggingface.co/Addax-Data-Science/AWC135-AWC-v1; source license: https://github.com/Australian-Wildlife-Conservancy-AWC/awc-wildlife-classifier/blob/main/LICENSE
- **`namibian`** — Addax Data Science. Namib-Desert-v1 (Namibian Desert classifier). Hugging Face model repository: https://huggingface.co/Addax-Data-Science/Namib-Desert-v1
- **`iran`** — Addax Data Science. Iran_v1 (Iran classifier). Hugging Face model repository: https://huggingface.co/Addax-Data-Science/Iran_v1
- **`nz-invasives`** — Addax Data Science / New Zealand Department of Conservation. New_Zealand_v1 (New Zealand invasives classifier). Hugging Face model repository: https://huggingface.co/Addax-Data-Science/New_Zealand_v1
- **`queensland`** — WildObs / Addax Data Science. WetTropics_WildObs (Queensland Wet Tropics classifier). Hugging Face model repository: https://huggingface.co/Addax-Data-Science/WetTropics_WildObs
- **`nz-species`** — New Zealand Department of Conservation / wekaResearch / Addax Data Science. NZS-WEK-v3-03 (New Zealand Species v3.03 classifier). Hugging Face model repository: https://huggingface.co/Addax-Data-Science/NZS-WEK-v3-03
- **`tropicam-ai`** — Andrea Zampetti / MNCN-CSIC / Addax Data Science. NEO-MNCN-v1-0 / TropiCam-AI v1.0 classifier. Hugging Face model repository: https://huggingface.co/Addax-Data-Science/NEO-MNCN-v1-0; original project: https://github.com/andrewzamp/TropiCam-AI
- **`terai-nepal`** — Alexander Merdian-Tarko. TeraiNet (Terai Nepal classifier). Hugging Face model repository: https://huggingface.co/alexvmt/TeraiNet; project details: https://github.com/alexvmt/TeraiNet
- **`tasmanian-vertebrates`** — MEWC / Addax Data Science. Tasmanian_vertebrates classifier. Hugging Face model repository: https://huggingface.co/Addax-Data-Science/Tasmanian_vertebrates; MEWC source: https://github.com/zaandahl/mewc
- **`peruvian-amazon-sdzwa`** — San Diego Zoo Wildlife Alliance / Addax Data Science. Peruvian_Amazon classifier. Hugging Face model repository: https://huggingface.co/Addax-Data-Science/Peruvian_Amazon
- **`md-audiobirds-v1`** — Microsoft AI for Good Lab. MegaDetector-Acoustic / MD_AudioBirds_V1. Source: https://github.com/microsoft/MegaDetector-Acoustic; model-zoo page: https://microsoft.github.io/Biodiversity/model_zoo/bioacoustics/. Cite: Hernandez et al. 2024, Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation, arXiv:2405.12930.
- **`orca-detector-dclde2026-v3`** — Microsoft AI for Good Lab and Microsoft Pytorch-Wildlife project. Sparrow Engine model zoo v0.9.0: orca-detector-dclde2026-v3. Zenodo. https://doi.org/10.5281/zenodo.20864372. Upstream/challenge context: DCLDE 2026 killer whale detection and ecotype classification dataset/challenge.
- **`orca-detector-v3-fp16-tflite`** — Converted TFLite FP16 variant of orca-detector-dclde2026-v3. Reference the ONNX sibling: Microsoft AI for Good Lab and Microsoft Pytorch-Wildlife project, Sparrow Engine model zoo v0.9.0, https://doi.org/10.5281/zenodo.20864372.
- **`orca-detector-v3-int8-tflite`** — Converted TFLite INT8 variant of orca-detector-dclde2026-v3. Reference the ONNX sibling: Microsoft AI for Good Lab and Microsoft Pytorch-Wildlife project, Sparrow Engine model zoo v0.9.0, https://doi.org/10.5281/zenodo.20864372.
- **`orca-ecotype-dclde2026-v1`** — Microsoft AI for Good Lab and Microsoft Pytorch-Wildlife project. Sparrow Engine model zoo v0.5.0: orca-ecotype-dclde2026-v1. Zenodo. https://doi.org/10.5281/zenodo.20563673. Upstream/challenge context: DCLDE 2026 killer whale detection and ecotype classification dataset/challenge.
- **`orca-ecotype-melinput-fp16-tflite`** — Converted TFLite FP16 variant of orca-ecotype-dclde2026-v1. Reference the ONNX sibling: Microsoft AI for Good Lab and Microsoft Pytorch-Wildlife project, Sparrow Engine model zoo v0.5.0, https://doi.org/10.5281/zenodo.20563673.
- **`orca-ecotype-melinput-int8-tflite`** — Converted TFLite INT8 variant of orca-ecotype-dclde2026-v1. Reference the ONNX sibling: Microsoft AI for Good Lab and Microsoft Pytorch-Wildlife project, Sparrow Engine model zoo v0.5.0, https://doi.org/10.5281/zenodo.20563673.
- **`perch-v2`** — Google Research. Perch / Perch 2 bioacoustics model. Source: https://github.com/google-research/perch. Cite: Ghani, Denton, Kahl et al., Global birdsong embeddings enable superior transfer learning for bioacoustic classification, Scientific Reports / arXiv:2307.06292, https://arxiv.org/abs/2307.06292.
- **`orca-cascade`** — Pipeline descriptor combining orca-detector-dclde2026-v3 and orca-ecotype-dclde2026-v1. Cite both component model references; cascade artifact first published in Sparrow Engine model zoo v0.7.0, https://doi.org/10.5281/zenodo.20723037.
- **`HerdNet_General_Dataset_2022`** — Delplanque, A., Foucher, S., Théau, J., Bussière, E., Vermeulen, C., and Lejeune, P. From crowd to herd counting: How to precisely detect and count African mammals using aerial imagery and deep learning? ISPRS Journal of Photogrammetry and Remote Sensing 197 (2023), 167-180. https://doi.org/10.1016/j.isprsjprs.2023.01.025. Source: https://github.com/Alexandre-Delplanque/HerdNet.
- **`OWL`** — Chacón et al. Overhead Wildlife Locator (OWL): Benchmarking Weakly Supervised Learning for Aerial Wildlife Surveys. arXiv:2606.13911. https://arxiv.org/abs/2606.13911. Source (MIT): https://github.com/microsoft/MegaDetector-Overhead (Microsoft AI for Good Lab).
- **`bioclip-2`** — Gu, J., Stevens, S., Campolongo, E. G., Thompson, M. J., Zhang, N., Wu, J., Kopanev, A., Mai, Z., White, A. E., Balhoff, J., Dahdul, W. M., Rubenstein, D., Lapp, H., Berger-Wolf, T., Chao, W.-L., and Su, Y. BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning. NeurIPS 2025. https://arxiv.org/abs/2505.23883. Model: imageomics/bioclip-2, Hugging Face, DOI https://doi.org/10.57967/hf/5765, https://huggingface.co/imageomics/bioclip-2.
- **`dinov3-vitl16`** — Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., et al. DINOv3. Meta AI, 2025. https://arxiv.org/abs/2508.10104. Model: facebook/dinov3-vitl16-pretrain-lvd1689m (LVD-1689M pretrain, ViT-L/16, 1024-d), Hugging Face, https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m. License: DINOv3 License (custom Meta license).
