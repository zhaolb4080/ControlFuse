<div align="center">

# ControlFuse

### Instruction-guided Multi-Granularity Controllable Image Fusion

**Official PyTorch implementation of the AAAI 2026 paper**

[![Paper](https://img.shields.io/badge/AAAI-2026-7B1FA2.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/38321)
![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-EE4C2C.svg?logo=pytorch&logoColor=white)
![Version](https://img.shields.io/badge/version-0.5.5-2F80ED.svg)

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/38321) ·
[Model Zoo](#model-zoo) ·
[Installation](#installation) ·
[Data Preparation](#data-preparation) ·
[Training](#training) ·
[Evaluation](#evaluation)

</div>

## Overview

ControlFuse is an instruction-guided infrared-visible image fusion framework that supports **global**, **semantic**, and **instance-level** control in a single model. Given the same registered infrared-visible pair, natural-language instructions determine which content should be emphasized and where the enhancement should occur.

This repository provides the model, multi-granularity data construction tools, training and inference pipelines, fusion-quality metrics, and control-localization evaluation.

<p align="center">
  <img src="assets/teaser.png" alt="Comparison of controllable infrared-visible image fusion paradigms" width="95%">
</p>
<p align="center"><em>Comparison of controllable IVIF fusion paradigms given the same VIS-IR inputs.</em></p>

## Framework

<p align="center">
  <img src="assets/framework.png" alt="Overview of the ControlFuse framework" width="100%">
</p>
<p align="center"><em>Overview of the proposed multi-granularity controllable fusion framework.</em></p>

## Highlights

- **Three control granularities.** One framework handles scene-level, category-level, and object-level instructions.
- **Unified multimodal interaction.** FMC projects visual and textual representations into a shared manifold space.
- **Fine-grained spatial alignment.** CGI uses curvature-aware interaction to connect instructions with spatial visual evidence.
- **Control-aware optimization.** Localization, boundary, same-class distractor, content-fidelity, and text-alignment objectives are jointly optimized.
- **Complete experiment pipeline.** The release covers data construction, training, inference, six fusion metrics, localization evaluation, and tests.

## Control Granularities

| Granularity | Example instruction | Expected behavior |
| --- | --- | --- |
| Global | *Enhance the entire scene.* | Fuse complementary information across the complete image. |
| Semantic | *Highlight all pedestrians.* | Emphasize all regions belonging to the requested category. |
| Instance | *Emphasize the leftmost pedestrian.* | Enhance only the specified object while suppressing same-class distractors. |

## Release Status

| Dataset | Data tool | Training config | Evaluation tool |
| --- | :---: | :---: | :---: |
| MSRS | ✓ | ✓ | ✓ |
| M3FD | ✓ | ✓ | ✓ |
| RoadScene | Aligned-pair manifest | Generic configuration | Fusion metrics |

## Model Zoo

| Dataset | Model | Download |
| --- | --- | --- |
| MSRS | ControlFuse | [Baidu Netdisk](https://pan.baidu.com/s/1gO1pnY19BiKmclZO05PtOQ?pwd=m6tf) (access code: `m6tf`) |

## Installation

Python 3.10 or 3.11 is recommended.

~~~bash
git clone https://github.com/zhaolb4080/ControlFuse.git
cd ControlFuse

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

On Windows, activate the environment with:

~~~powershell
.venv\Scripts\activate
~~~

PyTorch 2.1–2.3 Windows wheels use the NumPy 1.x ABI. If the environment already contains NumPy 2.x:

~~~bash
pip uninstall -y numpy
pip install --no-cache-dir "numpy>=1.24,<2"
~~~

## Data Format

Each JSONL row represents one image-pair/instruction/mask sample:

~~~json
{"name":"00001_instance_3","visible":"images/00001.png","infrared":"infrared/00001.png","mask":"masks/00001_instance_3.png","instruction":"Emphasize the leftmost pedestrian.","negative_instruction":"Emphasize an absent car on the right.","granularity":"instance"}
~~~

The same registered image pair may appear in multiple rows with different instructions and masks. A global row may omit `mask`; the loader then creates an all-one mask.

## Data Preparation

### MSRS

For the official `train|test/{vi,ir,Segmentation_labels}` layout, generate semantic and connected-component pseudo-instance rows:

~~~bash
python tools/build_msrs_multigranularity.py \
  --split-root D:/Datasets/MSRS/train \
  --output data/msrs_train_control.jsonl \
  --min-instance-area 20 \
  --max-instances 5
~~~

Create global rows from the aligned folders:

~~~bash
python tools/build_manifest.py \
  --visible-dir D:/Datasets/MSRS/train/vi \
  --infrared-dir D:/Datasets/MSRS/train/ir \
  --output data/msrs_train_global.jsonl \
  --instruction "Enhance the entire scene." \
  --negative-instruction "Suppress the entire scene." \
  --granularity global
~~~

Combine the global and control rows into `data/msrs_train.jsonl`, then verify the formal v5 data:

~~~bash
python tools/check_v5_data.py --manifest data/msrs_train.jsonl
~~~

MSRS instance masks are obtained from 8-connected components of indexed semantic labels. Touching same-class objects can be refined with SAM when more precise instance boundaries are required.

<details>
<summary><b>M3FD preparation</b></summary>

M3FD uses `Ir/`, `Vis/`, and Pascal VOC `Annotation/*.xml` folders. Create the deterministic 3,780/420 split:

~~~bash
python tools/split_m3fd.py \
  --root D:/Datasets/M3FD \
  --train-count 3780 \
  --test-count 420 \
  --seed 2026
~~~

Install Segment Anything and obtain an official SAM checkpoint:

~~~bash
pip install git+https://github.com/facebookresearch/segment-anything.git
~~~

Build the combined global/semantic/instance training manifest:

~~~bash
python tools/build_m3fd_multigranularity.py \
  --split-root D:/Datasets/M3FD/train \
  --output data/m3fd_train.jsonl \
  --sam-checkpoint D:/Weights/sam_vit_b_01ec64.pth \
  --sam-model-type vit_b \
  --device cuda \
  --include-global \
  --min-instance-area 64 \
  --max-instances 5
~~~

Build the semantic/instance localization test manifest:

~~~bash
python tools/build_m3fd_multigranularity.py \
  --split-root D:/Datasets/M3FD/test \
  --output data/m3fd_test_control.jsonl \
  --sam-checkpoint D:/Weights/sam_vit_b_01ec64.pth \
  --sam-model-type vit_b \
  --device cuda \
  --min-instance-area 64 \
  --max-instances 5
~~~

Create the global test manifest:

~~~bash
python tools/build_manifest.py \
  --visible-dir D:/Datasets/M3FD/test/Vis \
  --infrared-dir D:/Datasets/M3FD/test/Ir \
  --output data/m3fd_test_global.jsonl \
  --instruction "Enhance the entire scene." \
  --negative-instruction "Suppress the entire scene." \
  --granularity global
~~~

## Training

### MSRS

~~~bash
python tools/check_v5_data.py --manifest data/msrs_train.jsonl
python train.py --config configs/msrs.yaml
~~~

Resume an interrupted finite run:

~~~bash
python train.py \
  --config configs/msrs.yaml \
  --resume runs/controlfuse_msrs_v5/last.pt
~~~

### M3FD

Start M3FD from random initialization:

~~~bash
python tools/check_v5_data.py --manifest data/m3fd_train.jsonl
python train.py --config configs/m3fd.yaml
~~~

Resume only an interrupted M3FD run:

~~~bash
python train.py \
  --config configs/m3fd.yaml \
  --resume runs/controlfuse_m3fd_v5/last.pt
~~~

## Inference

### Single image pair

~~~bash
python infer.py \
  --config configs/msrs.yaml \
  --checkpoint runs/controlfuse_msrs_v5/last.pt \
  --visible examples/visible.png \
  --infrared examples/infrared.png \
  --instruction "Highlight the leftmost pedestrian." \
  --output results/fused.png \
  --mask-output results/location.png
~~~

### Whole manifest

~~~bash
python infer_manifest.py \
  --config configs/msrs.yaml \
  --checkpoint runs/controlfuse_msrs_v5/last.pt \
  --manifest data/msrs_test_global.jsonl \
  --output-dir results/msrs_global_v5 \
  --batch-size 4
~~~

For controllability analysis, keep the infrared-visible pair fixed and vary only the instruction. Compare both the fused image and the predicted location map.

## Evaluation

### Global fusion quality

~~~bash
python evaluate.py \
  --manifest data/msrs_test_global.jsonl \
  --fused-dir results/msrs_global_v5 \
  --output results/msrs_global_metrics_v5.csv
~~~

`evaluate.py` reports EN, SD, SF, AG, VIF, and Qabf.

### Semantic and instance localization

~~~bash
python infer_manifest.py \
  --config configs/msrs.yaml \
  --checkpoint runs/controlfuse_msrs_v5/last.pt \
  --manifest data/msrs_test_control.jsonl \
  --output-dir results/msrs_control_v5 \
  --batch-size 4 \
  --save-masks

python evaluate_localization.py \
  --manifest data/msrs_test_control.jsonl \
  --predicted-dir results/msrs_control_v5/masks \
  --output results/msrs_localization_v5.csv \
  --summary results/msrs_localization_summary_v5.json \
  --threshold 0.5
~~~

The localization evaluator reports IoU, F1, precision, and recall for semantic and instance instructions separately.

## Repository Structure

~~~text
ControlFuse/
├── assets/                   # README teaser and framework figures
├── controlfuse/              # Model, data loader, losses, metrics, checkpoints
├── configs/                  # Generic, MSRS, M3FD, and smoke-test configs
├── tools/                    # Dataset builders, validators, and profiling
├── tests/                    # Smoke, data, and loss-behavior tests
├── train.py                  # Training entry point
├── infer.py                  # Single-pair inference
├── infer_manifest.py         # Batch inference
├── evaluate.py               # Six fusion-quality metrics
├── evaluate_localization.py  # Semantic/instance localization metrics
└── IMPLEMENTATION.md         # Module mapping and protocol details
~~~

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the module-to-code mapping, released engineering details, and the complete evaluation protocol. Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Citation

If ControlFuse is useful in your research, please cite:

~~~bibtex
@inproceedings{zhao2026controlfuse,
  title     = {ControlFuse: Instruction-guided Multi-Granularity Controllable Image Fusion},
  author    = {Zhao, Libo and Zhang, Xiaoli and Wang, Zeyu},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {16},
  pages     = {13199--13207},
  year      = {2026}
}
~~~
