# Implementation Details and Evaluation Protocol

This document maps the main ControlFuse components to the released code and records the engineering and evaluation details used by the public training pipeline.

## Module-to-Code Mapping

| Component | Released implementation |
| --- | --- |
| Global, semantic, and instance control | A unified JSONL schema provides an instruction and its spatial supervision mask. `ControlFuse.forward` is shared across all granularities. |
| Visual encoders | `ImageEncoder` and `RestormerBlock` in `controlfuse/model.py`. |
| Feature Manifold Converter | `SparseFeatureManifoldConverter` projects both visual modalities and text into a common dimension, applies local visual messages, and exchanges pooled visual-text messages bidirectionally. |
| Curvature-Guided Interaction | `CurvatureGuidedInteraction` computes spatial Laplacians and token second differences, inserts normalized curvature into cross-modal interaction, selects instruction-relevant tokens, and predicts a refined location map. |
| Multi-scale reconstruction | Full-, half-, and quarter-resolution Restormer branches are merged before RGB projection. |
| Localization objective | Size-balanced focal and FP-aware Tversky supervision, local boundary refinement, same-class distractor constraints, an empty-crop background term, and reduced weight for global all-one masks. |
| Content fidelity | Instruction-mask-conditioned intensity and gradient targets, SSIM, and visible-light chroma preservation. |
| Text-visual alignment | Dense foreground/background positive-negative ranking, including correct handling of empty local crops, weighted by visual and text curvature. |
| Adaptive loss weighting | Three learned log standard deviations with bounded values and a fixed localization floor. |

## Released Engineering Details

1. **Text encoding.** The public pipeline uses frozen CLIP token embeddings. The text adapter is isolated in `controlfuse/text.py`.
2. **Sparse manifold interaction.** Local depthwise visual messages retain spatial adjacency, while pooled cross-attention on a `16×16` grid supplies cross-modal interaction at practical memory cost.
3. **One instruction-mask pair per row.** Each JSONL row represents one requested object, semantic group, or full scene. Repeated rows may share the same infrared-visible pair while using different instructions and masks.
4. **Dense paired ranking.** Foreground features rank the positive instruction above its negative counterpart; background features use the opposite ordering. This keeps small regions from being diluted by global pooling.
5. **Size-balanced localization.** Per-sample positive weighting and a region-normalized Tversky term provide useful gradients for sparse targets while controlling false positives.
6. **Mask-conditioned fidelity.** Global instructions use max-source luminance and max-magnitude signed Sobel targets. Local instructions use those targets inside the requested mask and preserve visible content outside it.
7. **Multi-scale target crops.** Local samples use target-centered 128, 192, or 256 pixel paired crops. Fifteen percent of semantic rows and thirty percent of instance rows retain the full scene.
8. **Same-class distractor supervision.** For each instance row, the loader subtracts the target instance from its paired semantic mask. Other same-class objects are then suppressed in the localization and text-alignment objectives.
9. **Boundary control.** A region-normalized morphological boundary Dice term sharpens semantic and instance location maps. Global and empty rows skip this term.
10. **Stable optimization.** Adaptive uncertainty weights are bounded, curvature is computed in FP32 under AMP, and training stops before any non-finite loss update.

## Training Protocol

The formal configuration uses:

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Epochs | 100 |
| Batch size | 8 |
| Initial learning rate | `1 × 10⁻⁴` |
| Learning-rate schedule | Multiply by 0.5 every 20 epochs |
| Input size | `256 × 256` |
| Mixed precision | Enabled |

MSRS uses all 1,083 training pairs and evaluates the final fixed-epoch checkpoint on the 361-pair test split. M3FD uses the fixed 3,780/420 split and starts from random initialization under a separate checkpoint schema.

## Results Reported in the Paper

| Dataset | EN | SD | SF | AG | VIF | Qabf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSRS | 5.99 | 61.15 | 11.94 | 4.15 | 1.07 | 0.76 |
| RoadScene | 5.19 | 55.96 | 13.06 | 5.05 | 0.85 | 0.59 |
| M3FD | 6.26 | 68.05 | 16.60 | 6.82 | 0.92 | 0.81 |

## Evaluation Protocol

1. Split datasets by source-image identity: MSRS 1,083/361, RoadScene 171/50, and M3FD 3,780/420 in the released pipeline.
2. Train every model and ablation for 100 epochs from the same seed.
3. Evaluate global instructions with `evaluate.py` and use the same metric implementation for every compared method.
4. Evaluate semantic and instance location maps with `evaluate_localization.py` at a fixed threshold, reporting IoU, F1, precision, and recall.
5. For the key controllability test, hold the infrared-visible pair fixed and vary only the instruction. Compare both fused images and predicted location maps.
6. Evaluate downstream detector or segmenter performance under one fixed protocol for all compared fusion outputs.
7. Keep test manifests out of `val_manifest` when following the fixed-epoch protocol.

## Checkpoint Compatibility

- `controlfuse-v5` is used by the generic and MSRS configurations.
- `controlfuse-v5-m3fd` is used by M3FD.
- `--resume` is intended only for continuing an interrupted run under the same training schema.
- M3FD rejects an MSRS checkpoint passed to `--resume`.
- If a run reports a non-finite loss, resume only from the latest finite periodic checkpoint created before that event.

