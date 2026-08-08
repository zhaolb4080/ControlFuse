# Changelog

## 0.5.5

- Add `configs/m3fd.yaml` for a fresh 100-epoch v5 run on the fixed 3,780-pair
  M3FD training split, using the combined SAM-derived multi-granularity
  manifest and no test-set model selection.
- Isolate M3FD checkpoints with the `controlfuse-v5-m3fd` training schema so an
  MSRS checkpoint cannot accidentally be used for cross-dataset fine-tuning.
- Save M3FD recovery checkpoints every five epochs under
  `runs/controlfuse_m3fd_v5` while retaining `last.pt` after every epoch.

## 0.5.4

- Add `tools/build_m3fd_multigranularity.py` for the official Pascal VOC M3FD
  annotations and fixed `train/test/{Vis,Ir,Annotation}` layout.
- Use human-annotated boxes as SAM prompts, cache all object masks, merge them
  into class-level semantic masks, and emit v5-compatible instance rows linked
  by `semantic_mask` for same-class distractor supervision.
- Support six normalized M3FD categories, spatial positive/negative
  instructions, at most five salient instances per image, resumable mask
  generation, and an explicitly debug-only rectangular-box mode.

## 0.5.3

- Replace `Path.write_text(..., newline=...)` with an atomic file writer that
  also works in older Python environments.
- Make `split_m3fd.py` safely resumable after an interrupted copy: verify that
  existing filenames and sizes belong to the same deterministic split, reuse
  valid files, copy missing files, and regenerate split metadata.

## 0.5.2

- Accept the official M3FD Pascal VOC files whose legacy internal `<filename>`
  values do not match the zero-padded external XML/image stems.
- Continue validating XML syntax and external `Ir/Vis/Annotation` stem pairing,
  report internal-name mismatches as warnings, and offer an opt-in
  `--strict-xml-filename` check for cleaned annotations.

## 0.5.1

- Add `tools/split_m3fd.py` for the official `Ir/`, `Vis/`, and Pascal VOC
  `Annotation/` layout.
- Create a deterministic cross-platform 3,780/420 split using a seeded SHA-256
  ordering, while preserving the original 4,200 image pairs and annotations.
- Validate exact stem correspondence and XML integrity before copying, then
  save train/test name lists and a deterministic split fingerprint.

## 0.5.0

- Preserve v4 target-centered multi-scale crops, size-balanced sampling, learned
  token selection, and dense foreground/background alignment.
- Shift Tversky from recall-biased 0.4/0.6 to FP-aware 0.55/0.45 and cap the
  per-sample positive weight at 10 instead of 20.
- Add a region-normalized morphological boundary Dice term and a very small
  local-only false-positive term; neither is applied to global or empty rows.
- Derive same-class instance distractors automatically by subtracting each
  instance mask from its paired semantic mask, without requiring old manifests
  to be rebuilt.
- Keep 15% of semantic and 30% of instance rows as full-scene training views so
  spatial instructions see their target and competing objects together; the
  remaining rows retain v4 multi-scale target crops.
- Suppress target probability and target-text similarity specifically on other
  same-class instances while preserving the ordinary background objective.
- Report boundary and distractor losses in the training log, isolate v5
  checkpoints under a new training schema, and add behavior tests for both
  constraints.
- Add `tools/check_v5_data.py` to verify semantic/instance mask containment and
  report same-class distractor coverage before formal training.

## 0.4.0

- Replace the v3 Focal/Dice/false-positive/area objective with per-sample
  size-balanced Focal and recall-aware Tversky localization. Empty local crops
  retain a separate low-weight background rejection objective.
- Add a fixed localization-weight floor so learned uncertainty cannot suppress
  the control branch when semantic or instance localization is difficult.
- Use target-centered multi-scale paired crops and joint class/target-size
  sampling to strengthen rare and small MSRS objects.
- Replace uniform text-token averaging with a learned token selector and add
  detail-preserving plus dilated-context branches to the location head.
- Compute positive/negative text alignment densely inside foreground and
  background regions instead of pooling each region to one feature vector.
- Select validation checkpoints by threshold-independent soft semantic/instance
  IoU and report hard IoU plus zero-IoU rates separately. v4 checkpoints are
  schema-incompatible with v3.

## 0.3.0

- Make fidelity instruction-mask conditioned: global rows retain max-source
  intensity/gradient fusion, while semantic and instance rows preserve visible
  content outside the requested region.
- Add explicit false-positive and predicted-area penalties and reduce the
  positive Focal/Dice bias that expanded v2 control masks.
- Reduce the global all-one-mask sampling mass to 20% and its contribution to
  localization optimization while retaining class-balanced local sampling.
- Replace foreground-only text alignment with symmetric foreground/background
  ranking. Empty local crops now skip foreground alignment and supervise only
  background rejection instead of treating the whole crop as a positive area.
- Add a checkpoint training-schema guard that rejects v1/v2 checkpoints during
  v3 training and writes new runs to `runs/controlfuse_msrs_v3`.
- Add behavior tests for over-expanded masks, empty local masks, sampling mass,
  and mask-conditioned fidelity targets.

## 0.2.0

- Replace the averaging-prone source fidelity objective with max-intensity,
  source-selective Sobel gradient, SSIM, and visible-chroma losses.
- Use paired native-resolution crops for VIS, IR, and masks during training,
  with positive-region crops for semantic and instance rows.
- Add balanced replacement sampling across global, semantic, and instance
  granularity and inverse-square-root class balancing within local rows.
- Pool visual features inside the ground-truth instruction mask for local
  positive/negative text alignment instead of aligning every instruction to a
  whole-image average.
- Add a trainable convolutional localization head on top of normalized
  text-visual similarity and use focal plus Dice mask supervision.
- Write new MSRS checkpoints to `runs/controlfuse_msrs_v2` so version 0.1
  checkpoints are preserved and cannot be mistaken for version 0.2 training.

## 0.1.7

- Restore batch inference outputs and location maps to each source image's
  original resolution before saving.
- Validate IR/VIS/mask spatial consistency in the dataset loader.
- Add `evaluate_localization.py` for semantic/instance IoU, F1, precision, and
  recall with separate granularity summaries.
- Detect resolution mismatches before computing fusion metrics.

## 0.1.6

- Add epsilon inside visual and textual curvature square roots, eliminating the
  singular gradient at exactly zero curvature in flat image regions.
- Run Restormer attention normalization and output embedding normalization in
  FP32 with an AMP-safe epsilon.
- Report the first parameters containing non-finite gradients when a step is
  skipped.

## 0.1.5

- Bound learned uncertainty log variances and clamp them after every optimizer
  step, preventing exponential loss-weight overflow late in training.
- Compute visual and textual curvature in FP32 and bound CGI's learned
  curvature scale under AMP.
- Stop before an optimizer step on non-finite losses, skip transient non-finite
  gradients, and print all loss components plus uncertainty values.

## 0.1.4

- Allow `val_manifest: null` for fixed-epoch training without repeatedly
  evaluating on the test set.
- Add `configs/msrs.yaml` for formal 100-epoch MSRS training using the complete
  1,083-pair training split.

## 0.1.3

- Force frozen CLIP to load `model.safetensors` instead of the pickle-based
  `pytorch_model.bin`, allowing safe use with PyTorch 2.1-2.5 and recent
  Transformers security checks.
- Declare the `safetensors` runtime dependency explicitly.

## 0.1.2

- Add `tools/build_msrs_multigranularity.py` for official indexed MSRS labels.
- Generate class-specific semantic masks, connected-component pseudo-instance
  masks, spatial instructions, mismatched negative instructions, and JSONL rows.
- Validate label IDs and IR/VIS/label spatial alignment before writing output.

## 0.1.1

- Rename the decoder's `half` child module to `half_branch`; the former name
  collided with `torch.nn.Module.half()` and caused a bound-method call.
- Pin NumPy to `>=1.24,<2` for compatibility with PyTorch Windows wheels built
  against the NumPy 1.x ABI.
