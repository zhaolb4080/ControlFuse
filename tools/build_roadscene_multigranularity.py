"""Build ControlFuse-v5 RoadScene manifests with Grounding DINO and SAM pseudo-labels."""

import argparse
import hashlib
import inspect
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

CLASSES = {
    0: {
        "key": "person",
        "singular": "pedestrian",
        "plural": "pedestrians",
        "prompt": "person",
        "aliases": ("person", "people", "pedestrian", "pedestrians"),
    },
    1: {
        "key": "car",
        "singular": "car",
        "plural": "cars",
        "prompt": "car",
        "aliases": ("car", "cars", "automobile", "vehicle"),
    },
    2: {
        "key": "bus",
        "singular": "bus",
        "plural": "buses",
        "prompt": "bus",
        "aliases": ("bus", "buses"),
    },
    3: {
        "key": "motorcycle",
        "singular": "motorcycle",
        "plural": "motorcycles",
        "prompt": "motorcycle",
        "aliases": ("motorcycle", "motorcycles", "motorbike", "motorbikes"),
    },
    4: {
        "key": "lamp",
        "singular": "street lamp",
        "plural": "street lamps",
        "prompt": "street lamp",
        "aliases": ("lamp", "lamps", "street lamp", "street lamps", "streetlight"),
    },
    5: {
        "key": "truck",
        "singular": "truck",
        "plural": "trucks",
        "prompt": "truck",
        "aliases": ("truck", "trucks", "lorry"),
    },
}


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


ALIASES = sorted(
    (
        (normalize_label(alias), class_id)
        for class_id, info in CLASSES.items()
        for alias in (info["key"], info["prompt"], *info["aliases"])
    ),
    key=lambda item: len(item[0]),
    reverse=True,
)


def map_label(value):
    normalized = normalize_label(value)
    for alias, class_id in ALIASES:
        if normalized == alias or alias in normalized:
            return class_id
    return None


def index_files(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    indexed = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in indexed:
            raise ValueError(
                f"Duplicate stem {path.stem!r} in {directory}: "
                f"{indexed[path.stem].name}, {path.name}"
            )
        indexed[path.stem] = path
    return indexed


def preview(values, limit=5):
    values = sorted(values)
    return ", ".join(values[:limit]) + (" ..." if len(values) > limit else "")


def matching_names(visible, infrared):
    all_names = set(visible) | set(infrared)
    common = set(visible) & set(infrared)
    if all_names != common:
        problems = []
        missing_vis = all_names - set(visible)
        missing_ir = all_names - set(infrared)
        if missing_vis:
            problems.append(f"missing from visible: {preview(missing_vis)}")
        if missing_ir:
            problems.append(f"missing from infrared: {preview(missing_ir)}")
        raise ValueError("RoadScene stems do not match; " + "; ".join(problems))
    if not common:
        raise RuntimeError("No matching stems found across visible and infrared.")
    return sorted(common)


def atomic_write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    os.replace(temporary, path)


def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(temporary)
    os.replace(temporary, path)


def load_mask(path: Path, width: int, height: int):
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 127
    if mask.shape != (height, width):
        raise ValueError(
            f"Cached mask size mismatch in {path}: mask={mask.shape[::-1]}, image={(width, height)}"
        )
    return mask


def spatial_phrase(center_x: float, center_y: float, width: int, height: int) -> str:
    horizontal = "left" if center_x < width / 3 else "right" if center_x > 2 * width / 3 else "center"
    vertical = "upper" if center_y < height / 3 else "lower" if center_y > 2 * height / 3 else "middle"
    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return horizontal
    if horizontal == "center":
        return f"{vertical}-center"
    return f"{vertical}-{horizontal}"


def opposite_phrase(phrase: str) -> str:
    if phrase == "center":
        return "upper-left"
    swap = {"upper": "lower", "lower": "upper", "left": "right", "right": "left"}
    return "-".join(swap.get(part, part) for part in phrase.split("-"))


def negative_semantic_instruction(class_id: int, present_ids):
    absent = [candidate for candidate in CLASSES if candidate not in present_ids]
    if absent:
        selected = absent[class_id % len(absent)]
        return f"Highlight all {CLASSES[selected]['plural']} in the fused image."
    return f"Suppress all {CLASSES[class_id]['plural']} in the fused image."


def sanitize_box(box, width: int, height: int):
    if len(box) != 4:
        raise ValueError(f"Expected a four-value box, received {box!r}")
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def box_area(box):
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def box_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    union = box_area(first) + box_area(second) - intersection
    return intersection / max(union, 1e-8)


def classwise_nms(detections, threshold: float):
    selected = []
    for class_id in sorted(CLASSES):
        candidates = [item for item in detections if item["class_id"] == class_id]
        candidates.sort(
            key=lambda item: (-item["score"], item["bbox"][0], item["bbox"][1], item["source"])
        )
        kept = []
        for candidate in candidates:
            if all(box_iou(candidate["bbox"], other["bbox"]) <= threshold for other in kept):
                kept.append(candidate)
        selected.extend(kept)
    selected.sort(
        key=lambda item: (
            item["class_id"],
            -item["score"],
            item["bbox"][0],
            item["bbox"][1],
            item["bbox"][2],
            item["bbox"][3],
            item["source"],
        )
    )
    for object_index, item in enumerate(selected, start=1):
        item["object_index"] = object_index
        x1, y1, x2, y2 = item["bbox"]
        item["center_x"] = (x1 + x2) / 2.0
        item["center_y"] = (y1 + y2) / 2.0
        item["box_area"] = box_area(item["bbox"])
    return selected


class GroundingDinoDetector:
    def __init__(self, model_name: str, device: str, local_files_only: bool = False):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Grounding DINO requires torch and a Transformers release containing "
                "AutoModelForZeroShotObjectDetection. Install this project's requirements."
            ) from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({device}) but torch.cuda.is_available() is false")
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            use_safetensors=True,
        ).to(device)
        self.model.eval()
        self.prompt = ". ".join(CLASSES[class_id]["prompt"] for class_id in sorted(CLASSES)) + "."

    def __call__(self, image: Image.Image, box_threshold: float, text_threshold: float, source: str):
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        post_process = self.processor.post_process_grounded_object_detection
        parameters = inspect.signature(post_process).parameters
        kwargs = {
            "input_ids": inputs.input_ids,
            "text_threshold": text_threshold,
            "target_sizes": [image.size[::-1]],
        }
        if "threshold" in parameters:
            kwargs["threshold"] = box_threshold
        else:
            kwargs["box_threshold"] = box_threshold
        result = post_process(outputs, **kwargs)[0]
        labels = result.get("text_labels", result.get("labels"))
        if labels is None:
            raise RuntimeError("Grounding DINO post-processing returned no labels")
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        scores = result["scores"].detach().float().cpu().tolist()
        boxes = result["boxes"].detach().float().cpu().tolist()
        detections = []
        unknown = Counter()
        for label, score, box in zip(labels, scores, boxes):
            if not isinstance(label, str):
                raise RuntimeError(
                    "Grounding DINO returned numeric labels rather than text labels. "
                    "Upgrade Transformers so grounded text labels are available."
                )
            class_id = map_label(label)
            if class_id is None:
                unknown[label] += 1
                continue
            sanitized = sanitize_box(box, image.width, image.height)
            if sanitized is None:
                continue
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": CLASSES[class_id]["key"],
                    "score": float(score),
                    "bbox": sanitized,
                    "source": source,
                    "raw_label": label,
                }
            )
        return detections, unknown


def normalize_external_detections(path: Path, width: int, height: int):
    if not path.is_file():
        raise FileNotFoundError(f"External detection file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("detections", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"External detections must be a list or {{'detections': [...]}}: {path}")
    detections = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Detection {index} in {path} is not an object")
        class_value = entry.get("class_name", entry.get("label"))
        class_id = map_label(class_value)
        if class_id is None:
            raise ValueError(f"Unknown class {class_value!r} in {path}, detection {index}")
        box = sanitize_box(entry.get("bbox", entry.get("box", [])), width, height)
        if box is None:
            raise ValueError(f"Invalid box in {path}, detection {index}")
        detections.append(
            {
                "class_id": class_id,
                "class_name": CLASSES[class_id]["key"],
                "score": float(entry.get("score", 1.0)),
                "bbox": box,
                "source": str(entry.get("source", "external")),
                "raw_label": str(class_value),
            }
        )
    return detections


def serialize_detection(item):
    return {
        "object_index": int(item["object_index"]),
        "class_id": int(item["class_id"]),
        "class_name": CLASSES[item["class_id"]]["key"],
        "score": float(item["score"]),
        "bbox": [float(value) for value in item["bbox"]],
        "source": item["source"],
        "raw_label": item.get("raw_label", CLASSES[item["class_id"]]["key"]),
    }


def load_detection_cache(path: Path, width: int, height: int, nms_threshold: float):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("width"), payload.get("height")) != (width, height):
        raise ValueError(
            f"Cached detection size mismatch in {path}: "
            f"cache={(payload.get('width'), payload.get('height'))}, image={(width, height)}"
        )
    normalized = []
    for entry in payload.get("detections", []):
        class_id = int(entry["class_id"])
        if class_id not in CLASSES:
            raise ValueError(f"Unknown cached class_id={class_id} in {path}")
        box = sanitize_box(entry["bbox"], width, height)
        if box is None:
            raise ValueError(f"Invalid cached box in {path}: {entry['bbox']}")
        normalized.append(
            {
                "class_id": class_id,
                "class_name": CLASSES[class_id]["key"],
                "score": float(entry["score"]),
                "bbox": box,
                "source": str(entry.get("source", "visible")),
                "raw_label": str(entry.get("raw_label", CLASSES[class_id]["key"])),
            }
        )
    return classwise_nms(normalized, nms_threshold)


def directory_fingerprint(directory: Path):
    digest = hashlib.sha256()
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".json")
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def box_mask(obj, width: int, height: int):
    x1, y1, x2, y2 = obj["bbox"]
    x1, y1 = int(np.floor(x1)), int(np.floor(y1))
    x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def build_sam_predictor(checkpoint: Path, model_type: str, device: str):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "segment_anything is required for --mask-mode sam. Install it with: "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc
    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model type {model_type!r}; choose vit_b, vit_l, or vit_h")
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return SamPredictor(model)


def sam_mask(predictor, obj, width, height, min_score, min_area_ratio):
    input_box = np.asarray(obj["bbox"], dtype=np.float32)
    masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
    region = box_mask(obj, width, height)
    candidates = []
    for mask, score in zip(masks, scores):
        clipped = np.asarray(mask, dtype=bool) & region
        area = int(clipped.sum())
        ratio = area / max(obj["box_area"], 1.0)
        if float(score) >= min_score and ratio >= min_area_ratio:
            candidates.append((float(score), area, clipped))
    if not candidates:
        return None, None
    score, _, mask = max(candidates, key=lambda item: (item[0], item[1]))
    return mask, score


def generation_signature(args, external_directory):
    signature = {
        "schema": "roadscene-pseudo-v1",
        "classes": [CLASSES[index]["key"] for index in sorted(CLASSES)],
        "detector_model": args.detector_model,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "nms_iou": args.nms_iou,
        "detect_infrared": args.detect_infrared,
        "mask_mode": args.mask_mode,
        "sam_image_source": args.sam_image_source,
        "sam_fallback": args.sam_fallback,
        "sam_min_score": args.sam_min_score,
        "sam_min_area_ratio": args.sam_min_area_ratio,
    }
    if external_directory is not None:
        signature["external_detections_dir"] = str(external_directory)
        signature["external_detections_fingerprint"] = directory_fingerprint(external_directory)
    if args.mask_mode == "sam":
        checkpoint = Path(args.sam_checkpoint).resolve()
        signature.update(
            {
                "sam_model_type": args.sam_model_type,
                "sam_checkpoint": str(checkpoint),
                "sam_checkpoint_size": checkpoint.stat().st_size if checkpoint.is_file() else None,
            }
        )
    return signature


def prepare_cache(mask_root: Path, signature, overwrite: bool):
    config_path = mask_root / "generation.json"
    existing_artifacts = False
    if mask_root.exists():
        existing_artifacts = any(mask_root.rglob("*.png")) or any(mask_root.rglob("*.json"))
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != signature and not overwrite:
            raise ValueError(
                f"Pseudo-label settings differ from {config_path}. "
                "Use a new --mask-dir or pass --overwrite-masks intentionally."
            )
    elif existing_artifacts and not overwrite:
        raise ValueError(
            f"Found pseudo-label artifacts without generation metadata under {mask_root}. "
            "Use a new --mask-dir or pass --overwrite-masks intentionally."
        )
    mask_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(config_path, json.dumps(signature, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate ControlFuse-v5 RoadScene semantic/instance pseudo-labels with "
            "Grounding DINO boxes and SAM masks."
        )
    )
    parser.add_argument(
        "--split-root",
        required=True,
        help="RoadScene train or test split containing visible/ and infrared/.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL manifest.")
    parser.add_argument(
        "--mask-dir",
        help="Pseudo-label cache/output directory. Default: <output_stem>_masks beside manifest.",
    )
    parser.add_argument(
        "--detector-model", default="IDEA-Research/grounding-dino-base"
    )
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument(
        "--detect-infrared",
        action="store_true",
        help="Also run Grounding DINO on infrared converted to RGB, then class-wise NMS-merge boxes.",
    )
    parser.add_argument(
        "--external-detections-dir",
        help=(
            "Optional directory containing <stem>.json detection files. When supplied, "
            "Grounding DINO is not loaded; useful for audited/manual boxes and offline tests."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mask-mode", choices=("sam", "box"), default="sam")
    parser.add_argument("--sam-checkpoint", help="Path to an official SAM checkpoint.")
    parser.add_argument("--sam-model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-image-source", choices=("visible", "infrared"), default="visible")
    parser.add_argument("--sam-min-score", type=float, default=0.0)
    parser.add_argument("--sam-min-area-ratio", type=float, default=0.02)
    parser.add_argument("--sam-fallback", choices=("box", "skip", "error"), default="box")
    parser.add_argument("--min-instance-area", type=int, default=64)
    parser.add_argument("--max-instances", type=int, default=5)
    parser.add_argument("--include-global", action="store_true")
    parser.add_argument(
        "--overwrite-masks",
        action="store_true",
        help="Regenerate both cached detections and masks with the current settings.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--limit", type=int, help="Debug only: process the first N source images.")
    args = parser.parse_args()

    if args.mask_mode == "sam" and not args.sam_checkpoint:
        parser.error("--sam-checkpoint is required when --mask-mode sam")
    if args.mask_mode == "sam" and not Path(args.sam_checkpoint).is_file():
        parser.error(f"--sam-checkpoint does not exist: {args.sam_checkpoint}")
    if not 0.0 <= args.box_threshold <= 1.0:
        parser.error("--box-threshold must be in [0, 1]")
    if not 0.0 <= args.text_threshold <= 1.0:
        parser.error("--text-threshold must be in [0, 1]")
    if not 0.0 <= args.nms_iou <= 1.0:
        parser.error("--nms-iou must be in [0, 1]")
    if not 0.0 <= args.sam_min_score <= 1.0:
        parser.error("--sam-min-score must be in [0, 1]")
    if not 0.0 <= args.sam_min_area_ratio <= 1.0:
        parser.error("--sam-min-area-ratio must be in [0, 1]")
    if args.min_instance_area < 1:
        parser.error("--min-instance-area must be at least 1")
    if args.max_instances < 0:
        parser.error("--max-instances cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    split_root = Path(args.split_root).resolve()
    visible = index_files(split_root / "visible")
    infrared = index_files(split_root / "infrared")
    names = matching_names(visible, infrared)
    if args.limit is not None:
        names = names[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_root = Path(args.mask_dir) if args.mask_dir else output.parent / f"{output.stem}_masks"
    mask_root = mask_root.resolve()
    detection_dir = mask_root / "detections"
    instance_dir = mask_root / "instance"
    semantic_dir = mask_root / "semantic"
    external_directory = (
        Path(args.external_detections_dir).resolve() if args.external_detections_dir else None
    )
    if external_directory is not None and not external_directory.is_dir():
        parser.error(f"--external-detections-dir does not exist: {external_directory}")
    signature = generation_signature(args, external_directory)
    prepare_cache(mask_root, signature, args.overwrite_masks)
    detection_dir.mkdir(parents=True, exist_ok=True)

    detector = None
    if external_directory is None:
        detector = GroundingDinoDetector(
            args.detector_model, args.device, local_files_only=args.local_files_only
        )
    predictor = None
    if args.mask_mode == "sam":
        predictor = build_sam_predictor(
            Path(args.sam_checkpoint).resolve(), args.sam_model_type, args.device
        )

    rows = []
    counts = Counter()
    class_counts = Counter()
    unknown_counts = Counter()

    for source_index, name in enumerate(names, start=1):
        with Image.open(visible[name]) as source:
            visible_image = source.convert("RGB")
            visible_array = np.asarray(visible_image, dtype=np.uint8)
            width, height = visible_image.size
        with Image.open(infrared[name]) as source:
            infrared_image = source.convert("RGB")
            if infrared_image.size != (width, height):
                raise ValueError(
                    f"IR/VIS size mismatch for {name}: "
                    f"ir={infrared_image.size}, vis={(width, height)}"
                )
            infrared_array = np.asarray(infrared_image, dtype=np.uint8)

        detection_path = detection_dir / f"{name}.json"
        if detection_path.is_file() and not args.overwrite_masks:
            objects = load_detection_cache(detection_path, width, height, args.nms_iou)
            counts["detections_reused"] += len(objects)
        else:
            if external_directory is not None:
                detections = normalize_external_detections(
                    external_directory / f"{name}.json", width, height
                )
            else:
                detections, unknown = detector(
                    visible_image, args.box_threshold, args.text_threshold, "visible"
                )
                unknown_counts.update(unknown)
                if args.detect_infrared:
                    ir_detections, ir_unknown = detector(
                        infrared_image, args.box_threshold, args.text_threshold, "infrared"
                    )
                    detections.extend(ir_detections)
                    unknown_counts.update(ir_unknown)
            objects = classwise_nms(detections, args.nms_iou)
            atomic_write_text(
                detection_path,
                json.dumps(
                    {
                        "source_name": name,
                        "width": width,
                        "height": height,
                        "detections": [serialize_detection(item) for item in objects],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            counts["detections_generated"] += len(objects)

        for obj in objects:
            class_counts[f"detection_{CLASSES[obj['class_id']]['key']}"] += 1
        if not objects:
            counts["images_without_detections"] += 1

        common = {
            "source_name": name,
            "visible": str(visible[name].resolve()),
            "infrared": str(infrared[name].resolve()),
        }
        if args.include_global:
            rows.append(
                {
                    **common,
                    "name": f"{name}_global",
                    "instruction": "Enhance the entire scene.",
                    "negative_instruction": "Suppress the entire scene.",
                    "granularity": "global",
                }
            )
            counts["global"] += 1

        object_paths = {}
        missing_objects = []
        for obj in objects:
            info = CLASSES[obj["class_id"]]
            path = instance_dir / f"{name}_instance_{info['key']}_{obj['object_index']:03d}.png"
            object_paths[obj["object_index"]] = path
            if args.overwrite_masks or not path.is_file():
                missing_objects.append(obj)
        if predictor is not None and missing_objects:
            predictor.set_image(
                visible_array if args.sam_image_source == "visible" else infrared_array
            )

        generated_objects = []
        for obj in objects:
            path = object_paths[obj["object_index"]]
            if path.is_file() and not args.overwrite_masks:
                mask = load_mask(path, width, height)
                counts["masks_reused"] += 1
            elif args.mask_mode == "box":
                mask = box_mask(obj, width, height)
                save_mask(mask, path)
                counts["box_masks"] += 1
            else:
                mask, _ = sam_mask(
                    predictor,
                    obj,
                    width,
                    height,
                    args.sam_min_score,
                    args.sam_min_area_ratio,
                )
                if mask is None:
                    if args.sam_fallback == "error":
                        raise RuntimeError(
                            f"SAM produced no valid mask for {name}, object {obj['object_index']}"
                        )
                    if args.sam_fallback == "skip":
                        counts["sam_skipped"] += 1
                        continue
                    mask = box_mask(obj, width, height)
                    counts["sam_box_fallback"] += 1
                else:
                    counts["sam_masks"] += 1
                save_mask(mask, path)
            area = int(mask.sum())
            if area == 0:
                raise ValueError(f"Empty cached/generated mask: {path}")
            generated_objects.append({**obj, "mask": mask, "mask_path": path, "area": area})

        by_class = defaultdict(list)
        for obj in generated_objects:
            by_class[obj["class_id"]].append(obj)
        present_ids = set(by_class)
        semantic_paths = {}
        for class_id in sorted(present_ids):
            info = CLASSES[class_id]
            semantic = np.logical_or.reduce([obj["mask"] for obj in by_class[class_id]])
            semantic_path = semantic_dir / f"{name}_semantic_{info['key']}.png"
            save_mask(semantic, semantic_path)
            semantic_paths[class_id] = semantic_path
            rows.append(
                {
                    **common,
                    "name": f"{name}_semantic_{info['key']}",
                    "mask": str(semantic_path.resolve()),
                    "instruction": f"Highlight all {info['plural']} in the fused image.",
                    "negative_instruction": negative_semantic_instruction(class_id, present_ids),
                    "granularity": "semantic",
                    "class_id": class_id,
                    "class_name": info["key"],
                    "pseudo_label": True,
                }
            )
            counts["semantic"] += 1
            class_counts[f"semantic_{info['key']}"] += 1

        candidates = [obj for obj in generated_objects if obj["area"] >= args.min_instance_area]
        candidates.sort(key=lambda obj: (-obj["area"], -obj["score"], obj["object_index"]))
        for selection_index, obj in enumerate(candidates[: args.max_instances], start=1):
            class_id = obj["class_id"]
            info = CLASSES[class_id]
            phrase = spatial_phrase(obj["center_x"], obj["center_y"], width, height)
            absent = [candidate for candidate in CLASSES if candidate not in present_ids]
            negative_class = (
                CLASSES[absent[selection_index % len(absent)]]["singular"]
                if absent
                else info["singular"]
            )
            rows.append(
                {
                    **common,
                    "name": f"{name}_instance_{info['key']}_{obj['object_index']:03d}",
                    "mask": str(obj["mask_path"].resolve()),
                    "semantic_mask": str(semantic_paths[class_id].resolve()),
                    "instruction": f"Emphasize the {info['singular']} in the {phrase} region.",
                    "negative_instruction": (
                        f"Emphasize the {negative_class} in the {opposite_phrase(phrase)} region."
                    ),
                    "granularity": "instance",
                    "class_id": class_id,
                    "class_name": info["key"],
                    "instance_area": obj["area"],
                    "detection_score": obj["score"],
                    "bbox": [float(value) for value in obj["bbox"]],
                    "pseudo_label": True,
                }
            )
            counts["instance"] += 1
            class_counts[f"instance_{info['key']}"] += 1

        if args.log_every and (source_index % args.log_every == 0 or source_index == len(names)):
            print(
                f"processed {source_index}/{len(names)} source images; rows={len(rows)} "
                f"semantic={counts['semantic']} instance={counts['instance']}"
            )

    atomic_write_text(
        output,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary = {
        "dataset": "RoadScene",
        "source_images": len(names),
        "rows": len(rows),
        "global": counts["global"],
        "semantic": counts["semantic"],
        "instance": counts["instance"],
        "images_without_detections": counts["images_without_detections"],
        "counts": dict(sorted(counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "generation": signature,
    }
    atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(f"matched source images: {len(names)}")
    print(
        f"wrote {len(rows)} rows to {output} "
        f"(global={counts['global']}, semantic={counts['semantic']}, instance={counts['instance']})"
    )
    print(f"saved pseudo-labels to {mask_root}")
    print(f"summary: {summary_path}")
    print(
        "mask generation: "
        f"sam={counts['sam_masks']} fallback_box={counts['sam_box_fallback']} "
        f"box={counts['box_masks']} skipped={counts['sam_skipped']} reused={counts['masks_reused']}"
    )
    if unknown_counts:
        print(
            "warning: ignored detector labels that did not map to the six classes: "
            + ", ".join(f"{key}={value}" for key, value in sorted(unknown_counts.items()))
        )


if __name__ == "__main__":
    main()
