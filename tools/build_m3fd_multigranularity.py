"""Build ControlFuse v5 multi-granularity data from M3FD Pascal VOC boxes."""

import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET
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
        "aliases": ("people", "person", "persons", "pedestrian", "pedestrians"),
    },
    1: {
        "key": "car",
        "singular": "car",
        "plural": "cars",
        "aliases": ("car", "cars"),
    },
    2: {
        "key": "bus",
        "singular": "bus",
        "plural": "buses",
        "aliases": ("bus", "buses"),
    },
    3: {
        "key": "motorcycle",
        "singular": "motorcycle",
        "plural": "motorcycles",
        "aliases": ("motorcycle", "motorcycles", "motorbike", "motorbikes", "motor"),
    },
    4: {
        "key": "lamp",
        "singular": "street lamp",
        "plural": "street lamps",
        "aliases": ("lamp", "lamps", "streetlamp", "streetlamps", "street light", "street lights"),
    },
    5: {
        "key": "truck",
        "singular": "truck",
        "plural": "trucks",
        "aliases": ("truck", "trucks"),
    },
}


def normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


ALIAS_TO_CLASS = {
    normalized_label(alias): class_id
    for class_id, info in CLASSES.items()
    for alias in info["aliases"]
}


def index_files(directory: Path, extensions):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    indexed = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in extensions:
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


def matching_names(visible, infrared, annotations):
    all_names = set(visible) | set(infrared) | set(annotations)
    common = set(visible) & set(infrared) & set(annotations)
    if all_names != common:
        problems = []
        for label, indexed in (("Vis", visible), ("Ir", infrared), ("Annotation", annotations)):
            missing = all_names - set(indexed)
            if missing:
                problems.append(f"missing from {label}: {preview(missing)}")
        raise ValueError("M3FD stems do not match; " + "; ".join(problems))
    if not common:
        raise RuntimeError("No matching stems found across Vis, Ir, and Annotation.")
    return sorted(common)


def atomic_write_text(path: Path, content: str):
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


def parse_float(node, field: str, annotation_path: Path):
    value = node.findtext(field)
    if value is None:
        raise ValueError(f"Missing <{field}> in {annotation_path}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid <{field}>={value!r} in {annotation_path}") from exc


def parse_annotation(path: Path, width: int, height: int, ignore_unknown: bool):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid Pascal VOC XML: {path}: {exc}") from exc
    if root.tag != "annotation":
        raise ValueError(f"Unexpected XML root in {path}: {root.tag!r}")

    size = root.find("size")
    if size is not None:
        xml_width = int(float(size.findtext("width", width)))
        xml_height = int(float(size.findtext("height", height)))
        if (xml_width, xml_height) != (width, height):
            raise ValueError(
                f"Annotation/image size mismatch for {path}: "
                f"xml={(xml_width, xml_height)}, image={(width, height)}"
            )

    objects = []
    unknown = Counter()
    for object_index, node in enumerate(root.findall("object"), start=1):
        raw_name = (node.findtext("name") or "").strip()
        class_id = ALIAS_TO_CLASS.get(normalized_label(raw_name))
        if class_id is None:
            unknown[raw_name or "<empty>"] += 1
            if ignore_unknown:
                continue
            raise ValueError(
                f"Unknown M3FD class {raw_name!r} in {path}. "
                f"Known classes: {', '.join(info['key'] for info in CLASSES.values())}"
            )
        box = node.find("bndbox")
        if box is None:
            raise ValueError(f"Missing <bndbox> for object {object_index} in {path}")

        # Pascal VOC coordinates are conventionally 1-based and inclusive.
        xmin = parse_float(box, "xmin", path)
        ymin = parse_float(box, "ymin", path)
        xmax = parse_float(box, "xmax", path)
        ymax = parse_float(box, "ymax", path)
        x1 = max(0, min(width, int(math.floor(xmin)) - 1))
        y1 = max(0, min(height, int(math.floor(ymin)) - 1))
        x2 = max(0, min(width, int(math.ceil(xmax))))
        y2 = max(0, min(height, int(math.ceil(ymax))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounding box in {path}, object {object_index}: "
                f"{(xmin, ymin, xmax, ymax)}"
            )
        objects.append(
            {
                "object_index": object_index,
                "class_id": class_id,
                "bbox": (x1, y1, x2, y2),
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
                "box_area": (x2 - x1) * (y2 - y1),
            }
        )
    return objects, unknown


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


def box_mask(obj, width: int, height: int):
    x1, y1, x2, y2 = obj["bbox"]
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
            "segment_anything is required for --mask-mode sam. Install the official package with: "
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
        ratio = area / max(obj["box_area"], 1)
        if float(score) >= min_score and ratio >= min_area_ratio:
            candidates.append((float(score), area, clipped))
    if not candidates:
        return None, None
    score, _, mask = max(candidates, key=lambda item: (item[0], item[1]))
    return mask, score


def generation_signature(args):
    signature = {
        "mask_mode": args.mask_mode,
        "image_source": args.sam_image_source,
        "fallback": args.sam_fallback,
        "sam_min_score": args.sam_min_score,
        "sam_min_area_ratio": args.sam_min_area_ratio,
    }
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


def prepare_mask_cache(mask_root: Path, signature, overwrite: bool):
    config_path = mask_root / "generation.json"
    existing_masks = any(mask_root.rglob("*.png")) if mask_root.exists() else False
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != signature and not overwrite:
            raise ValueError(
                f"Mask generation settings differ from {config_path}. "
                "Use a new --mask-dir or pass --overwrite-masks intentionally."
            )
    elif existing_masks and not overwrite:
        raise ValueError(
            f"Found masks without generation metadata under {mask_root}. "
            "Use a new --mask-dir or pass --overwrite-masks intentionally."
        )
    mask_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(config_path, json.dumps(signature, ensure_ascii=False, indent=2) + "\n")


def negative_semantic_instruction(class_id: int, present_ids):
    absent = [candidate for candidate in CLASSES if candidate not in present_ids]
    if absent:
        selected = absent[class_id % len(absent)]
        return f"Highlight all {CLASSES[selected]['plural']} in the fused image."
    return f"Suppress all {CLASSES[class_id]['plural']} in the fused image."


def main():
    parser = argparse.ArgumentParser(
        description="Generate ControlFuse v5 semantic/instance masks from M3FD VOC boxes."
    )
    parser.add_argument(
        "--split-root",
        required=True,
        help="M3FD train or test split containing Vis/, Ir/, and Annotation/.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL manifest.")
    parser.add_argument(
        "--mask-dir",
        help="Mask cache/output directory. Default: <output_stem>_masks beside the manifest.",
    )
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
    parser.add_argument("--ignore-unknown-classes", action="store_true")
    parser.add_argument("--overwrite-masks", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--limit", type=int, help="Debug only: process the first N source images.")
    args = parser.parse_args()

    if args.mask_mode == "sam" and not args.sam_checkpoint:
        parser.error("--sam-checkpoint is required when --mask-mode sam")
    if args.mask_mode == "sam" and not Path(args.sam_checkpoint).is_file():
        parser.error(f"--sam-checkpoint does not exist: {args.sam_checkpoint}")
    if args.min_instance_area < 1:
        parser.error("--min-instance-area must be at least 1")
    if args.max_instances < 0:
        parser.error("--max-instances cannot be negative")
    if not 0.0 <= args.sam_min_score <= 1.0:
        parser.error("--sam-min-score must be in [0, 1]")
    if not 0.0 <= args.sam_min_area_ratio <= 1.0:
        parser.error("--sam-min-area-ratio must be in [0, 1]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    split_root = Path(args.split_root).resolve()
    visible = index_files(split_root / "Vis", IMAGE_EXTENSIONS)
    infrared = index_files(split_root / "Ir", IMAGE_EXTENSIONS)
    annotations = index_files(split_root / "Annotation", (".xml",))
    names = matching_names(visible, infrared, annotations)
    if args.limit is not None:
        names = names[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_root = Path(args.mask_dir) if args.mask_dir else output.parent / f"{output.stem}_masks"
    mask_root = mask_root.resolve()
    instance_dir = mask_root / "instance"
    semantic_dir = mask_root / "semantic"
    signature = generation_signature(args)
    prepare_mask_cache(mask_root, signature, args.overwrite_masks)

    predictor = None
    if args.mask_mode == "sam":
        predictor = build_sam_predictor(
            Path(args.sam_checkpoint).resolve(), args.sam_model_type, args.device
        )

    rows = []
    counts = Counter()
    raw_class_counts = Counter()
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

        objects, unknown = parse_annotation(
            annotations[name], width, height, args.ignore_unknown_classes
        )
        unknown_counts.update(unknown)
        for obj in objects:
            raw_class_counts[CLASSES[obj["class_id"]]["key"]] += 1

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
        missing_mask_objects = []
        for obj in objects:
            info = CLASSES[obj["class_id"]]
            path = instance_dir / f"{name}_instance_{info['key']}_{obj['object_index']:03d}.png"
            object_paths[obj["object_index"]] = path
            if args.overwrite_masks or not path.is_file():
                missing_mask_objects.append(obj)

        if predictor is not None and missing_mask_objects:
            predictor.set_image(
                visible_array if args.sam_image_source == "visible" else infrared_array
            )

        generated_objects = []
        for obj in objects:
            path = object_paths[obj["object_index"]]
            if path.is_file() and not args.overwrite_masks:
                mask = load_mask(path, width, height)
                counts["mask_reused"] += 1
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
                }
            )
            counts["semantic"] += 1
            counts[f"semantic_{info['key']}"] += 1

        candidates = [obj for obj in generated_objects if obj["area"] >= args.min_instance_area]
        candidates.sort(key=lambda obj: (-obj["area"], obj["object_index"]))
        for selection_index, obj in enumerate(candidates[: args.max_instances], start=1):
            class_id = obj["class_id"]
            info = CLASSES[class_id]
            phrase = spatial_phrase(obj["center_x"], obj["center_y"], width, height)
            absent = [candidate for candidate in CLASSES if candidate not in present_ids]
            if absent:
                negative_class = CLASSES[absent[selection_index % len(absent)]]["singular"]
            else:
                negative_class = info["singular"]
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
                    "bbox": list(obj["bbox"]),
                }
            )
            counts["instance"] += 1
            counts[f"instance_{info['key']}"] += 1

        if args.log_every and (source_index % args.log_every == 0 or source_index == len(names)):
            print(
                f"processed {source_index}/{len(names)} source images; "
                f"rows={len(rows)} semantic={counts['semantic']} instance={counts['instance']}"
            )

    manifest = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(output, manifest)
    print(f"matched source images: {len(names)}")
    print(
        f"wrote {len(rows)} rows to {output} "
        f"(global={counts['global']}, semantic={counts['semantic']}, instance={counts['instance']})"
    )
    print(f"saved masks to {mask_root}")
    print("VOC object counts: " + ", ".join(f"{key}={value}" for key, value in sorted(raw_class_counts.items())))
    print(
        "mask generation: "
        f"sam={counts['sam_masks']} fallback_box={counts['sam_box_fallback']} "
        f"box={counts['box_masks']} skipped={counts['sam_skipped']} reused={counts['mask_reused']}"
    )
    if unknown_counts:
        print(
            "warning: ignored unknown classes: "
            + ", ".join(f"{key}={value}" for key, value in sorted(unknown_counts.items()))
        )


if __name__ == "__main__":
    main()
