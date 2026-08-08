"""Build semantic- and instance-level ControlFuse samples from MSRS labels."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Official MSRS indexed-label mapping. ID 0 is unlabeled/background.
CLASSES = {
    1: {"key": "car", "singular": "car", "plural": "cars"},
    2: {"key": "person", "singular": "pedestrian", "plural": "pedestrians"},
    3: {"key": "bike", "singular": "bicycle", "plural": "bicycles"},
    4: {"key": "curve", "singular": "road curve", "plural": "road curves"},
    5: {"key": "car_stop", "singular": "car-stop sign", "plural": "car-stop signs"},
    6: {"key": "guardrail", "singular": "guardrail", "plural": "guardrails"},
    7: {"key": "color_cone", "singular": "traffic cone", "plural": "traffic cones"},
    8: {"key": "bump", "singular": "road bump", "plural": "road bumps"},
}

# Thing-like classes for which connected components are meaningful instances.
DEFAULT_INSTANCE_CLASSES = (1, 2, 3, 5, 7)


def index_images(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return {path.stem: path for path in directory.iterdir() if path.suffix.lower() in EXTENSIONS}


def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


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
    parts = phrase.split("-")
    return "-".join(swap.get(part, part) for part in parts)


def negative_semantic_instruction(class_id: int, present_ids: set) -> str:
    absent = [candidate for candidate in CLASSES if candidate not in present_ids]
    if absent:
        selected = absent[class_id % len(absent)]
        return f"Highlight all {CLASSES[selected]['plural']} in the fused image."
    return f"Suppress all {CLASSES[class_id]['plural']} in the fused image."


def instance_candidates(label_map: np.ndarray, instance_class_ids, min_area: int):
    structure = ndimage.generate_binary_structure(2, 2)
    candidates = []
    for class_id in instance_class_ids:
        component_map, component_count = ndimage.label(label_map == class_id, structure=structure)
        for component_id in range(1, component_count + 1):
            mask = component_map == component_id
            area = int(mask.sum())
            if area < min_area:
                continue
            ys, xs = np.nonzero(mask)
            candidates.append(
                {
                    "class_id": class_id,
                    "mask": mask,
                    "area": area,
                    "center_x": float(xs.mean()),
                    "center_y": float(ys.mean()),
                }
            )
    return sorted(candidates, key=lambda item: item["area"], reverse=True)


def parse_instance_classes(value: str):
    try:
        class_ids = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Instance classes must be comma-separated integer IDs.") from exc
    invalid = [class_id for class_id in class_ids if class_id not in CLASSES]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown MSRS class IDs: {invalid}")
    return class_ids


def main():
    parser = argparse.ArgumentParser(
        description="Generate MSRS semantic/instance masks and a ControlFuse JSONL manifest."
    )
    parser.add_argument(
        "--split-root",
        required=True,
        help="MSRS split containing vi/, ir/, and Segmentation_labels/.",
    )
    parser.add_argument("--output", required=True, help="Output semantic/instance JSONL path.")
    parser.add_argument(
        "--mask-dir",
        help="Output mask directory. Default: <output_stem>_masks beside the JSONL.",
    )
    parser.add_argument("--min-instance-area", type=int, default=20)
    parser.add_argument("--max-instances", type=int, default=5, help="Maximum instances per source image.")
    parser.add_argument(
        "--instance-classes",
        type=parse_instance_classes,
        default=DEFAULT_INSTANCE_CLASSES,
        help="Comma-separated class IDs. Default: 1,2,3,5,7.",
    )
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Also add one global row per image. Leave off when a global manifest already exists.",
    )
    args = parser.parse_args()

    if args.min_instance_area < 1:
        parser.error("--min-instance-area must be at least 1")
    if args.max_instances < 0:
        parser.error("--max-instances cannot be negative")

    split_root = Path(args.split_root)
    visible = index_images(split_root / "vi")
    infrared = index_images(split_root / "ir")
    labels = index_images(split_root / "Segmentation_labels")
    names = sorted(set(visible) & set(infrared) & set(labels))
    if not names:
        raise RuntimeError("No matching stems found across vi, ir, and Segmentation_labels.")

    unmatched = (set(visible) | set(infrared) | set(labels)) - set(names)
    if unmatched:
        print(f"warning: skipped {len(unmatched)} stems missing from one or more input folders")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_root = Path(args.mask_dir) if args.mask_dir else output.parent / f"{output.stem}_masks"
    semantic_dir = mask_root / "semantic"
    instance_dir = mask_root / "instance"
    rows = []
    counts = Counter()

    for name in names:
        with Image.open(labels[name]) as label_image:
            label_map = np.asarray(label_image)
        if label_map.ndim != 2:
            raise ValueError(
                f"{labels[name]} is not an indexed single-channel label. "
                "Use the original Segmentation_labels, not colorized visualization images."
            )
        invalid_ids = sorted(set(np.unique(label_map).tolist()) - {0, *CLASSES.keys()})
        if invalid_ids:
            raise ValueError(f"Unexpected class IDs in {labels[name]}: {invalid_ids}")

        with Image.open(visible[name]) as image:
            visible_size = image.size
        with Image.open(infrared[name]) as image:
            infrared_size = image.size
        label_size = (label_map.shape[1], label_map.shape[0])
        if visible_size != label_size or infrared_size != label_size:
            raise ValueError(
                f"Size mismatch for {name}: vi={visible_size}, ir={infrared_size}, label={label_size}"
            )

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

        present_ids = {class_id for class_id in CLASSES if np.any(label_map == class_id)}
        for class_id in sorted(present_ids):
            class_info = CLASSES[class_id]
            mask_path = semantic_dir / f"{name}_semantic_{class_info['key']}.png"
            save_mask(label_map == class_id, mask_path)
            rows.append(
                {
                    **common,
                    "name": f"{name}_semantic_{class_info['key']}",
                    "mask": str(mask_path.resolve()),
                    "instruction": f"Highlight all {class_info['plural']} in the fused image.",
                    "negative_instruction": negative_semantic_instruction(class_id, present_ids),
                    "granularity": "semantic",
                    "class_id": class_id,
                    "class_name": class_info["key"],
                }
            )
            counts["semantic"] += 1
            counts[f"semantic_{class_info['key']}"] += 1

        candidates = instance_candidates(label_map, args.instance_classes, args.min_instance_area)
        for instance_index, candidate in enumerate(candidates[: args.max_instances], start=1):
            class_id = candidate["class_id"]
            class_info = CLASSES[class_id]
            phrase = spatial_phrase(
                candidate["center_x"], candidate["center_y"], label_map.shape[1], label_map.shape[0]
            )
            mask_path = instance_dir / f"{name}_instance_{class_info['key']}_{instance_index:02d}.png"
            save_mask(candidate["mask"], mask_path)
            absent = [candidate_id for candidate_id in CLASSES if candidate_id not in present_ids]
            if absent:
                negative_class = CLASSES[absent[instance_index % len(absent)]]["singular"]
                negative_instruction = f"Emphasize the {negative_class} in the {opposite_phrase(phrase)} region."
            else:
                negative_instruction = (
                    f"Emphasize the {class_info['singular']} in the {opposite_phrase(phrase)} region."
                )
            rows.append(
                {
                    **common,
                    "name": f"{name}_instance_{class_info['key']}_{instance_index:02d}",
                    "mask": str(mask_path.resolve()),
                    "semantic_mask": str(
                        (semantic_dir / f"{name}_semantic_{class_info['key']}.png").resolve()
                    ),
                    "instruction": f"Emphasize the {class_info['singular']} in the {phrase} region.",
                    "negative_instruction": negative_instruction,
                    "granularity": "instance",
                    "class_id": class_id,
                    "class_name": class_info["key"],
                    "instance_area": candidate["area"],
                }
            )
            counts["instance"] += 1
            counts[f"instance_{class_info['key']}"] += 1

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"matched source images: {len(names)}")
    print(
        f"wrote {len(rows)} rows to {output} "
        f"(global={counts['global']}, semantic={counts['semantic']}, instance={counts['instance']})"
    )
    print(f"saved masks to {mask_root}")


if __name__ == "__main__":
    main()
