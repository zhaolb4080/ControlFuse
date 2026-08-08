import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def polygon_mask(width, height, segmentation):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon) >= 6:
            draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=255)
    return np.asarray(mask, dtype=np.uint8)


def position_word(box, width):
    center = box[0] + box[2] / 2
    return "leftmost" if center < width / 3 else "rightmost" if center > 2 * width / 3 else "central"


def main():
    parser = argparse.ArgumentParser(description="Create global/semantic/instance rows from COCO polygons.")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--visible-dir", required=True)
    parser.add_argument("--infrared-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-instances", type=int, default=5)
    args = parser.parse_args()
    with Path(args.coco).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    output = Path(args.output)
    mask_dir = output.parent / f"{output.stem}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id, info in images.items():
        name = Path(info["file_name"]).stem
        common = {"name": name, "visible": str((Path(args.visible_dir) / info["file_name"]).resolve()), "infrared": str((Path(args.infrared_dir) / info["file_name"]).resolve())}
        global_mask = np.full((info["height"], info["width"]), 255, np.uint8)
        global_path = mask_dir / f"{name}_global.png"
        Image.fromarray(global_mask).save(global_path)
        rows.append({**common, "mask": str(global_path.resolve()), "instruction": "Enhance the entire scene.", "negative_instruction": "Suppress the entire scene.", "granularity": "global"})
        by_category = defaultdict(list)
        valid = []
        for ann in annotations[image_id]:
            segmentation = ann.get("segmentation", [])
            if not isinstance(segmentation, list):
                continue
            mask = polygon_mask(info["width"], info["height"], segmentation)
            if mask.any():
                by_category[ann["category_id"]].append(mask)
                valid.append((ann, mask))
        for category_id, masks in by_category.items():
            category = categories[category_id]
            path = mask_dir / f"{name}_semantic_{category_id}.png"
            Image.fromarray(np.maximum.reduce(masks)).save(path)
            rows.append({**common, "mask": str(path.resolve()), "instruction": f"Highlight all {category} objects.", "negative_instruction": f"Suppress all {category} objects.", "granularity": "semantic"})
        valid.sort(key=lambda pair: pair[0].get("area", 0), reverse=True)
        for index, (ann, mask) in enumerate(valid[: args.max_instances]):
            category = categories[ann["category_id"]]
            position = position_word(ann["bbox"], info["width"])
            path = mask_dir / f"{name}_instance_{ann['id']}.png"
            Image.fromarray(mask).save(path)
            rows.append({**common, "mask": str(path.resolve()), "instruction": f"Emphasize the {position} {category}.", "negative_instruction": f"Emphasize an absent {category} on the opposite side.", "granularity": "instance"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} multi-granularity samples to {output}")


if __name__ == "__main__":
    main()
