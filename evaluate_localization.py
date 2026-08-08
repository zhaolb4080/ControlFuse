import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def find_prediction(directory: Path, name: str):
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = directory / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def summarize(rows):
    if not rows:
        return None
    keys = ("IoU", "F1", "Precision", "Recall")
    return {"count": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in keys}}


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic/instance instruction localization.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predicted-dir", required=True)
    parser.add_argument("--output", default="localization_metrics.csv")
    parser.add_argument("--summary", default="localization_summary.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")

    manifest = Path(args.manifest)
    predicted_dir = Path(args.predicted_dir)
    rows = []
    missing = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("granularity") not in {"semantic", "instance"} or not item.get("mask"):
                continue
            name = item.get("name", Path(item["visible"]).stem)
            predicted_path = find_prediction(predicted_dir, name)
            if predicted_path is None:
                missing.append(name)
                continue
            ground_truth_path = resolve_path(item["mask"], manifest.parent)
            with Image.open(ground_truth_path) as image:
                ground_truth = np.asarray(image.convert("L")) > 127
            with Image.open(predicted_path) as image:
                prediction_image = image.convert("L")
                if prediction_image.size != (ground_truth.shape[1], ground_truth.shape[0]):
                    prediction_image = prediction_image.resize(
                        (ground_truth.shape[1], ground_truth.shape[0]), Image.Resampling.BILINEAR
                    )
                prediction = np.asarray(prediction_image, dtype=np.float32) / 255.0 >= args.threshold

            true_positive = int(np.logical_and(prediction, ground_truth).sum())
            false_positive = int(np.logical_and(prediction, ~ground_truth).sum())
            false_negative = int(np.logical_and(~prediction, ground_truth).sum())
            iou = true_positive / max(true_positive + false_positive + false_negative, 1)
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            f1 = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
            rows.append(
                {
                    "name": name,
                    "granularity": item["granularity"],
                    "class_name": item.get("class_name", "unknown"),
                    "IoU": iou,
                    "F1": f1,
                    "Precision": precision,
                    "Recall": recall,
                }
            )

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} predicted masks; first entries: {preview}")
    if not rows:
        raise RuntimeError("No semantic or instance rows were evaluated.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["granularity"]].append(row)
    summary = {
        "threshold": args.threshold,
        "overall": summarize(rows),
        "semantic": summarize(grouped["semantic"]),
        "instance": summarize(grouped["instance"]),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
