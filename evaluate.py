import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from controlfuse.metrics import all_metrics


def load(path: Path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def main():
    parser = argparse.ArgumentParser(description="Compute the six fusion metrics used in the paper.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fused-dir", required=True)
    parser.add_argument("--output", default="metrics.csv")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    fused_dir = Path(args.fused_dir)
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            name = item.get("name", Path(item["visible"]).stem)
            visible_path = Path(item["visible"])
            infrared_path = Path(item["infrared"])
            if not visible_path.is_absolute():
                visible_path = manifest.parent / visible_path
            if not infrared_path.is_absolute():
                infrared_path = manifest.parent / infrared_path
            fused_path = next((fused_dir / f"{name}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".bmp") if (fused_dir / f"{name}{suffix}").exists()), None)
            if fused_path is None:
                print(f"skip missing fused image: {name}")
                continue
            infrared = load(infrared_path)
            visible = load(visible_path)
            fused = load(fused_path)
            if infrared.shape != visible.shape or fused.shape != visible.shape:
                raise ValueError(
                    f"Image size mismatch for {name}: IR={infrared.shape}, "
                    f"VIS={visible.shape}, fused={fused.shape}"
                )
            metrics = all_metrics(infrared, visible, fused)
            rows.append({"name": name, **metrics})
    if not rows:
        raise RuntimeError("No matching fused images were evaluated.")
    mean = {"name": "MEAN", **{key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "name"}}
    rows.append(mean)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(" ".join(f"{key}={value:.4f}" for key, value in mean.items() if key != "name"))


if __name__ == "__main__":
    main()
