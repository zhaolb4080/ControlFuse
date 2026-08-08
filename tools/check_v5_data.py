import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def semantic_key(item: dict):
    source = str(item.get("source_name", item.get("visible", "")))
    return source, str(item.get("class_name", "unknown"))


def main():
    parser = argparse.ArgumentParser(description="Validate v5 semantic/instance mask linkage.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    root = manifest.parent
    with manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {manifest}")

    counts = Counter(str(row.get("granularity", "global")) for row in rows)
    semantic_masks = {
        semantic_key(row): row["mask"]
        for row in rows
        if row.get("granularity") == "semantic" and row.get("mask")
    }
    instances = [row for row in rows if row.get("granularity") == "instance"]
    matched = 0
    with_distractors = 0
    missing = []
    for row in instances:
        semantic_value = row.get("semantic_mask") or semantic_masks.get(semantic_key(row))
        if not semantic_value:
            missing.append(str(row.get("name", "unknown")))
            continue
        semantic_path = resolve(root, semantic_value)
        instance_path = resolve(root, row["mask"])
        with Image.open(semantic_path) as source:
            semantic = np.asarray(source.convert("L"), dtype=np.uint8) > 127
        with Image.open(instance_path) as source:
            instance = np.asarray(source.convert("L"), dtype=np.uint8) > 127
        if semantic.shape != instance.shape:
            raise ValueError(
                f"Mask size mismatch for {row.get('name', 'unknown')}: "
                f"semantic={semantic.shape}, instance={instance.shape}"
            )
        if np.any(instance & ~semantic):
            raise ValueError(
                f"Instance mask is not contained in its semantic mask: "
                f"{row.get('name', 'unknown')}"
            )
        matched += 1
        with_distractors += int(np.any(semantic & ~instance))

    print(f"rows={len(rows)} global={counts['global']} semantic={counts['semantic']} instance={counts['instance']}")
    print(f"instance_semantic_matches={matched}/{len(instances)}")
    print(f"instances_with_same_class_distractors={with_distractors}/{len(instances)}")
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Missing semantic masks for {len(missing)} instance rows; first rows: {preview}"
        )
    print("v5 data preflight passed")


if __name__ == "__main__":
    main()
