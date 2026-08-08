import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controlfuse.data import FusionManifestDataset


def main():
    random.seed(7)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        height, width = 48, 64
        base = np.arange(height * width, dtype=np.uint16).reshape(height, width) % 256
        visible = np.repeat(base[..., None], 3, axis=2).astype(np.uint8)
        semantic_mask = np.zeros((height, width), dtype=np.uint8)
        semantic_mask[12:20, 44:52] = 255
        semantic_mask[28:36, 10:18] = 255
        instance_mask = np.zeros((height, width), dtype=np.uint8)
        instance_mask[12:20, 44:52] = 255
        Image.fromarray(visible).save(root / "visible.png")
        Image.fromarray(base.astype(np.uint8)).save(root / "infrared.png")
        Image.fromarray(semantic_mask).save(root / "semantic_mask.png")
        Image.fromarray(instance_mask).save(root / "instance_mask.png")

        rows = [
            {
                "name": "global",
                "visible": "visible.png",
                "infrared": "infrared.png",
                "instruction": "Enhance the scene.",
                "granularity": "global",
            },
            {
                "name": "semantic",
                "visible": "visible.png",
                "infrared": "infrared.png",
                "mask": "semantic_mask.png",
                "instruction": "Highlight the car.",
                "granularity": "semantic",
                "class_name": "car",
            },
            {
                "name": "instance",
                "visible": "visible.png",
                "infrared": "infrared.png",
                "mask": "instance_mask.png",
                "instruction": "Highlight the right car.",
                "granularity": "instance",
                "class_name": "car",
            },
        ]
        manifest = root / "samples.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        dataset = FusionManifestDataset(
            str(manifest),
            image_size=32,
            augment=True,
            positive_crop_probability=1.0,
            local_crop_sizes=(16, 24, 32),
            local_crop_probabilities=(1.0, 0.0, 0.0),
        )
        local = dataset[1]
        assert torch.equal(local["visible"][0], local["infrared"][0])
        assert local["mask"].sum() > 0
        instance = FusionManifestDataset(str(manifest), image_size=32, augment=False)[2]
        assert instance["distractor_mask"].sum() > 0
        assert not torch.logical_and(
            instance["mask"].bool(), instance["distractor_mask"].bool()
        ).any()
        full_scene_instance = FusionManifestDataset(
            str(manifest),
            image_size=32,
            augment=True,
            positive_crop_probability=1.0,
            instance_full_scene_probability=1.0,
        )[2]
        assert full_scene_instance["mask"].sum() > 0
        assert full_scene_instance["distractor_mask"].sum() > 0
        weights = dataset.sample_weights(
            {"global": 0.5, "semantic": 1.0, "instance": 1.0},
            class_balance_power=0.5,
            size_balance_power=0.25,
        )
        assert torch.allclose(weights, torch.tensor((0.2, 0.4, 0.4), dtype=torch.double))
    print("v5 paired crop, balanced sampling, and distractor-mask test passed")


if __name__ == "__main__":
    main()
