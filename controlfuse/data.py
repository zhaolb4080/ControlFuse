import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _resize_if_needed(images: Sequence[Image.Image], crop_size: int) -> List[Image.Image]:
    width, height = images[0].size
    scale = max(crop_size / width, crop_size / height, 1.0)
    if scale == 1.0:
        return list(images)
    new_size = (max(crop_size, round(width * scale)), max(crop_size, round(height * scale)))
    resized = []
    for index, image in enumerate(images):
        interpolation = Image.Resampling.NEAREST if index >= 2 else Image.Resampling.BICUBIC
        resized.append(image.resize(new_size, interpolation))
    return resized


def _random_crop_box(
    mask: Image.Image,
    crop_size: int,
    prefer_positive: bool,
) -> Tuple[int, int, int, int]:
    width, height = mask.size
    max_left = width - crop_size
    max_top = height - crop_size
    if prefer_positive:
        positive = np.argwhere(np.asarray(mask, dtype=np.uint8) > 127)
        if positive.size:
            y, x = positive[random.randrange(len(positive))]
            anchor_x = random.randint(crop_size // 4, max(crop_size // 4, 3 * crop_size // 4))
            anchor_y = random.randint(crop_size // 4, max(crop_size // 4, 3 * crop_size // 4))
            left = min(max(int(x) - anchor_x, 0), max_left)
            top = min(max(int(y) - anchor_y, 0), max_top)
            return left, top, left + crop_size, top + crop_size
    left = random.randint(0, max_left) if max_left > 0 else 0
    top = random.randint(0, max_top) if max_top > 0 else 0
    return left, top, left + crop_size, top + crop_size


class FusionManifestDataset(Dataset):
    """JSONL dataset with paired VIS/IR transforms and instruction masks."""

    def __init__(
        self,
        manifest: str,
        image_size: int = 256,
        augment: bool = False,
        positive_crop_probability: float = 0.7,
        local_crop_sizes: Optional[Sequence[int]] = None,
        local_crop_probabilities: Optional[Sequence[float]] = None,
        semantic_full_scene_probability: float = 0.0,
        instance_full_scene_probability: float = 0.0,
    ):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.image_size = int(image_size)
        self.augment = augment
        self.positive_crop_probability = float(positive_crop_probability)
        self.local_crop_sizes = tuple(int(value) for value in (local_crop_sizes or (image_size,)))
        probabilities = local_crop_probabilities or (1.0,) * len(self.local_crop_sizes)
        self.local_crop_probabilities = tuple(float(value) for value in probabilities)
        self.full_scene_probabilities = {
            "semantic": float(semantic_full_scene_probability),
            "instance": float(instance_full_scene_probability),
        }
        if self.image_size < 8:
            raise ValueError("image_size must be at least 8")
        if not 0 <= self.positive_crop_probability <= 1:
            raise ValueError("positive_crop_probability must be in [0, 1]")
        if not self.local_crop_sizes or any(value < 8 for value in self.local_crop_sizes):
            raise ValueError("local_crop_sizes must contain values of at least 8")
        if len(self.local_crop_sizes) != len(self.local_crop_probabilities):
            raise ValueError("local_crop_sizes and local_crop_probabilities must have equal lengths")
        if any(value < 0 for value in self.local_crop_probabilities) or sum(self.local_crop_probabilities) <= 0:
            raise ValueError("local_crop_probabilities must be non-negative with a positive sum")
        if any(not 0 <= value <= 1 for value in self.full_scene_probabilities.values()):
            raise ValueError("full-scene probabilities must be in [0, 1]")
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.items: List[Dict] = [json.loads(line) for line in handle if line.strip()]
        if not self.items:
            raise ValueError(f"No samples found in {self.manifest}")
        self.semantic_masks: Dict[Tuple[str, str], str] = {}
        for item in self.items:
            if item.get("granularity") == "semantic" and item.get("mask"):
                self.semantic_masks[self._semantic_key(item)] = item["mask"]

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _semantic_key(item: Dict) -> Tuple[str, str]:
        source = str(item.get("source_name", item.get("visible", "")))
        return source, str(item.get("class_name", "unknown"))

    def __len__(self) -> int:
        return len(self.items)

    def sample_weights(
        self,
        granularity_weights: Optional[Dict[str, float]] = None,
        class_balance_power: float = 0.5,
        size_balance_power: float = 0.0,
        maximum_size_weight: float = 4.0,
    ) -> torch.Tensor:
        granularities = [item.get("granularity", "global") for item in self.items]
        available = sorted(set(granularities))
        requested = granularity_weights or {name: 1.0 for name in available}
        desired = {name: float(requested.get(name, 0.0)) for name in available}
        if any(value < 0 for value in desired.values()) or sum(desired.values()) <= 0:
            raise ValueError("granularity weights must be non-negative with a positive sum")
        if class_balance_power < 0:
            raise ValueError("class_balance_power must be non-negative")
        if size_balance_power < 0:
            raise ValueError("size_balance_power must be non-negative")
        if maximum_size_weight < 1:
            raise ValueError("maximum_size_weight must be at least 1")

        raw = np.zeros(len(self.items), dtype=np.float64)
        for granularity in available:
            indices = [index for index, value in enumerate(granularities) if value == granularity]
            if granularity == "global" or class_balance_power == 0:
                correction = np.ones(len(indices), dtype=np.float64)
            else:
                labels = [str(self.items[index].get("class_name", "unknown")) for index in indices]
                counts = Counter(labels)
                correction = np.asarray(
                    [counts[label] ** (-class_balance_power) for label in labels], dtype=np.float64
                )
                if size_balance_power > 0:
                    areas = np.asarray(
                        [self._mask_fraction(self.items[index]) for index in indices], dtype=np.float64
                    )
                    positive_areas = areas[areas > 0]
                    reference = float(np.median(positive_areas)) if positive_areas.size else 1.0
                    size_correction = (np.maximum(areas, 1e-8) / max(reference, 1e-8)) ** (
                        -size_balance_power
                    )
                    size_correction = np.clip(
                        size_correction, 1.0 / maximum_size_weight, maximum_size_weight
                    )
                    correction *= size_correction
            correction /= correction.sum()
            raw[indices] = desired[granularity] * correction
        raw /= raw.sum()
        return torch.as_tensor(raw, dtype=torch.double)

    def _mask_fraction(self, item: Dict) -> float:
        stored = item.get("mask_fraction")
        if stored is not None:
            return float(stored)
        mask_path = item.get("mask")
        if not mask_path:
            return 1.0
        with Image.open(self._resolve(mask_path)) as source:
            array = np.asarray(source.convert("L"), dtype=np.uint8)
        return float((array > 127).mean())

    def _load_images(self, item: Dict, index: int):
        visible_path = self._resolve(item["visible"])
        infrared_path = self._resolve(item["infrared"])
        with Image.open(visible_path) as source:
            visible = source.convert("RGB")
        with Image.open(infrared_path) as source:
            infrared = source.convert("L")
        original_size = visible.size
        if infrared.size != original_size:
            raise ValueError(
                f"IR/VIS size mismatch for {item.get('name', index)}: "
                f"visible={original_size}, infrared={infrared.size}"
            )
        if item.get("mask"):
            with Image.open(self._resolve(item["mask"])) as source:
                mask = source.convert("L")
            if mask.size != original_size:
                raise ValueError(
                    f"Image/mask size mismatch for {item.get('name', index)}: "
                    f"image={original_size}, mask={mask.size}"
                )
        else:
            mask = Image.new("L", original_size, color=255)

        distractor = Image.new("L", original_size, color=0)
        if item.get("granularity") == "instance":
            semantic_path = item.get("semantic_mask") or self.semantic_masks.get(
                self._semantic_key(item)
            )
            if semantic_path:
                with Image.open(self._resolve(semantic_path)) as source:
                    semantic_mask = source.convert("L")
                if semantic_mask.size != original_size:
                    raise ValueError(
                        f"Image/semantic-mask size mismatch for {item.get('name', index)}: "
                        f"image={original_size}, semantic_mask={semantic_mask.size}"
                    )
                semantic_array = np.asarray(semantic_mask, dtype=np.uint8) > 127
                instance_array = np.asarray(mask, dtype=np.uint8) > 127
                distractor = Image.fromarray(
                    (semantic_array & ~instance_array).astype(np.uint8) * 255,
                    mode="L",
                )
        return visible, infrared, mask, distractor, original_size

    def __getitem__(self, index: int) -> Dict:
        item = self.items[index]
        visible, infrared, mask, distractor, original_size = self._load_images(item, index)
        granularity = item.get("granularity", "global")

        if self.augment:
            full_scene_probability = self.full_scene_probabilities.get(granularity, 0.0)
            use_full_scene = granularity != "global" and random.random() < full_scene_probability
            if use_full_scene:
                target_size = (self.image_size, self.image_size)
                visible = visible.resize(target_size, Image.Resampling.BICUBIC)
                infrared = infrared.resize(target_size, Image.Resampling.BICUBIC)
                mask = mask.resize(target_size, Image.Resampling.NEAREST)
                distractor = distractor.resize(target_size, Image.Resampling.NEAREST)
            else:
                prefer_positive = (
                    granularity != "global"
                    and random.random() < self.positive_crop_probability
                )
                crop_size = self.image_size
                if granularity != "global" and prefer_positive:
                    crop_size = random.choices(
                        self.local_crop_sizes,
                        weights=self.local_crop_probabilities,
                        k=1,
                    )[0]
                visible, infrared, mask, distractor = _resize_if_needed(
                    (visible, infrared, mask, distractor), crop_size
                )
                crop_box = _random_crop_box(mask, crop_size, prefer_positive)
                visible, infrared, mask, distractor = [
                    image.crop(crop_box) for image in (visible, infrared, mask, distractor)
                ]
                if crop_size != self.image_size:
                    target_size = (self.image_size, self.image_size)
                    visible = visible.resize(target_size, Image.Resampling.BICUBIC)
                    infrared = infrared.resize(target_size, Image.Resampling.BICUBIC)
                    mask = mask.resize(target_size, Image.Resampling.NEAREST)
                    distractor = distractor.resize(target_size, Image.Resampling.NEAREST)
        else:
            target_size = (self.image_size, self.image_size)
            visible = visible.resize(target_size, Image.Resampling.BICUBIC)
            infrared = infrared.resize(target_size, Image.Resampling.BICUBIC)
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
            distractor = distractor.resize(target_size, Image.Resampling.NEAREST)

        visible_tensor = _to_tensor(visible)
        infrared_tensor = _to_tensor(infrared)
        mask_tensor = (_to_tensor(mask) > 0.5).float()
        distractor_tensor = (_to_tensor(distractor) > 0.5).float()
        if self.augment:
            if random.random() < 0.5:
                visible_tensor, infrared_tensor, mask_tensor, distractor_tensor = [
                    value.flip(-1)
                    for value in (visible_tensor, infrared_tensor, mask_tensor, distractor_tensor)
                ]
            if random.random() < 0.5:
                visible_tensor, infrared_tensor, mask_tensor, distractor_tensor = [
                    value.flip(-2)
                    for value in (visible_tensor, infrared_tensor, mask_tensor, distractor_tensor)
                ]

        instruction = item.get("instruction", "Enhance the entire scene.")
        negative = item.get("negative_instruction", "Suppress every relevant object in the scene.")
        return {
            "visible": visible_tensor,
            "infrared": infrared_tensor,
            "mask": mask_tensor,
            "distractor_mask": distractor_tensor,
            "has_target": bool(mask_tensor.any()),
            "mask_fraction": float(mask_tensor.mean()),
            "instruction": instruction,
            "negative_instruction": negative,
            "granularity": granularity,
            "class_name": item.get("class_name", "global"),
            "name": item.get("name", self._resolve(item["visible"]).stem),
            "original_width": original_size[0],
            "original_height": original_size[1],
        }
