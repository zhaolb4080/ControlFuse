import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controlfuse.config import load_config
from controlfuse.losses import ControlFuseCriterion
from controlfuse.model import build_model


def main():
    config = load_config(str(ROOT / "configs" / "smoke.yaml"))
    model = build_model(config)
    criterion = ControlFuseCriterion(config)
    visible = torch.rand(2, 3, 32, 32)
    infrared = torch.rand(2, 1, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    mask[0] = 1
    mask[1, :, 8:24, 10:22] = 1
    distractor = torch.zeros_like(mask)
    distractor[1, :, 8:24, 24:28] = 1
    output = model(
        visible,
        infrared,
        ["Enhance the entire scene.", "Highlight the left pedestrian."],
        ["Suppress the scene.", "Highlight an absent vehicle."],
    )
    assert output["fused"].shape == (2, 3, 32, 32)
    assert output["location_logits"].shape == (2, 1, 32, 32)
    assert output["visual_feature_map"].shape == (2, 8, 32, 32)
    losses = criterion(
        output, visible, infrared, mask, ["global", "instance"], distractor
    )
    losses["total"].backward()
    for name, value in losses.items():
        assert torch.isfinite(value), f"non-finite {name}: {value}"
    empty_mask = mask.clone()
    empty_mask[1] = 0
    empty_losses = criterion(
        output, visible, infrared, empty_mask, ["global", "instance"], distractor
    )
    for name, value in empty_losses.items():
        assert torch.isfinite(value), f"non-finite empty-mask {name}: {value}"
    print({key: float(value.detach()) for key, value in losses.items()})


if __name__ == "__main__":
    main()
