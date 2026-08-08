import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controlfuse.config import load_config
from controlfuse.losses import ControlFuseCriterion


def main():
    config = load_config(str(ROOT / "configs" / "smoke.yaml"))
    criterion = ControlFuseCriterion(config)
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 5:11, 6:10] = 1
    good_logits = torch.full_like(mask, -5.0)
    good_logits[mask.bool()] = 5.0
    broad_logits = torch.full_like(mask, 5.0)
    good, good_parts = criterion.localization_loss(good_logits, mask, ["instance"])
    broad, broad_parts = criterion.localization_loss(broad_logits, mask, ["instance"])
    assert good < broad
    assert good_parts["tversky"] < broad_parts["tversky"]
    assert good_parts["boundary"] < broad_parts["boundary"]
    assert good_parts["false_positive"] < broad_parts["false_positive"]
    assert good_parts["area_error"] < broad_parts["area_error"]
    assert good_parts["positive_weight"] > 1

    empty = torch.zeros_like(mask)
    suppressed, _ = criterion.localization_loss(-broad_logits, empty, ["instance"])
    activated, _ = criterion.localization_loss(broad_logits, empty, ["instance"])
    assert suppressed < activated

    distractor = torch.zeros_like(mask)
    distractor[:, :, 5:11, 12:15] = 1
    clean_logits = good_logits.clone()
    confused_logits = good_logits.clone()
    confused_logits[distractor.bool()] = 5.0
    clean, clean_parts = criterion.localization_loss(
        clean_logits, mask, ["instance"], distractor
    )
    confused, confused_parts = criterion.localization_loss(
        confused_logits, mask, ["instance"], distractor
    )
    assert clean < confused
    assert clean_parts["distractor"] < confused_parts["distractor"]

    visible = torch.zeros(1, 3, 16, 16)
    infrared = torch.ones(1, 1, 16, 16)
    local_target = mask.repeat(1, 3, 1, 1)
    _, local_parts = criterion.fidelity_loss(local_target, visible, infrared, mask)
    assert local_parts["intensity"] < 1e-6
    global_mask = torch.ones_like(mask)
    _, global_parts = criterion.fidelity_loss(torch.ones_like(visible), visible, infrared, global_mask)
    assert global_parts["intensity"] < 1e-6

    positive = torch.zeros(1, 8)
    negative = torch.zeros(1, 8)
    positive[:, 0] = 1
    negative[:, 1] = 1
    background_good = torch.zeros(1, 8, 16, 16)
    background_bad = torch.zeros(1, 8, 16, 16)
    background_good[:, 1] = 1
    background_bad[:, 0] = 1
    common = {
        "text_embedding": positive,
        "negative_text_embedding": negative,
        "visual_curvature": torch.zeros(1),
        "text_curvature": torch.zeros(1),
    }
    good_alignment = criterion.alignment_loss(
        {**common, "visual_feature_map": background_good}, empty, ["instance"]
    )
    bad_alignment = criterion.alignment_loss(
        {**common, "visual_feature_map": background_bad}, empty, ["instance"]
    )
    assert good_alignment < bad_alignment
    print("v5 boundary, distractor, balanced localization, and fidelity test passed")


if __name__ == "__main__":
    main()
