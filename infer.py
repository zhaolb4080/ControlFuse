import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from controlfuse.checkpoint import load_model_weights
from controlfuse.config import load_config
from controlfuse.model import build_model


def load_image(path: str, mode: str, size: int):
    image = Image.open(path).convert(mode)
    original_size = image.size
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0), original_size


def save_tensor(tensor: torch.Tensor, path: str, size):
    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    image = Image.fromarray((array * 255).round().astype(np.uint8))
    image.resize(size, Image.Resampling.BICUBIC).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msrs.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--visible", required=True)
    parser.add_argument("--infrared", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    load_model_weights(model, args.checkpoint, device)
    model.eval()
    size = config["data"]["image_size"]
    visible, original_size = load_image(args.visible, "RGB", size)
    infrared, _ = load_image(args.infrared, "L", size)
    with torch.no_grad():
        output = model(visible.to(device), infrared.to(device), [args.instruction])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_tensor(output["fused"], args.output, original_size)
    if args.mask_output:
        mask = output["location_logits"].sigmoid().repeat(1, 3, 1, 1)
        save_tensor(mask, args.mask_output, original_size)


if __name__ == "__main__":
    main()
