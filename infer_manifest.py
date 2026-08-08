import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from controlfuse.checkpoint import load_model_weights
from controlfuse.config import load_config
from controlfuse.data import FusionManifestDataset
from controlfuse.model import build_model


def save_image(tensor: torch.Tensor, path: Path, size, resample):
    tensor = tensor.clamp(0, 1).mul(255).byte().cpu()
    if tensor.shape[0] == 1:
        array = tensor[0].numpy()
    else:
        array = tensor.permute(1, 2, 0).numpy()
    image = Image.fromarray(array)
    if image.size != size:
        image = image.resize(size, resample)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Run ControlFuse on a JSONL manifest.")
    parser.add_argument("--config", default="configs/controlfuse.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = FusionManifestDataset(args.manifest, config["data"]["image_size"], augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=config["data"].get("workers", 4))
    model = build_model(config).to(device)
    load_model_weights(model, args.checkpoint, device)
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (output_dir / "masks").mkdir(exist_ok=True)
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(batch["visible"].to(device), batch["infrared"].to(device), batch["instruction"])
            for index, name in enumerate(batch["name"]):
                original_size = (
                    int(batch["original_width"][index]),
                    int(batch["original_height"][index]),
                )
                save_image(
                    output["fused"][index],
                    output_dir / f"{name}.png",
                    original_size,
                    Image.Resampling.BICUBIC,
                )
                if args.save_masks:
                    mask = output["location_logits"][index].sigmoid()
                    save_image(
                        mask,
                        output_dir / "masks" / f"{name}.png",
                        original_size,
                        Image.Resampling.BILINEAR,
                    )
                processed += 1
            if processed % 50 < len(batch["name"]):
                print(f"processed {processed}/{len(dataset)}")
    print(f"saved {len(dataset)} fused images to {output_dir}")


if __name__ == "__main__":
    main()
