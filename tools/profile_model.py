import argparse

import torch
from torch import nn

from controlfuse.config import load_config
from controlfuse.model import build_model


class ProfileWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, visible, infrared):
        return self.model(visible, infrared, ["Enhance the entire scene."])["fused"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlfuse.yaml")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("text", {}).get("backend") != "hash":
        print("Tip: use configs/smoke.yaml for offline profiling without downloading BLIP.")
    model = build_model(config).eval()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters: trainable={trainable / 1e6:.3f}M total={total / 1e6:.3f}M")
    try:
        from fvcore.nn import FlopCountAnalysis
        visible = torch.rand(1, 3, args.size, args.size)
        infrared = torch.rand(1, 1, args.size, args.size)
        flops = FlopCountAnalysis(ProfileWrapper(model), (visible, infrared))
        print(f"FLOPs: {flops.total() / 1e9:.3f}G")
    except ImportError:
        print("Install fvcore to report FLOPs.")


if __name__ == "__main__":
    main()
