from pathlib import Path

import torch


def save_checkpoint(path, model, criterion, optimizer, scheduler, epoch, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def load_model_weights(model, path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Checkpoint notice: missing={len(missing)}, unexpected={len(unexpected)}")
    return checkpoint
