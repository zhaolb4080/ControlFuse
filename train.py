import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from controlfuse.checkpoint import save_checkpoint
from controlfuse.config import load_config
from controlfuse.data import FusionManifestDataset
from controlfuse.losses import ControlFuseCriterion
from controlfuse.model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, criterion, loader, device, thresholds=None):
    model.eval()
    totals = {key: 0.0 for key in ("total", "localization", "fidelity", "alignment")}
    thresholds = thresholds or {"semantic": 0.5, "instance": 0.5}
    local_iou = {"semantic": 0.0, "instance": 0.0}
    local_soft_iou = {"semantic": 0.0, "instance": 0.0}
    local_zero = {"semantic": 0, "instance": 0}
    local_count = {"semantic": 0, "instance": 0}
    count = 0
    for batch in loader:
        visible = batch["visible"].to(device)
        infrared = batch["infrared"].to(device)
        mask = batch["mask"].to(device)
        distractor_mask = batch["distractor_mask"].to(device)
        output = model(visible, infrared, batch["instruction"], batch["negative_instruction"])
        losses = criterion(
            output, visible, infrared, mask, batch["granularity"], distractor_mask
        )
        for key in totals:
            totals[key] += float(losses[key]) * visible.shape[0]
        probability = output["location_logits"].sigmoid()
        for index, granularity in enumerate(batch["granularity"]):
            if granularity not in local_iou:
                continue
            threshold = thresholds.get(
                f"{granularity}_threshold", thresholds.get(granularity, 0.5)
            )
            prediction = probability[index] >= float(threshold)
            target = mask[index] >= 0.5
            intersection = (prediction & target).sum().item()
            union = (prediction | target).sum().item()
            iou = intersection / max(union, 1)
            soft_target = target.float()
            soft_prediction = probability[index].float()
            soft_intersection = (soft_prediction * soft_target).sum().item()
            soft_union = (
                soft_prediction + soft_target - soft_prediction * soft_target
            ).sum().item()
            local_iou[granularity] += iou
            local_soft_iou[granularity] += soft_intersection / max(soft_union, 1e-6)
            local_zero[granularity] += int(intersection == 0)
            local_count[granularity] += 1
        count += visible.shape[0]
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    for granularity in local_iou:
        denominator = max(local_count[granularity], 1)
        metrics[f"{granularity}_iou"] = local_iou[granularity] / denominator
        metrics[f"{granularity}_soft_iou"] = local_soft_iou[granularity] / denominator
        metrics[f"{granularity}_zero_iou_rate"] = local_zero[granularity] / denominator
        metrics[f"{granularity}_count"] = float(local_count[granularity])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msrs.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    requested = config.get("device", "cuda")
    device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")

    data_cfg, train_cfg = config["data"], config["train"]
    train_set = FusionManifestDataset(
        data_cfg["train_manifest"],
        data_cfg["image_size"],
        augment=True,
        positive_crop_probability=data_cfg.get("positive_crop_probability", 0.5),
        local_crop_sizes=data_cfg.get("local_crop_sizes"),
        local_crop_probabilities=data_cfg.get("local_crop_probabilities"),
        semantic_full_scene_probability=data_cfg.get(
            "semantic_full_scene_probability", 0.0
        ),
        instance_full_scene_probability=data_cfg.get(
            "instance_full_scene_probability", 0.0
        ),
    )
    val_manifest = data_cfg.get("val_manifest")
    val_set = (
        FusionManifestDataset(val_manifest, data_cfg["image_size"], augment=False)
        if val_manifest
        else None
    )
    sampling_cfg = train_cfg.get("sampling", {})
    sampler = None
    if sampling_cfg.get("balanced", True):
        sample_weights = train_set.sample_weights(
            sampling_cfg.get("granularity_weights"),
            sampling_cfg.get("class_balance_power", 0.5),
            sampling_cfg.get("size_balance_power", 0.0),
            sampling_cfg.get("maximum_size_weight", 4.0),
        )
        sampler_generator = torch.Generator().manual_seed(config.get("seed", 42))
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_set),
            replacement=True,
            generator=sampler_generator,
        )
        print(
            "balanced sampling enabled: "
            f"granularity_weights={sampling_cfg.get('granularity_weights', 'equal')} "
            f"class_balance_power={sampling_cfg.get('class_balance_power', 0.5)} "
            f"size_balance_power={sampling_cfg.get('size_balance_power', 0.0)}"
        )
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=data_cfg["workers"],
        pin_memory=device.type == "cuda",
        drop_last=len(train_set) >= train_cfg["batch_size"],
        persistent_workers=data_cfg["workers"] > 0,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["workers"],
            pin_memory=device.type == "cuda",
            persistent_workers=data_cfg["workers"] > 0,
        )
        if val_set is not None
        else None
    )
    model = build_model(config).to(device)
    criterion = ControlFuseCriterion(config).to(device)
    parameters = [p for p in list(model.parameters()) + list(criterion.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train_cfg["lr_step"], train_cfg["lr_gamma"])
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_schema = checkpoint.get("config", {}).get("training_schema")
        current_schema = config.get("training_schema")
        if current_schema and checkpoint_schema != current_schema:
            raise ValueError(
                f"Checkpoint training_schema={checkpoint_schema!r} is incompatible with "
                f"the current training_schema={current_schema!r}. Start this version without --resume."
            )
        model.load_state_dict(checkpoint["model"])
        criterion.load_state_dict(checkpoint["criterion"])
        criterion.clamp_parameters_()
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")
    best_score = float("-inf")
    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        for step, batch in enumerate(train_loader, 1):
            visible = batch["visible"].to(device, non_blocking=True)
            infrared = batch["infrared"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            distractor_mask = batch["distractor_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(visible, infrared, batch["instruction"], batch["negative_instruction"])
                losses = criterion(
                    output, visible, infrared, mask, batch["granularity"], distractor_mask
                )
            nonfinite = [key for key, value in losses.items() if not torch.isfinite(value).all()]
            if nonfinite:
                values = " ".join(f"{key}={losses[key].detach().float().item()}" for key in losses)
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, step={step}: {nonfinite}; {values}. "
                    "The optimizer step was not executed. Resume from the last finite periodic checkpoint."
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm):
                bad_gradients = []
                for prefix, module in (("model", model), ("criterion", criterion)):
                    for name, parameter in module.named_parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            bad_gradients.append(f"{prefix}.{name}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                first_bad = ", ".join(bad_gradients[:5]) if bad_gradients else "unknown"
                print(
                    f"warning: skipped non-finite gradient at epoch={epoch + 1}, step={step}; "
                    f"parameters={first_bad}"
                )
                continue
            scaler.step(optimizer)
            scaler.update()
            criterion.clamp_parameters_()
            if step % 20 == 0 or step == len(train_loader):
                log_vars = ",".join(f"{value:.3f}" for value in criterion.log_vars.detach().cpu().tolist())
                print(
                    f"epoch={epoch + 1:03d} step={step:04d}/{len(train_loader):04d} "
                    f"loss={losses['total'].item():.4f} "
                    f"loc={losses['localization'].item():.4f} "
                    f"fid={losses['fidelity'].item():.4f} "
                    f"align={losses['alignment'].item():.4f} "
                    f"tv={losses['tversky'].item():.4f} "
                    f"bnd={losses['boundary'].item():.4f} "
                    f"dst={losses['distractor'].item():.4f} "
                    f"fgp={losses['foreground_probability'].item():.4f} "
                    f"posw={losses['positive_weight'].item():.2f} "
                    f"fp={losses['false_positive'].item():.4f} "
                    f"area={losses['area_error'].item():.4f} "
                    f"int={losses['intensity'].item():.4f} "
                    f"grad={losses['gradient'].item():.4f} "
                    f"ssim={losses['structure'].item():.4f} "
                    f"color={losses['color'].item():.4f} log_vars=[{log_vars}]"
                )
        scheduler.step()
        save_checkpoint(output_dir / "last.pt", model, criterion, optimizer, scheduler, epoch, config)
        if val_loader is not None:
            metrics = validate(model, criterion, val_loader, device, config.get("validation", {}))
            print("validation", " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
            available = [
                metrics[f"{name}_soft_iou"]
                for name in ("semantic", "instance")
                if metrics[f"{name}_count"] > 0
            ]
            score = sum(available) / len(available) if available else -metrics["total"]
            if score > best_score:
                best_score = score
                save_checkpoint(output_dir / "best.pt", model, criterion, optimizer, scheduler, epoch, config)
        else:
            print(f"epoch={epoch + 1:03d} complete; fixed-epoch checkpoint saved to {output_dir / 'last.pt'}")
        if (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pt", model, criterion, optimizer, scheduler, epoch, config)


if __name__ == "__main__":
    main()
