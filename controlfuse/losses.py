from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    padding = window // 2
    mu_x = F.avg_pool2d(x, window, 1, padding)
    mu_y = F.avg_pool2d(y, window, 1, padding)
    sigma_x = F.avg_pool2d(x * x, window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, window, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return score.clamp(-1.0, 1.0).mean()


def _luminance(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image * weights).sum(1, keepdim=True)


def _chroma(image: torch.Tensor) -> torch.Tensor:
    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue
    return torch.cat((cb, cr), dim=1)


def _sobel(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_x = image.new_tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)))
    kernel_x = kernel_x.view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel_x), F.conv2d(padded, kernel_y)


def _morphological_boundary(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


class ControlFuseCriterion(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        loss_cfg = config.get("loss", {})
        self.curvature_weight = float(loss_cfg.get("curvature_weight", 0.2))
        self.temperature = float(loss_cfg.get("contrastive_temperature", 0.07))
        self.alignment_margin = float(loss_cfg.get("alignment_margin", 0.1))
        self.focal_weight = float(loss_cfg.get("focal_weight", 1.0))
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        self.maximum_positive_weight = float(loss_cfg.get("maximum_positive_weight", 10.0))
        self.tversky_weight = float(loss_cfg.get("tversky_weight", 1.0))
        self.tversky_alpha = float(loss_cfg.get("tversky_alpha", 0.55))
        self.tversky_beta = float(loss_cfg.get("tversky_beta", 0.45))
        self.boundary_weight = float(loss_cfg.get("boundary_weight", 0.2))
        self.distractor_weight = float(loss_cfg.get("distractor_weight", 0.1))
        self.false_positive_weight = float(loss_cfg.get("false_positive_weight", 0.05))
        self.distractor_alignment_weight = float(
            loss_cfg.get("distractor_alignment_weight", 0.5)
        )
        self.empty_negative_weight = float(loss_cfg.get("empty_negative_weight", 0.15))
        self.global_localization_weight = float(loss_cfg.get("global_localization_weight", 0.25))
        self.localization_floor_weight = float(loss_cfg.get("localization_floor_weight", 0.5))
        self.intensity_weight = float(loss_cfg.get("intensity_weight", 10.0))
        self.gradient_weight = float(loss_cfg.get("gradient_weight", 10.0))
        self.ssim_weight = float(loss_cfg.get("ssim_weight", 1.0))
        self.color_weight = float(loss_cfg.get("color_weight", 1.0))
        self.log_var_min = float(loss_cfg.get("uncertainty_log_var_min", -3.0))
        self.log_var_max = float(loss_cfg.get("uncertainty_log_var_max", 3.0))
        if self.temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")
        if self.maximum_positive_weight < 1:
            raise ValueError("maximum_positive_weight must be at least 1")
        if self.tversky_alpha < 0 or self.tversky_beta < 0:
            raise ValueError("Tversky alpha and beta must be non-negative")
        if self.tversky_alpha + self.tversky_beta <= 0:
            raise ValueError("At least one Tversky coefficient must be positive")
        auxiliary_weights = (
            self.boundary_weight,
            self.distractor_weight,
            self.false_positive_weight,
            self.distractor_alignment_weight,
            self.empty_negative_weight,
            self.localization_floor_weight,
        )
        if any(value < 0 for value in auxiliary_weights):
            raise ValueError("Loss weights must be non-negative")
        if self.global_localization_weight <= 0:
            raise ValueError("global_localization_weight must be positive")
        if self.log_var_min >= self.log_var_max:
            raise ValueError("uncertainty_log_var_min must be smaller than uncertainty_log_var_max")
        self.log_vars = nn.Parameter(torch.zeros(3))

    @torch.no_grad()
    def clamp_parameters_(self):
        if not torch.isfinite(self.log_vars).all():
            raise FloatingPointError(
                "Non-finite uncertainty log_vars found in the checkpoint. Resume from an earlier checkpoint."
            )
        self.log_vars.clamp_(self.log_var_min, self.log_var_max)

    @staticmethod
    def _local_flags(
        granularities: Optional[Sequence[str]], mask: torch.Tensor
    ) -> torch.Tensor:
        if granularities is None:
            return mask.flatten(1).amin(1) < 0.5
        if len(granularities) != mask.shape[0]:
            raise ValueError("granularities length must match the batch size")
        return torch.tensor(
            [str(value) != "global" for value in granularities],
            dtype=torch.bool,
            device=mask.device,
        )

    def localization_loss(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        granularities: Optional[Sequence[str]] = None,
        distractor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.shape[-2:] != mask.shape[-2:]:
            mask = F.interpolate(mask, logits.shape[-2:], mode="nearest")
        mask = mask.float().clamp(0.0, 1.0)
        if distractor_mask is None:
            distractor_mask = torch.zeros_like(mask)
        elif distractor_mask.shape[-2:] != logits.shape[-2:]:
            distractor_mask = F.interpolate(
                distractor_mask.float(), logits.shape[-2:], mode="nearest"
            )
        distractor_mask = distractor_mask.float().clamp(0.0, 1.0) * (1.0 - mask)
        probability = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
        p_t = probability * mask + (1.0 - probability) * (1.0 - mask)
        foreground_count = mask.sum((1, 2, 3))
        background_count = (1.0 - mask).sum((1, 2, 3))
        positive_weight = torch.sqrt(
            background_count / foreground_count.clamp_min(1.0)
        ).clamp(1.0, self.maximum_positive_weight)
        pixel_weight = 1.0 + mask * (positive_weight.view(-1, 1, 1, 1) - 1.0)
        focal_map = (1.0 - p_t).pow(self.focal_gamma) * bce * pixel_weight
        focal = focal_map.sum((1, 2, 3)) / pixel_weight.sum((1, 2, 3)).clamp_min(1.0)

        true_positive = (probability * mask).sum((1, 2, 3))
        foreground_probability = true_positive / foreground_count.clamp_min(1.0)
        background = 1.0 - mask
        false_negative = ((1.0 - probability) * mask).sum((1, 2, 3))
        false_positive_sum = (probability * background).sum((1, 2, 3))
        tversky = 1.0 - (true_positive + 1.0) / (
            true_positive
            + self.tversky_alpha * false_positive_sum
            + self.tversky_beta * false_negative
            + 1.0
        )
        false_positive = (probability * background).sum((1, 2, 3)) / (
            background.sum((1, 2, 3)).clamp_min(1.0)
        )
        predicted_boundary = _morphological_boundary(probability)
        target_boundary = _morphological_boundary(mask)
        boundary_intersection = (predicted_boundary * target_boundary).sum((1, 2, 3))
        boundary = 1.0 - (2.0 * boundary_intersection + 1.0) / (
            predicted_boundary.sum((1, 2, 3)) + target_boundary.sum((1, 2, 3)) + 1.0
        )
        distractor_count = distractor_mask.sum((1, 2, 3))
        distractor = (probability * distractor_mask).sum((1, 2, 3)) / distractor_count.clamp_min(
            1.0
        )
        area_error = (probability.mean((1, 2, 3)) - mask.mean((1, 2, 3))).abs()
        has_foreground = foreground_count > 0
        local = self._local_flags(granularities, mask)
        local_positive = local & has_foreground
        positive_loss = self.focal_weight * focal + self.tversky_weight * tversky
        positive_loss = positive_loss + local_positive.float() * (
            self.boundary_weight * boundary
            + self.distractor_weight * distractor
            + self.false_positive_weight * false_positive
        )
        empty_loss = self.empty_negative_weight * bce.mean((1, 2, 3))
        sample_loss = torch.where(has_foreground, positive_loss, empty_loss)
        sample_weights = torch.where(
            local,
            torch.ones_like(sample_loss),
            torch.full_like(sample_loss, self.global_localization_weight),
        )
        total = (sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
        return total, {
            "false_positive": false_positive[local].mean() if local.any() else false_positive.mean(),
            "area_error": area_error[local].mean() if local.any() else area_error.mean(),
            "tversky": tversky[has_foreground].mean() if has_foreground.any() else tversky.mean(),
            "boundary": boundary[local_positive].mean() if local_positive.any() else boundary.mean(),
            "distractor": (
                distractor[distractor_count > 0].mean()
                if (distractor_count > 0).any()
                else distractor.mean()
            ),
            "foreground_probability": (
                foreground_probability[local_positive].mean()
                if local_positive.any()
                else foreground_probability.mean()
            ),
            "positive_weight": (
                positive_weight[has_foreground].mean() if has_foreground.any() else positive_weight.mean()
            ),
        }

    def fidelity_loss(
        self,
        fused: torch.Tensor,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fused_y = _luminance(fused.float())
        visible_y = _luminance(visible.float())
        infrared_y = infrared.float()
        if mask.shape[-2:] != fused_y.shape[-2:]:
            mask = F.interpolate(mask.float(), fused_y.shape[-2:], mode="nearest")
        mask = mask.float().clamp(0.0, 1.0)

        selected_intensity = torch.maximum(visible_y, infrared_y)
        intensity_target = mask * selected_intensity + (1.0 - mask) * visible_y
        intensity = F.l1_loss(fused_y, intensity_target)

        fused_gx, fused_gy = _sobel(fused_y)
        visible_gx, visible_gy = _sobel(visible_y)
        infrared_gx, infrared_gy = _sobel(infrared_y)
        selected_gx = torch.where(visible_gx.abs() >= infrared_gx.abs(), visible_gx, infrared_gx)
        selected_gy = torch.where(visible_gy.abs() >= infrared_gy.abs(), visible_gy, infrared_gy)
        target_gx = mask * selected_gx + (1.0 - mask) * visible_gx
        target_gy = mask * selected_gy + (1.0 - mask) * visible_gy
        gradient = F.l1_loss(fused_gx, target_gx) + F.l1_loss(fused_gy, target_gy)

        structure = 1.0 - _ssim(fused_y, intensity_target)
        color = F.l1_loss(_chroma(fused.float()), _chroma(visible.float()))
        total = (
            self.intensity_weight * intensity
            + self.gradient_weight * gradient
            + self.ssim_weight * structure
            + self.color_weight * color
        )
        return total, {
            "intensity": intensity,
            "gradient": gradient,
            "structure": structure,
            "color": color,
        }

    def alignment_loss(
        self,
        output: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        granularities: Optional[Sequence[str]] = None,
        distractor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature = output["visual_feature_map"].float()
        if mask.shape[-2:] != feature.shape[-2:]:
            mask = F.interpolate(mask.float(), feature.shape[-2:], mode="nearest")
        foreground = mask.float().clamp(0.0, 1.0)
        background = 1.0 - foreground
        if distractor_mask is None:
            distractor = torch.zeros_like(foreground)
        else:
            if distractor_mask.shape[-2:] != feature.shape[-2:]:
                distractor_mask = F.interpolate(
                    distractor_mask.float(), feature.shape[-2:], mode="nearest"
                )
            distractor = distractor_mask.float().clamp(0.0, 1.0) * background
        local = self._local_flags(granularities, foreground)
        foreground_valid = foreground.sum((1, 2, 3)) > 0
        background_valid = local & (background.sum((1, 2, 3)) > 0)
        distractor_valid = local & (distractor.sum((1, 2, 3)) > 0)

        normalized_feature = F.normalize(feature, dim=1, eps=1e-6)
        positive = F.normalize(output["text_embedding"].float(), dim=-1, eps=1e-6)
        positive_similarity = (normalized_feature * positive[:, :, None, None]).sum(1, keepdim=True)

        sample_loss = feature.new_zeros(feature.shape[0])
        term_count = feature.new_zeros(feature.shape[0])
        if "negative_text_embedding" in output:
            negative = F.normalize(output["negative_text_embedding"].float(), dim=-1, eps=1e-6)
            negative_similarity = (normalized_feature * negative[:, :, None, None]).sum(1, keepdim=True)
            foreground_map = F.softplus(
                (negative_similarity - positive_similarity + self.alignment_margin) / self.temperature
            )
            background_map = F.softplus(
                (positive_similarity - negative_similarity + self.alignment_margin) / self.temperature
            )
        else:
            foreground_map = 1.0 - positive_similarity
            background_map = F.relu(positive_similarity - self.alignment_margin)
        distractor_map = F.relu(positive_similarity - self.alignment_margin)

        foreground_rank = (foreground_map * foreground).sum((1, 2, 3)) / foreground.sum(
            (1, 2, 3)
        ).clamp_min(1.0)
        background_rank = (background_map * background).sum((1, 2, 3)) / background.sum(
            (1, 2, 3)
        ).clamp_min(1.0)
        distractor_rank = (distractor_map * distractor).sum((1, 2, 3)) / distractor.sum(
            (1, 2, 3)
        ).clamp_min(1.0)

        sample_loss = sample_loss + foreground_rank * foreground_valid.float()
        term_count = term_count + foreground_valid.float()
        sample_loss = sample_loss + background_rank * background_valid.float()
        term_count = term_count + background_valid.float()
        sample_loss = sample_loss + (
            self.distractor_alignment_weight * distractor_rank * distractor_valid.float()
        )
        term_count = term_count + self.distractor_alignment_weight * distractor_valid.float()
        valid = term_count > 0
        sample_loss = sample_loss / term_count.clamp_min(1.0)

        curvature = output["visual_curvature"].float() + output["text_curvature"].float()
        curvature = curvature / curvature.detach().mean().clamp_min(1e-6)
        weighted = sample_loss * (1.0 + self.curvature_weight * curvature)
        if valid.any():
            return weighted[valid].mean()
        return feature.sum() * 0.0

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        visible: torch.Tensor,
        infrared: torch.Tensor,
        mask: torch.Tensor,
        granularities: Optional[Sequence[str]] = None,
        distractor_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        localization, localization_parts = self.localization_loss(
            output["location_logits"], mask, granularities, distractor_mask
        )
        fidelity, fidelity_parts = self.fidelity_loss(output["fused"], visible, infrared, mask)
        alignment = self.alignment_loss(output, mask, granularities, distractor_mask)
        terms = torch.stack((localization, fidelity, alignment))
        bounded_log_vars = self.log_vars.clamp(self.log_var_min, self.log_var_max)
        total = (0.5 * torch.exp(-2.0 * bounded_log_vars) * terms + bounded_log_vars).sum()
        total = total + self.localization_floor_weight * localization
        return {
            "total": total,
            "localization": localization,
            "fidelity": fidelity,
            "alignment": alignment,
            **localization_parts,
            **fidelity_parts,
        }
