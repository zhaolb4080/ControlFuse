import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .text import build_text_encoder


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class RestormerBlock(nn.Module):
    """Compact Restormer-style block with channel self-attention."""

    def __init__(self, dim: int, heads: int = 4, expansion: float = 2.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.norm1 = LayerNorm2d(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.project = nn.Conv2d(dim, dim, 1, bias=False)
        hidden = int(dim * expansion)
        self.norm2 = LayerNorm2d(dim)
        self.ffn_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.ffn_dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.ffn_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(self.norm1(x))).chunk(3, dim=1)
        q = q.reshape(b, self.heads, c // self.heads, h * w)
        k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)
        # Normalize in FP32 with an AMP-safe epsilon. The default 1e-12 epsilon
        # underflows in fp16 and can yield non-finite gradients for small norms.
        q = F.normalize(q.float(), dim=-1, eps=1e-6)
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        v = v.float()
        attn = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        attended = (attn @ v).reshape(b, c, h, w).to(x.dtype)
        x = x + self.project(attended)
        a, gate = self.ffn_dw(self.ffn_in(self.norm2(x))).chunk(2, dim=1)
        return x + self.ffn_out(F.gelu(a) * gate)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, blocks: int, heads: int):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.body = nn.Sequential(*[RestormerBlock(dim, heads) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.stem(x))


class SparseFeatureManifoldConverter(nn.Module):
    """Memory-safe FMC using local graph messages and pooled cross-modal GAT."""

    def __init__(self, image_dim: int, text_dim: int, manifold_dim: int, heads: int, layers: int, grid: int = 16):
        super().__init__()
        self.grid = grid
        self.vis_in = nn.Conv2d(image_dim, manifold_dim, 1)
        self.ir_in = nn.Conv2d(image_dim, manifold_dim, 1)
        self.text_in = nn.Linear(text_dim, manifold_dim)
        self.local_vis = nn.ModuleList()
        self.local_ir = nn.ModuleList()
        self.v_to_t = nn.ModuleList()
        self.t_to_v = nn.ModuleList()
        self.text_norm = nn.ModuleList()
        for _ in range(layers):
            self.local_vis.append(nn.Sequential(
                nn.Conv2d(manifold_dim, manifold_dim, 3, padding=1, groups=manifold_dim),
                nn.Conv2d(manifold_dim, manifold_dim, 1), nn.GELU(),
            ))
            self.local_ir.append(nn.Sequential(
                nn.Conv2d(manifold_dim, manifold_dim, 3, padding=1, groups=manifold_dim),
                nn.Conv2d(manifold_dim, manifold_dim, 1), nn.GELU(),
            ))
            self.v_to_t.append(nn.MultiheadAttention(manifold_dim, heads, batch_first=True))
            self.t_to_v.append(nn.MultiheadAttention(manifold_dim, heads, batch_first=True))
            self.text_norm.append(nn.LayerNorm(manifold_dim))

    def forward(
        self, vis: torch.Tensor, ir: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vis = self.vis_in(vis)
        ir = self.ir_in(ir)
        text = self.text_in(text)
        b, d, h, w = vis.shape
        gh, gw = min(self.grid, h), min(self.grid, w)
        for local_v, local_i, v_to_t, t_to_v, text_norm in zip(
            self.local_vis, self.local_ir, self.v_to_t, self.t_to_v, self.text_norm
        ):
            vis = vis + local_v(vis)
            ir = ir + local_i(ir)
            pooled = F.adaptive_avg_pool2d((vis + ir) * 0.5, (gh, gw)).flatten(2).transpose(1, 2)
            v_msg, _ = v_to_t(pooled, text, text, key_padding_mask=~text_mask)
            t_msg, _ = t_to_v(text, pooled, pooled)
            text = text_norm(text + t_msg)
            v_msg = v_msg.transpose(1, 2).reshape(b, d, gh, gw)
            v_msg = F.interpolate(v_msg, size=(h, w), mode="bilinear", align_corners=False)
            vis = vis + v_msg
            ir = ir + v_msg
        return vis, ir, text


def _laplacian_curvature(x: torch.Tensor) -> torch.Tensor:

    x = x.float()
    kernel = x.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    squared = F.conv2d(x, kernel, padding=1, groups=x.shape[1]).square().mean(1)

    return torch.sqrt(squared + 1e-6)


def _normalize_map(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        x = x.masked_fill(~mask, 0)
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    return (x - lo) / (hi - lo + 1e-6)


class CurvatureGuidedInteraction(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Conv2d(dim * 2, dim, 1)
        self.omega = nn.Parameter(torch.tensor(0.1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.token_selector = nn.Linear(dim, 1)
        location_hidden = max(dim // 2, 8)
        self.location_stem = nn.Sequential(
            nn.Conv2d(dim * 4 + 1, dim, 1),
            nn.GELU(),
        )
        self.location_detail = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.location_context = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.location_out = nn.Sequential(
            nn.Conv2d(dim * 2, location_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(location_hidden, 1, 1),
        )
        self.base = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU())
        self.delta = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 2, 3, padding=1), nn.GELU(), nn.Conv2d(dim * 2, dim, 3, padding=1)
        )

    def forward(
        self, vis: torch.Tensor, ir: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, d, h, w = vis.shape
        visual_curv = (_laplacian_curvature(vis) + _laplacian_curvature(ir)) * 0.5
        text_fp32 = text.float()
        text_curv = text_fp32.new_zeros(b, text.shape[1])
        if text.shape[1] > 2:
            text_difference = text_fp32[:, :-2] - 2 * text_fp32[:, 1:-1] + text_fp32[:, 2:]
            text_curv[:, 1:-1] = torch.sqrt(text_difference.square().sum(dim=-1) + 1e-6)
        visual_curv_n = _normalize_map(visual_curv.flatten(1))
        text_curv_n = _normalize_map(text_curv, text_mask)

        query_map = self.query(torch.cat([vis, ir], dim=1))
        query = F.normalize(query_map.flatten(2).transpose(1, 2).float(), dim=-1, eps=1e-6)
        normalized_text = F.normalize(text_fp32, dim=-1, eps=1e-6)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = torch.einsum("bnd,bld->bnl", query, normalized_text) * logit_scale
        bounded_omega = torch.tanh(self.omega) * 5.0
        logits = logits + bounded_omega * (visual_curv_n.unsqueeze(-1) + text_curv_n.unsqueeze(1))
        logits = logits.masked_fill(~text_mask.unsqueeze(1), -1e4)
        attention = logits.softmax(dim=-1)
        context = torch.einsum("bnl,bld->bnd", attention, text_fp32).transpose(1, 2).reshape(b, d, h, w)
        context = context.to(vis.dtype)

        token_scores = self.token_selector(text_fp32).squeeze(-1)
        token_scores = token_scores.masked_fill(~text_mask, -1e4)
        token_weights = token_scores.softmax(dim=-1)
        location_prior = (logits * token_weights.unsqueeze(1)).sum(-1).reshape(b, 1, h, w)
        detail = (vis - ir).abs()
        location_input = self.location_stem(
            torch.cat([vis, ir, detail, context, location_prior.to(vis.dtype)], dim=1)
        )
        location_residual = self.location_out(
            torch.cat(
                [self.location_detail(location_input), self.location_context(location_input)],
                dim=1,
            )
        )
        location_logits = location_prior + location_residual.float()
        gate = location_logits.sigmoid()
        base = self.base(torch.cat([vis, ir], dim=1))
        delta = self.delta(torch.cat([vis, ir, context], dim=1))
        fused = base + gate.to(delta.dtype) * delta
        aux = {
            "visual_curvature": visual_curv.flatten(1).mean(-1),
            "text_curvature": (text_curv * text_mask).sum(-1) / text_mask.sum(-1).clamp_min(1),
        }
        return fused, location_logits, aux


class MultiScaleDecoder(nn.Module):
    def __init__(self, dim: int, blocks: int, heads: int, out_channels: int = 3):
        super().__init__()
        branch_blocks = max(1, blocks // 3)

        self.full_branch = nn.Sequential(*[RestormerBlock(dim, heads) for _ in range(branch_blocks)])
        self.half_branch = nn.Sequential(*[RestormerBlock(dim, heads) for _ in range(branch_blocks)])
        self.quarter_branch = nn.Sequential(*[RestormerBlock(dim, heads) for _ in range(branch_blocks)])
        self.merge = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            *[RestormerBlock(dim, heads) for _ in range(max(1, blocks - 2 * branch_blocks))],
            nn.Conv2d(dim, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        full = self.full_branch(x)
        half = self.half_branch(F.avg_pool2d(x, 2, ceil_mode=True))
        quarter = self.quarter_branch(F.avg_pool2d(x, 4, ceil_mode=True))
        half = F.interpolate(half, size=(h, w), mode="bilinear", align_corners=False)
        quarter = F.interpolate(quarter, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.merge(torch.cat([full, half, quarter], dim=1)))


class ControlFuse(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        dim = model_cfg.get("dim", 48)
        manifold_dim = model_cfg.get("manifold_dim", 64)
        heads = model_cfg.get("heads", 4)
        self.text_encoder = build_text_encoder(config.get("text", {}))
        self.vis_encoder = ImageEncoder(3, dim, model_cfg.get("encoder_blocks", 3), heads)
        self.ir_encoder = ImageEncoder(1, dim, model_cfg.get("encoder_blocks", 3), heads)
        self.fmc = SparseFeatureManifoldConverter(
            dim,
            self.text_encoder.output_dim,
            manifold_dim,
            heads,
            model_cfg.get("fmc_layers", 2),
            model_cfg.get("fmc_grid", 16),
        )
        self.cgi = CurvatureGuidedInteraction(manifold_dim)
        self.decoder = MultiScaleDecoder(
            manifold_dim, model_cfg.get("decoder_blocks", 4), heads, model_cfg.get("out_channels", 3)
        )

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def _encode_text(self, texts: List[str], device: torch.device):
        return self.text_encoder(texts, device)

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        instructions: List[str],
        negative_instructions: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        text, text_mask = self._encode_text(instructions, visible.device)
        vis = self.vis_encoder(visible)
        ir = self.ir_encoder(infrared)
        vis_m, ir_m, text_m = self.fmc(vis, ir, text, text_mask)
        fused_features, location_logits, aux = self.cgi(vis_m, ir_m, text_m, text_mask)
        fused = self.decoder(fused_features)
        output = {
            "fused": fused,
            "location_logits": location_logits,
            "visual_feature_map": fused_features,
            "visual_embedding": F.normalize(
                fused_features.mean((2, 3)).float(), dim=-1, eps=1e-6
            ),
            "text_embedding": F.normalize(
                self._masked_mean(text_m, text_mask).float(), dim=-1, eps=1e-6
            ),
            **aux,
        }
        if negative_instructions is not None:
            neg_text, neg_mask = self._encode_text(negative_instructions, visible.device)
            neg_text = self.fmc.text_in(neg_text)
            output["negative_text_embedding"] = F.normalize(
                self._masked_mean(neg_text, neg_mask).float(), dim=-1, eps=1e-6
            )
        return output


def build_model(config: dict) -> ControlFuse:
    return ControlFuse(config)
