import hashlib
import re
from typing import List, Tuple

import torch
from torch import nn


class HashTextEncoder(nn.Module):
    """Dependency-free deterministic encoder for tests, not final training."""

    def __init__(self, dim: int = 512, max_length: int = 48):
        super().__init__()
        self.output_dim = dim
        self.max_length = max_length
        basis = torch.exp(-torch.arange(dim, dtype=torch.float32) / max(dim - 1, 1) * 8.0)
        self.register_buffer("basis", basis, persistent=False)

    @staticmethod
    def _token_id(token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")

    def forward(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros(len(texts), self.max_length, device=device)
        mask = torch.zeros(len(texts), self.max_length, dtype=torch.bool, device=device)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[\w'-]+|[^\w\s]", text.lower(), flags=re.UNICODE)[: self.max_length]
            if not tokens:
                tokens = ["<empty>"]
            values = [self._token_id(t) % 1_000_003 for t in tokens]
            ids[row, : len(values)] = torch.tensor(values, device=device)
            mask[row, : len(values)] = True
        phase = ids.unsqueeze(-1) * self.basis.to(device).view(1, 1, -1)
        features = torch.sin(phase) + torch.cos(phase * 0.61803398875)
        return features, mask


class FrozenCLIPTextEncoder(nn.Module):
    """Frozen CLIP token encoder used by the released training pipeline."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 77):
        super().__init__()
        try:
            from transformers import AutoTokenizer, CLIPTextModel
        except ImportError as exc:
            raise ImportError("Install `transformers` to use the CLIP text backend.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.encoder = CLIPTextModel.from_pretrained(model_name, use_safetensors=True)
        self.encoder.requires_grad_(False)
        self.output_dim = self.encoder.config.hidden_size
        self.max_length = min(max_length, self.encoder.config.max_position_embeddings)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self.encoder.eval()
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        output = self.encoder(**batch).last_hidden_state
        return output, batch["attention_mask"].bool()


def build_text_encoder(config: dict) -> nn.Module:
    backend = config.get("backend", "clip").lower()
    if backend == "clip":
        return FrozenCLIPTextEncoder(
            model_name=config.get("model_name", "openai/clip-vit-base-patch32"),
            max_length=config.get("max_length", 77),
        )
    if backend == "hash":
        return HashTextEncoder(dim=config.get("dim", 512), max_length=config.get("max_length", 48))
    raise ValueError(f"Unsupported text backend: {backend}")
