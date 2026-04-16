from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class UnifiedUnmixingConfig:
    input_dim: int = 1024
    n_main_endmembers: int = 3
    n_residual_endmembers: int = 2
    hidden_dim: int = 256
    latent_dim: int = 128
    mode: str = "semi"  # blind, fixed, semi, weak, full


class UnifiedRamanUnmixingNet(nn.Module):
    def __init__(self, config: UnifiedUnmixingConfig, endmember_anchors: Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        self.n_total_endmembers = config.n_main_endmembers + config.n_residual_endmembers

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, self.n_total_endmembers),
        )

        self.main_endmembers = nn.Parameter(torch.rand(config.n_main_endmembers, config.input_dim))
        self.residual_endmembers = nn.Parameter(torch.rand(config.n_residual_endmembers, config.input_dim))

        if endmember_anchors is None:
            self.register_buffer("endmember_anchors", torch.empty(0))
        else:
            self.register_buffer("endmember_anchors", endmember_anchors.float())

        self.label_head = nn.Sequential(
            nn.Linear(self.n_total_endmembers + config.n_main_endmembers, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 3),
        )

        if config.mode in {"fixed", "semi", "weak", "full"} and self.endmember_anchors.numel() > 0:
            with torch.no_grad():
                self.main_endmembers.copy_(self.endmember_anchors[: config.n_main_endmembers])
        if config.mode == "fixed" and self.endmember_anchors.numel() > 0:
            self.main_endmembers.requires_grad_(False)

    def get_endmember_matrix(self) -> Tensor:
        main = torch.relu(self.main_endmembers)
        residual = torch.relu(self.residual_endmembers)
        return torch.cat([main, residual], dim=0)

    def forward(
        self,
        x: Tensor,
        microplastic_mask: Tensor | None = None,
        allowed_main_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        abundance_logits = self.encoder(x)
        if allowed_main_mask is not None:
            main_logits = abundance_logits[:, : self.config.n_main_endmembers]
            invalid_mask = allowed_main_mask[:, : self.config.n_main_endmembers] <= 0
            main_logits = main_logits.masked_fill(invalid_mask, -1e9)
            abundance_logits = torch.cat([main_logits, abundance_logits[:, self.config.n_main_endmembers :]], dim=-1)
        abundances = torch.softmax(abundance_logits, dim=-1)
        endmembers = self.get_endmember_matrix()
        reconstruction = abundances @ endmembers
        if microplastic_mask is None:
            microplastic_mask = torch.zeros(
                (x.shape[0], self.config.n_main_endmembers),
                dtype=x.dtype,
                device=x.device,
            )
        label_features = torch.cat([abundances, microplastic_mask], dim=-1)
        label_logits = self.label_head(label_features)
        microplastic_score = torch.sum(abundances[:, : self.config.n_main_endmembers] * microplastic_mask, dim=-1, keepdim=True)
        return {
            "reconstruction": reconstruction,
            "abundances": abundances,
            "endmembers": endmembers,
            "label_logits": label_logits,
            "microplastic_score": microplastic_score,
        }
