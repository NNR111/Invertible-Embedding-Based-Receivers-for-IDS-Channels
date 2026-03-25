from __future__ import annotations
import torch
import torch.nn.functional as F


def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = torch.matmul(z_a, z_b.t()) / temperature
    targets = torch.arange(z_a.size(0), device=z_a.device)
    loss_ab = F.cross_entropy(logits, targets)
    loss_ba = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_ab + loss_ba)
