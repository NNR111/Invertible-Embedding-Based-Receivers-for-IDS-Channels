from __future__ import annotations
import os
import json
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def auto_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_ber_bler(msg_logits: torch.Tensor, msg_bits: torch.Tensor):
    pred = (torch.sigmoid(msg_logits) > 0.5).float()
    bit_err = (pred != msg_bits).float()
    ber = bit_err.mean().item()
    bler = (bit_err.sum(dim=1) > 0).float().mean().item()
    return ber, bler


def compute_code_ber(code_logits: torch.Tensor, coded_bits: torch.Tensor):
    pred = (torch.sigmoid(code_logits) > 0.5).float()
    err = (pred != coded_bits).float()
    return err.mean().item()


def save_json(obj: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
