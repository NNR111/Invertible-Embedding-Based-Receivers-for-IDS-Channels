from __future__ import annotations

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from ids_receiver.config import PAD_VALUE
from ids_receiver.data.coding import (
    random_message,
    conv_encode_bits,
    build_marker_patterns,
    marcode,
)
from ids_receiver.data.channel import ids_channel


def _to_1d_int_array(x):
    x = np.asarray(x).squeeze()
    return x.astype(np.int64)


def _to_scalar_float(x):
    x = np.asarray(x).squeeze()
    return float(x)


class IDSDataset(Dataset):
    def __init__(self, data_path: str):
        data = loadmat(data_path)

        self.all_msg = data["all_msg"]
        self.all_ub = data["all_ub"]
        self.all_x = data["all_x"]
        self.all_y = data["all_y"]
        self.all_ps = data["all_ps"]

        self.n_samples = self.all_msg.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        msg_bits = _to_1d_int_array(self.all_msg[idx, 0])
        coded_bits = _to_1d_int_array(self.all_ub[idx, 0])
        clean_syms = _to_1d_int_array(self.all_x[idx, 0])
        noisy_syms = _to_1d_int_array(self.all_y[idx, 0])
        p_sub = _to_scalar_float(self.all_ps[idx, 0])

        return {
            "msg_bits": torch.tensor(msg_bits, dtype=torch.float32),
            "coded_bits": torch.tensor(coded_bits, dtype=torch.float32),
            "clean_syms": torch.tensor(clean_syms, dtype=torch.long),
            "noisy_syms": torch.tensor(noisy_syms, dtype=torch.long),
            "p_sub": torch.tensor(p_sub, dtype=torch.float32),
        }


class SyntheticIDSDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        p_ins: float,
        p_del: float,
        p_sub_min: float,
        p_sub_max: float,
        use_marker: bool = False,
        marker: tuple[int, ...] = (0, 3),
        num_blocks: int = 20,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.p_ins = p_ins
        self.p_del = p_del
        self.p_sub_min = p_sub_min
        self.p_sub_max = p_sub_max
        self.seed = seed


        self.use_marker = use_marker
        self.marker = marker
        self.num_blocks = num_blocks

        self.T = 142
        self.Np = 7
        self.mp1, self.mp2 = build_marker_patterns(T=self.T, Np=self.Np)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)

        # Original information bits u
        msg = random_message(100, rng)

        # Terminated convolutional code
        coded_bits = conv_encode_bits(msg, g=(0o5, 0o7), K=3, terminate=True)

        # Clean transmitted 4-ary symbols 
        clean_syms = marcode(coded_bits, self.mp1, self.mp2)

        # p_sub is sampled per example
        p_sub = float(rng.uniform(self.p_sub_min, self.p_sub_max))

        # Noisy received symbols
        noisy_syms = ids_channel(
            clean_syms,
            self.p_ins,
            self.p_del,
            p_sub,
            vocab=4,
            rng=rng,
            l_max=0 if self.p_ins == 0 else 2,
        )

        return {
            "msg_bits": torch.tensor(msg, dtype=torch.float32),
            "coded_bits": torch.tensor(coded_bits, dtype=torch.float32),
            "clean_syms": torch.tensor(clean_syms, dtype=torch.long),
            "noisy_syms": torch.tensor(noisy_syms, dtype=torch.long),
            "p_sub": torch.tensor(p_sub, dtype=torch.float32),
        }


def collate_batch(batch):
    bsz = len(batch)

    clean_lens = [len(x["clean_syms"]) for x in batch]
    noisy_lens = [len(x["noisy_syms"]) for x in batch]

    max_clean = max(clean_lens)
    max_noisy = max(noisy_lens)


    clean_pad = torch.full((bsz, max_clean), PAD_VALUE, dtype=torch.long)
    noisy_pad = torch.full((bsz, max_noisy), PAD_VALUE, dtype=torch.long)

    msg_bits = torch.stack([x["msg_bits"] for x in batch], 0)
    coded_bits = torch.stack([x["coded_bits"] for x in batch], 0)
    p_sub = torch.stack([x["p_sub"] for x in batch], 0)

    for i, item in enumerate(batch):
        clean_pad[i, : len(item["clean_syms"])] = item["clean_syms"]
        noisy_pad[i, : len(item["noisy_syms"])] = item["noisy_syms"]

    return {
        "msg_bits": msg_bits,
        "coded_bits": coded_bits,
        "clean_syms": clean_pad,
        "clean_lens": torch.tensor(clean_lens, dtype=torch.long),
        "noisy_syms": noisy_pad,
        "noisy_lens": torch.tensor(noisy_lens, dtype=torch.long),
        "p_sub": p_sub,
    }
