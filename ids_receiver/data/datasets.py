from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from ids_receiver.data.coding import random_message, encode_message_to_codeword, conv_encode_bits, insert_markers
from ids_receiver.data.channel import ids_channel
from ids_receiver.config import MSG_LEN, PAD_VALUE


class SyntheticIDSDataset(Dataset):
    def __init__(self,
                 n_samples: int,
                 p_ins: float,
                 p_del: float,
                 p_sub_min: float,
                 p_sub_max: float,
                 use_marker: bool = False,
                 marker: tuple[int, ...] = (0, 3, 0, 3),
                 num_blocks: int = 5,
                 seed: int = 0):
        self.n_samples = n_samples
        self.p_ins = p_ins
        self.p_del = p_del
        self.p_sub_min = p_sub_min
        self.p_sub_max = p_sub_max
        self.use_marker = use_marker
        self.marker = marker
        self.num_blocks = num_blocks
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        msg = random_message(MSG_LEN, rng)
        coded_bits = conv_encode_bits(msg)
        clean_payload = encode_message_to_codeword(msg)
        clean_syms = insert_markers(clean_payload, self.use_marker, self.marker, self.num_blocks)
        p_sub = float(rng.uniform(self.p_sub_min, self.p_sub_max))
        noisy_syms = ids_channel(clean_syms, self.p_ins, self.p_del, p_sub, vocab=4, rng=rng)
        return {
            'msg_bits': torch.tensor(msg, dtype=torch.float32),
            'coded_bits': torch.tensor(coded_bits, dtype=torch.float32),
            'clean_syms': torch.tensor(clean_syms, dtype=torch.long),
            'noisy_syms': torch.tensor(noisy_syms, dtype=torch.long),
            'p_sub': torch.tensor(p_sub, dtype=torch.float32),
        }


def collate_batch(batch):
    bsz = len(batch)
    clean_lens = [len(x['clean_syms']) for x in batch]
    noisy_lens = [len(x['noisy_syms']) for x in batch]
    max_clean = max(clean_lens)
    max_noisy = max(noisy_lens)
    clean_pad = torch.full((bsz, max_clean), PAD_VALUE, dtype=torch.long)
    noisy_pad = torch.full((bsz, max_noisy), PAD_VALUE, dtype=torch.long)
    msg_bits = torch.stack([x['msg_bits'] for x in batch], 0)
    coded_bits = torch.stack([x['coded_bits'] for x in batch], 0)
    p_sub = torch.stack([x['p_sub'] for x in batch], 0)
    for i, item in enumerate(batch):
        clean_pad[i, :len(item['clean_syms'])] = item['clean_syms']
        noisy_pad[i, :len(item['noisy_syms'])] = item['noisy_syms']
    return {
        'msg_bits': msg_bits,
        'coded_bits': coded_bits,
        'clean_syms': clean_pad,
        'clean_lens': torch.tensor(clean_lens, dtype=torch.long),
        'noisy_syms': noisy_pad,
        'noisy_lens': torch.tensor(noisy_lens, dtype=torch.long),
        'p_sub': p_sub,
    }
