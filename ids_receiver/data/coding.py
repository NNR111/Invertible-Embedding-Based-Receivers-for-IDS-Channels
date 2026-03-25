from __future__ import annotations
import numpy as np
from typing import Iterable, List, Sequence, Tuple
from ids_receiver.config import MSG_LEN, CC_K, CC_G, TERMINATE


def bits_to_symbols_4ary(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64)
    if bits.ndim != 1 or len(bits) % 2 != 0:
        raise ValueError('bits must be 1D with even length')
    b = bits.reshape(-1, 2)
    return (b[:, 0] << 1) + b[:, 1]


def conv_encode_bits(msg_bits: np.ndarray,
                     g: Tuple[int, int] = CC_G,
                     K: int = CC_K,
                     terminate: bool = TERMINATE) -> np.ndarray:
    msg_bits = np.asarray(msg_bits, dtype=np.int64).tolist()
    state = [0] * (K - 1)
    out: List[int] = []
    seq = msg_bits + ([0] * (K - 1) if terminate else [])
    for bit in seq:
        reg = [bit] + state
        for poly in g:
            taps = [(poly >> i) & 1 for i in range(K - 1, -1, -1)]
            acc = 0
            for r, t in zip(reg, taps):
                if t:
                    acc ^= r
            out.append(acc)
        state = [bit] + state[:-1]
    return np.asarray(out, dtype=np.int64)


def encode_message_to_codeword(msg_bits: np.ndarray) -> np.ndarray:
    coded = conv_encode_bits(msg_bits)
    syms = bits_to_symbols_4ary(coded)
    return syms


def random_message(msg_len: int = MSG_LEN, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.integers(0, 2, size=(msg_len,), dtype=np.int64)


def insert_markers(symbols: np.ndarray,
                   use_marker: bool,
                   marker: Sequence[int],
                   num_blocks: int) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.int64)
    if not use_marker:
        return symbols.copy()
    if num_blocks < 2:
        return np.concatenate([symbols, np.asarray(marker, dtype=np.int64)], axis=0)
    marker = np.asarray(marker, dtype=np.int64)
    chunks = np.array_split(symbols, num_blocks)
    out = []
    for i, ch in enumerate(chunks):
        out.append(ch)
        if i < len(chunks) - 1:
            out.append(marker)
    return np.concatenate(out, axis=0)


def parse_marker(marker_str: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in marker_str.split(',') if x.strip() != '']
    if len(vals) == 0:
        raise ValueError('marker must have at least one symbol')
    if any(v < 0 or v > 3 for v in vals):
        raise ValueError('marker symbols must be in {0,1,2,3}')
    return tuple(vals)
