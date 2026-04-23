from __future__ import annotations

import numpy as np
from typing import List, Sequence, Tuple


# Convert pairs of bits into 4-ary symbols:
# 00->0, 01->1, 10->2, 11->3
def bits_to_symbols_4ary(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64)
    if bits.ndim != 1 or len(bits) % 2 != 0:
        raise ValueError("bits must be 1D with even length")
    b = bits.reshape(-1, 2)
    return (b[:, 0] << 1) + b[:, 1]


# Generate one random binary message
def random_message(
    msg_len: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.integers(0, 2, size=(msg_len,), dtype=np.int64)


# convolutional encoder
# - constraint length K=3
# - generators = (5, 7) in octal
# - append tail bits [0, 0]
def conv_encode_bits(
    msg_bits: np.ndarray,
    g: Tuple[int, int] = (0o5, 0o7),
    K: int = 3,
    terminate: bool = True,
) -> np.ndarray:
    msg_bits = np.asarray(msg_bits, dtype=np.int64)
    if msg_bits.ndim != 1:
        raise ValueError("msg_bits must be 1D")

    seq = msg_bits.tolist()
    if terminate:
        seq = seq + [0] * (K - 1)

    state = [0] * (K - 1)
    out: List[int] = []

    for bit in seq:
        # Current register content: [current_bit, previous_bit_1, previous_bit_2]
        reg = [int(bit)] + state

        for poly in g:
            taps = [(poly >> i) & 1 for i in range(K - 1, -1, -1)]
            acc = 0
            for r, t in zip(reg, taps):
                if t:
                    acc ^= r
            out.append(acc)

        state = [int(bit)] + state[:-1]

    return np.asarray(out, dtype=np.int64)



def encode_message_to_codeword(msg_bits: np.ndarray) -> np.ndarray:
    coded = conv_encode_bits(msg_bits)
    syms = bits_to_symbols_4ary(coded)
    return syms



def parse_marker(marker_str: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in marker_str.split(",") if x.strip() != ""]
    if len(vals) == 0:
        raise ValueError("marker must have at least one symbol")
    if any(v < 0 or v > 3 for v in vals):
        raise ValueError("marker symbols must be in {0,1,2,3}")
    return tuple(vals)



def insert_markers(
    symbols: np.ndarray,
    use_marker: bool,
    marker: Sequence[int],
    num_blocks: int,
) -> np.ndarray:
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


# Build the two binary marker patterns 
# T = 142, Np = 7
def build_marker_patterns(T: int = 142, Np: int = 7) -> tuple[np.ndarray, np.ndarray]:
    mp1 = -1 * np.ones((T,), dtype=np.int64)
    mp2 = -1 * np.ones((T,), dtype=np.int64)

    # mp1(Np+1:Np:end) = 1; mp1(Np+2:Np:end) = 0;
    # mp2(Np+1:Np:end) = 1; mp2(Np+2:Np:end) = 0;
    mp1[Np:T:Np] = 1
    mp1[Np + 1:T:Np] = 0

    mp2[Np:T:Np] = 1
    mp2[Np + 1:T:Np] = 0

    return mp1, mp2


# fill -1 positions of mp1/mp2 using coded bits, then merge into 4-ary symbols
def marcode(ub: np.ndarray, mp1: np.ndarray, mp2: np.ndarray) -> np.ndarray:
    ub = np.asarray(ub, dtype=np.int64).reshape(-1)
    mp1 = np.asarray(mp1, dtype=np.int64).reshape(-1)
    mp2 = np.asarray(mp2, dtype=np.int64).reshape(-1)

    j1 = np.where(mp1 == -1)[0]
    j2 = np.where(mp2 == -1)[0]

    if len(ub) != len(j1) + len(j2):
        raise ValueError(
            f"ub length mismatch: got {len(ub)}, expected {len(j1) + len(j2)}"
        )

    xb1 = mp1.copy()
    xb2 = mp2.copy()

    xb1[j1] = ub[: len(j1)]
    xb2[j2] = ub[len(j1): len(j1) + len(j2)]

    # Combine two binary streams into one 4-ary stream
    x = xb1 * 2 + xb2
    return x.astype(np.int64)
