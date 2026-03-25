from __future__ import annotations
import numpy as np


def ids_channel(symbols: np.ndarray,
                p_ins: float,
                p_del: float,
                p_sub: float,
                vocab: int = 4,
                rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = []
    for s in symbols.tolist():
        if rng.random() < p_del:
            if rng.random() < p_ins:
                out.append(int(rng.integers(0, vocab)))
            continue
        cur = int(s)
        if rng.random() < p_sub:
            candidates = [x for x in range(vocab) if x != cur]
            cur = int(rng.choice(candidates))
        out.append(cur)
        if rng.random() < p_ins:
            out.append(int(rng.integers(0, vocab)))
    return np.asarray(out, dtype=np.int64)
