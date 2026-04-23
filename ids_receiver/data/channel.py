from __future__ import annotations

import numpy as np


def ids_channel(
    symbols: np.ndarray,
    p_ins: float,
    p_del: float,
    p_sub: float,
    vocab: int = 4,
    rng: np.random.Generator | None = None,
    l_max: int = 2,
) -> np.ndarray:
    if vocab != 4:
        raise ValueError("This MATLAB-exact path expects vocab=4")

    rng = rng or np.random.default_rng()
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)

    if p_ins == 0:
        l_max = 0

    out: list[int] = []

    for s in symbols.tolist():
        # MATLAB inserts random symbols before processing the current symbol
        ins_num = 0
        rpi = int(rng.random() < p_ins)

        while ins_num < l_max and rpi == 1:
            ins_sym = int(rng.integers(0, vocab))
            out.append(ins_sym)
            ins_num += 1

        # Then decide delete / keep / substitute for the true symbol
        rpd = int(rng.random() < p_del)
        rps = int(rng.random() < p_sub)

        if rpd == 1:
            pass
        elif rpd == 0 and rps == 0:
            out.append(int(s))
        else:
            # MATLAB uses modular addition for substitution
            sn = int(rng.integers(0, vocab))
            out.append(int((int(s) + sn) % vocab))

    return np.asarray(out, dtype=np.int64)
