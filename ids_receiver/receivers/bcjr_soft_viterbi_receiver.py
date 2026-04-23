from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

MSG_LEN = 100
TAIL_BITS = 2
CODE_BITS = 2 * (MSG_LEN + TAIL_BITS)


def octal_to_taps(g_oct: int, constraint_length: int) -> np.ndarray:
    """
    Convert generator polynomial from octal to binary taps.
    Example: 5(octal)->101, 7(octal)->111 for K=3.
    """
    val = int(str(g_oct), 8)
    bits = [(val >> i) & 1 for i in range(constraint_length - 1, -1, -1)]
    return np.array(bits, dtype=np.int64)


def convenc(
    msg_bits: Sequence[int],
    generators_octal: Sequence[int],
    constraint_length: int,
) -> np.ndarray:
    """
    Terminated convolutional encoder.
    for the K=3, g=[5 7] setting.
    """
    taps = [octal_to_taps(g, constraint_length) for g in generators_octal]

    state = np.zeros(constraint_length, dtype=np.int64)
    coded: List[int] = []

    for bit in np.asarray(msg_bits, dtype=np.int64):
        state[1:] = state[:-1]
        state[0] = bit
        for tap in taps:
            coded.append(int(np.sum(state * tap) % 2))

    return np.array(coded, dtype=np.int64)


@dataclass(frozen=True)
class Trellis:
    next_state: np.ndarray
    output_bits: np.ndarray
    output_sym: np.ndarray


def build_trellis(
    generators_octal: Sequence[int],
    constraint_length: int,
) -> Trellis:
    taps = [octal_to_taps(g, constraint_length) for g in generators_octal]

    memory = constraint_length - 1
    num_states = 2 ** memory
    n_out = len(generators_octal)

    next_state = np.zeros((num_states, 2), dtype=np.int64)
    output_bits = np.zeros((num_states, 2, n_out), dtype=np.int64)
    output_sym = np.zeros((num_states, 2), dtype=np.int64)

    for s in range(num_states):
        mem_bits = np.array(
            [(s >> (memory - 1 - i)) & 1 for i in range(memory)],
            dtype=np.int64,
        )

        for inp in (0, 1):
            full_reg = np.concatenate(([inp], mem_bits))

            out = np.array(
                [int(np.sum(full_reg * tap) % 2) for tap in taps],
                dtype=np.int64,
            )
            output_bits[s, inp, :] = out
            output_sym[s, inp] = int("".join(str(int(b)) for b in out), 2)

            new_mem = full_reg[:-1]
            ns = 0
            for b in new_mem:
                ns = (ns << 1) | int(b)
            next_state[s, inp] = ns

    return Trellis(
        next_state=next_state,
        output_bits=output_bits,
        output_sym=output_sym,
    )


def bit_cost_from_llr(llr: float, expected_bit: int) -> float:
    """
    Negative log-likelihood branch metric from input LLR.
    """
    if expected_bit == 0:
        return np.log1p(np.exp(-llr))
    return np.log1p(np.exp(llr))


def soft_input_viterbi(
    llr_vec: Sequence[float],
    generators_octal: Sequence[int],
    constraint_length: int,
    terminated: bool = True,
) -> np.ndarray:
    """
    Soft-input Viterbi decoder using bit LLRs.
    """
    llr = np.asarray(llr_vec, dtype=np.float64)

    trellis = build_trellis(generators_octal, constraint_length)
    n_out = len(generators_octal)

    if llr.size % n_out != 0:
        raise ValueError(
            "llr_vec length must be divisible by number of output bits per trellis step"
        )

    num_steps = llr.size // n_out
    rx = llr.reshape(num_steps, n_out)

    memory = constraint_length - 1
    num_states = 2 ** memory

    inf = 1e300
    path_metric = np.full((num_steps + 1, num_states), inf, dtype=np.float64)
    predecessor = np.full((num_steps + 1, num_states), -1, dtype=np.int64)
    decided_input = np.full((num_steps + 1, num_states), -1, dtype=np.int64)

    path_metric[0, 0] = 0.0

    for t in range(num_steps):
        for s in range(num_states):
            pm = path_metric[t, s]
            if pm >= inf / 10:
                continue

            for inp in (0, 1):
                ns = trellis.next_state[s, inp]
                out_bits = trellis.output_bits[s, inp]

                branch = sum(
                    bit_cost_from_llr(rx[t, j], int(out_bits[j]))
                    for j in range(n_out)
                )
                cand = pm + branch

                if cand < path_metric[t + 1, ns]:
                    path_metric[t + 1, ns] = cand
                    predecessor[t + 1, ns] = s
                    decided_input[t + 1, ns] = inp

    final_state = 0 if terminated else int(np.argmin(path_metric[num_steps]))

    decoded = np.zeros(num_steps, dtype=np.int64)
    s = final_state

    for t in range(num_steps, 0, -1):
        inp = decided_input[t, s]
        if inp < 0:
            inp = 0
        decoded[t - 1] = inp

        s_prev = predecessor[t, s]
        s = 0 if s_prev < 0 else s_prev

    return decoded


def rfz(
    mp1: np.ndarray,
    mp2: np.ndarray,
    ps: float,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build rho, f, zeta for q=4 marker-coded setup.
    """
    rho_rows = []

    for k in range(T):
        a = int(mp1[k])
        b = int(mp2[k])

        if a == 0 and b == 0:
            rhok = [1.0, 0.0, 0.0, 0.0]
        elif a == 0 and b == 1:
            rhok = [0.0, 1.0, 0.0, 0.0]
        elif a == 1 and b == 0:
            rhok = [0.0, 0.0, 1.0, 0.0]
        elif a == 1 and b == 1:
            rhok = [0.0, 0.0, 0.0, 1.0]
        elif a == 0 and b == -1:
            rhok = [0.5, 0.5, 0.0, 0.0]
        elif a == 1 and b == -1:
            rhok = [0.0, 0.0, 0.5, 0.5]
        elif a == -1 and b == 0:
            rhok = [0.5, 0.0, 0.5, 0.0]
        elif a == -1 and b == 1:
            rhok = [0.0, 0.5, 0.0, 0.5]
        else:
            rhok = [0.25, 0.25, 0.25, 0.25]

        rho_rows.append(rhok)

    rho = np.asarray(rho_rows, dtype=np.float64)

    q = 4
    f = np.full((q, q), ps / (q - 1), dtype=np.float64)
    np.fill_diagonal(f, 1.0 - ps)

    zeta = rho @ f
    return rho, f, zeta


def marcode(ub: np.ndarray, mp1: np.ndarray, mp2: np.ndarray) -> np.ndarray:
    """
    Insert convolutional-coded bits into marker pattern and map to q=4 symbols.
    """
    j1 = np.where(mp1 == -1)[0]
    j2 = np.where(mp2 == -1)[0]

    xb1 = mp1.copy()
    xb2 = mp2.copy()

    xb1[j1] = ub[: len(j1)]
    xb2[j2] = ub[len(j1): len(j1) + len(j2)]

    x = xb1 * 2 + xb2
    return x.astype(np.int64)


def ids_channel(
    x: np.ndarray,
    T: int,
    pi: float,
    pd: float,
    ps: float,
    l_max: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    q-ary IDS channel simulator.
    """
    q = 4
    y: List[int] = []

    for k in range(T):
        ins_num = 0
        rpi = int(rng.random() < pi)

        while ins_num < l_max and rpi == 1:
            ins_sym = int(np.ceil(rng.random() * q) - 1)
            y.append(ins_sym)
            ins_num += 1

        rpd = int(rng.random() < pd)
        rps = int(rng.random() < ps)

        if rpd == 1:
            pass
        elif rpd == 0 and rps == 0:
            y.append(int(x[k]))
        else:
            # force actual substitution offset in {1,2,3}
            sn = int(np.ceil((q - 1) * rng.random()))
            y.append(int((x[k] + sn) % q))

    return np.asarray(y, dtype=np.int64)


def _log_add_lookup(
    cur: float,
    cand: float,
    log_map_vec: np.ndarray,
    delta_step: float,
) -> float:
    """
    Approximate log(exp(cur)+exp(cand)) using lookup table.
    """
    t = abs(cand - cur)
    idx = min(len(log_map_vec) - 1, int(np.floor(t / delta_step)))
    return max(cur, cand) + float(log_map_vec[idx])


def FB_decode(
    y: np.ndarray,
    T: int,
    mu: np.ndarray,
    rho: np.ndarray,
    f: np.ndarray,
    zeta: np.ndarray,
    mp1: np.ndarray,
    mp2: np.ndarray,
    log_map_vec: np.ndarray,
    delta_step: float,
    l_max: int,
) -> np.ndarray:
    """
    Forward-backward decoder that outputs posterior probabilities
    for the convolutional coded bits.
    """
    Mval = 1e7
    R = len(y)
    q = 4
    no = 2 + l_max

    log_alf = -Mval * np.ones((T + 1, R + 2 * l_max + 3), dtype=np.float64)
    log_bet = -Mval * np.ones((T + 1, R + 2 * l_max + 3), dtype=np.float64)

    mu00 = mu[0, 0]

    log_alf[0, 0 + no] = 0.0
    if mu00 == 0:
        log_alf[1:T + 1, 0 + no] = -Mval
    else:
        log_alf[1:T + 1, 0 + no] = np.log(mu00) * np.arange(1, T + 1)

    log_bet[T, R + no] = 0.0
    if mu00 > 0:
        log_bet[0:T, R + no] = 0.0
    else:
        log_bet[0:T, R + no] = -Mval

    # forward recursion
    for k in range(1, T + 1):
        for n in range(1, R + 1):
            for l in range(l_max + 1):
                for b in (0, 1):
                    prev_col = n - l - b + no
                    if prev_col < 0 or prev_col >= log_alf.shape[1]:
                        continue

                    coef1 = mu[l, b] * (q ** (-l)) * (zeta[k - 1, y[n - 1]] ** b)

                    if coef1 == 0:
                        calf = -Mval
                    else:
                        calf = np.log(coef1) + log_alf[k - 1, prev_col]

                    log_alf[k, n + no] = _log_add_lookup(
                        log_alf[k, n + no],
                        calf,
                        log_map_vec,
                        delta_step,
                    )

    # backward recursion
    for k in range(T - 1, -1, -1):
        for n in range(R - 1, -1, -1):
            for l in range(l_max + 1):
                for b in (0, 1):
                    next_col = n + l + b + no
                    if next_col < 0 or next_col >= log_bet.shape[1]:
                        continue

                    y_idx = min(R - 1, n + l)
                    zky = zeta[k, y[y_idx]]
                    coef1 = mu[l, b] * (q ** (-l)) * (zky ** b)

                    if coef1 == 0:
                        cbet = -Mval
                    else:
                        cbet = np.log(coef1) + log_bet[k + 1, next_col]

                    log_bet[k, n + no] = _log_add_lookup(
                        log_bet[k, n + no],
                        cbet,
                        log_map_vec,
                        delta_step,
                    )

    # a posteriori symbol probabilities
    log_ap_prob = -Mval * np.ones((T, q), dtype=np.float64)

    for k in range(1, T + 1):
        upper_val = min(R, (l_max + 1) * (k - 1))

        for a in range(q):
            for l in range(l_max + 1):
                for b in (0, 1):
                    for n in range(0, upper_val + 1):
                        left_col = n + no
                        right_col = n + l + b + no

                        if left_col < 0 or left_col >= log_alf.shape[1]:
                            continue
                        if right_col < 0 or right_col >= log_bet.shape[1]:
                            continue

                        y_idx = min(R - 1, n + l)
                        fbb = f[a, y[y_idx]] ** b
                        coef1 = mu[l, b] * (q ** (-l)) * fbb

                        if coef1 == 0:
                            log_add_term = -Mval
                        else:
                            log_add_term = (
                                np.log(coef1)
                                + log_alf[k - 1, left_col]
                                + log_bet[k, right_col]
                            )

                        log_ap_prob[k - 1, a] = _log_add_lookup(
                            log_ap_prob[k - 1, a],
                            log_add_term,
                            log_map_vec,
                            delta_step,
                        )

    mmap = np.max(log_ap_prob)
    joint_prob = rho * np.exp(log_ap_prob - mmap)

    bit_prob1 = np.zeros((2, T), dtype=np.float64)
    bit_prob2 = np.zeros((2, T), dtype=np.float64)

    for k in range(T):
        bit_prob1[0, k] = joint_prob[k, 0] + joint_prob[k, 1]
        bit_prob1[1, k] = joint_prob[k, 2] + joint_prob[k, 3]

        bit_prob2[0, k] = joint_prob[k, 0] + joint_prob[k, 2]
        bit_prob2[1, k] = joint_prob[k, 1] + joint_prob[k, 3]

    j1 = np.where(mp1 == -1)[0]
    j2 = np.where(mp2 == -1)[0]

    eps = 1e-300
    llr1 = np.log(
        np.maximum(bit_prob1[1, j1], eps) / np.maximum(bit_prob1[0, j1], eps)
    )
    llr2 = np.log(
        np.maximum(bit_prob2[1, j2], eps) / np.maximum(bit_prob2[0, j2], eps)
    )

    bit_llrs = np.concatenate([llr1, llr2])
    p_ub_1 = np.exp(bit_llrs) / (1.0 + np.exp(bit_llrs))

    return p_ub_1


def decode_one_bcjr_softviterbi(
    msg_bits: np.ndarray,
    p_ins: float,
    p_del: float,
    p_sub: float,
    rng: np.random.Generator | None = None,
    T: int = 142,
    Np: int = 7,
    l_max: int = 2,
) -> dict:
    """
    End-to-end classical baseline:
    msg -> CC -> marker -> IDS -> FB/BCJR -> soft Viterbi -> decoded msg
    """
    rng = rng or np.random.default_rng()

    const_length = 3
    g = [5, 7]

    delta_step = 0.01
    xx_vec = np.arange(0.0, 10.0 + 1e-12, delta_step)
    log_map_vec = np.log(1.0 + np.exp(-xx_vec))

    if p_ins == 0:
        l_max = 0

    mu = np.array(
        [
            [p_del, 1 - p_del - p_ins],
            [0.0, 0.0],
            [p_ins, 0.0],
        ],
        dtype=np.float64,
    )
    mu = mu / np.sum(mu)

    mp1 = -1 * np.ones(T, dtype=np.int64)
    mp2 = -1 * np.ones(T, dtype=np.int64)

    mp1[Np::Np] = 1
    mp1[Np + 1::Np] = 0

    mp2[Np::Np] = 1
    mp2[Np + 1::Np] = 0

    rho, f, zeta = rfz(mp1, mp2, p_sub, T)

    msg = np.asarray(msg_bits, dtype=np.int64)
    msg_tb = np.concatenate([msg, np.array([0, 0], dtype=np.int64)])

    ub = convenc(msg_tb, g, const_length)
    x = marcode(ub, mp1, mp2)
    y = ids_channel(x, T, p_ins, p_del, p_sub, l_max, rng)

    p_ub_1 = FB_decode(
        y=y,
        T=T,
        mu=mu,
        rho=rho,
        f=f,
        zeta=zeta,
        mp1=mp1,
        mp2=mp2,
        log_map_vec=log_map_vec,
        delta_step=delta_step,
        l_max=l_max,
    )

    eps = 1e-300
    LLR_vec = -np.log(np.maximum(p_ub_1, eps) / np.maximum(1 - p_ub_1, eps))

    msg_dec = soft_input_viterbi(
        LLR_vec,
        generators_octal=g,
        constraint_length=const_length,
        terminated=True,
    )

    return {
        "msg_hat": msg_dec[: len(msg)],
        "LLR_vec": LLR_vec,
        "p_ub_1": p_ub_1,
        "x": x,
        "y": y,
        "ub": ub,
    }
