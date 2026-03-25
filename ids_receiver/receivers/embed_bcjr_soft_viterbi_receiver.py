from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch

from ids_receiver.data.channel import ids_channel
from ids_receiver.data.coding import bits_to_symbols_4ary, conv_encode_bits, insert_markers
from ids_receiver.config import CC_G, CC_K, MSG_LEN, PAD_VALUE, TERMINATE
from ids_receiver.models.models_embed import FullEmbedModel

LOG_ZERO = -1e100


def logsumexp_list(vals: Iterable[float]) -> float:
    vals = list(vals)
    if not vals:
        return LOG_ZERO
    m = max(vals)
    if m <= LOG_ZERO / 2:
        return LOG_ZERO
    return m + math.log(sum(math.exp(v - m) for v in vals))


def qary_sub_prob(obs: int, tx: int, p_sub: float, vocab: int = 4) -> float:
    if obs == tx:
        return 1.0 - p_sub
    return p_sub / (vocab - 1)


def build_tx_with_marker_info(msg_bits: np.ndarray,
                              use_marker: bool,
                              marker: Sequence[int],
                              num_blocks: int):
    coded_bits = conv_encode_bits(msg_bits)
    data_syms = bits_to_symbols_4ary(coded_bits)
    if not use_marker:
        tx_syms = data_syms.copy()
        is_marker = np.zeros_like(tx_syms, dtype=bool)
        data_symbol_indices = np.arange(len(tx_syms), dtype=np.int64)
        return tx_syms, is_marker, data_symbol_indices, coded_bits

    marker = np.asarray(marker, dtype=np.int64)
    chunks = np.array_split(data_syms, num_blocks)
    out = []
    is_marker = []
    data_idx = []
    ptr = 0
    for i, ch in enumerate(chunks):
        out.extend(ch.tolist())
        is_marker.extend([False] * len(ch))
        data_idx.extend(range(ptr, ptr + len(ch)))
        ptr += len(ch)
        if i < len(chunks) - 1:
            out.extend(marker.tolist())
            is_marker.extend([True] * len(marker))
            data_idx.extend([-1] * len(marker))
    return np.asarray(out, dtype=np.int64), np.asarray(is_marker, dtype=bool), np.asarray(data_idx, dtype=np.int64), coded_bits


def transition_logprob_known(obs: np.ndarray, tx_symbol: int, p_ins: float, p_del: float, p_sub: float, vocab: int = 4) -> float:
    m = len(obs)
    if m == 0:
        p = p_del * (1.0 - p_ins)
    elif m == 1:
        p = p_del * p_ins * (1.0 / vocab)
        p += (1.0 - p_del) * (1.0 - p_ins) * qary_sub_prob(int(obs[0]), int(tx_symbol), p_sub, vocab=vocab)
    elif m == 2:
        p = (1.0 - p_del) * p_ins * qary_sub_prob(int(obs[0]), int(tx_symbol), p_sub, vocab=vocab) * (1.0 / vocab)
    else:
        p = 0.0
    return math.log(max(p, 1e-300))


def symbol_logpriors_from_nbm(code_logits: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert per-coded-bit NBM logits into per-data-symbol log priors.
    Symbol map: 0->00, 1->01, 2->10, 3->11.
    Assumes independence between the two coded bits inside each 4-ary symbol.
    """
    code_logits = np.asarray(code_logits, dtype=np.float64).reshape(-1)
    if len(code_logits) % 2 != 0:
        raise ValueError('Expected an even number of coded-bit logits.')
    p1 = 1.0 / (1.0 + np.exp(-np.clip(code_logits, -40.0, 40.0)))
    p0 = 1.0 - p1
    p = p0.reshape(-1, 2)
    q = p1.reshape(-1, 2)
    sym_prob = np.stack([
        p[:, 0] * p[:, 1],
        p[:, 0] * q[:, 1],
        q[:, 0] * p[:, 1],
        q[:, 0] * q[:, 1],
    ], axis=1)
    sym_prob = np.clip(sym_prob, eps, 1.0)
    sym_prob /= sym_prob.sum(axis=1, keepdims=True)
    return np.log(sym_prob)


class EmbedNBMPrior:
    def __init__(self, nbm_ckpt: str, device: str | None = None):
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = FullEmbedModel().to(self.device)
        ckpt = torch.load(nbm_ckpt, map_location='cpu')
        if 'encoder_state' in ckpt:
            self.model.encoder.load_state_dict(ckpt['encoder_state'])
        if 'nbm_state' in ckpt:
            self.model.nbm.load_state_dict(ckpt['nbm_state'])
        elif 'model_state' in ckpt:
            # fallback if the user later saves the entire model
            self.model.load_state_dict(ckpt['model_state'], strict=False)
        else:
            raise KeyError('Checkpoint must contain encoder_state and nbm_state (or model_state).')
        self.model.eval()

    @torch.no_grad()
    def code_logits_from_noisy_syms(self, noisy_syms: np.ndarray) -> np.ndarray:
        x = torch.full((1, len(noisy_syms)), PAD_VALUE, dtype=torch.long, device=self.device)
        if len(noisy_syms) > 0:
            x[0, :len(noisy_syms)] = torch.as_tensor(noisy_syms, dtype=torch.long, device=self.device)
        lens = torch.tensor([len(noisy_syms)], dtype=torch.long, device=self.device)
        code_logits, _ = self.model.forward_nbm(x, lens)
        return code_logits[0].detach().cpu().numpy().astype(np.float64)


def bcjr_symbol_app_with_prior(tx_syms: np.ndarray,
                               is_marker: np.ndarray,
                               data_symbol_indices: np.ndarray,
                               y: np.ndarray,
                               p_ins: float,
                               p_del: float,
                               p_sub: float,
                               symbol_logpriors: np.ndarray,
                               prior_scale: float = 1.0,
                               vocab: int = 4):
    tx_syms = np.asarray(tx_syms, dtype=np.int64)
    is_marker = np.asarray(is_marker, dtype=bool)
    data_symbol_indices = np.asarray(data_symbol_indices, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    T = len(tx_syms)
    R = len(y)

    alpha = np.full((R + 1, T + 1), LOG_ZERO, dtype=np.float64)
    beta = np.full((R + 1, T + 1), LOG_ZERO, dtype=np.float64)
    alpha[0, 0] = 0.0
    beta[R, T] = 0.0

    for k in range(T):
        max_n = min(2 * k, R)
        for n in range(max_n + 1):
            a = alpha[n, k]
            if a <= LOG_ZERO / 2:
                continue
            for m in (0, 1, 2):
                n2 = n + m
                if n2 > R:
                    continue
                obs = y[n:n2]
                if is_marker[k]:
                    lp = transition_logprob_known(obs, int(tx_syms[k]), p_ins, p_del, p_sub, vocab=vocab)
                else:
                    di = int(data_symbol_indices[k])
                    vals = []
                    for s in range(vocab):
                        vals.append(
                            transition_logprob_known(obs, s, p_ins, p_del, p_sub, vocab=vocab)
                            + prior_scale * float(symbol_logpriors[di, s])
                        )
                    lp = logsumexp_list(vals)
                alpha[n2, k + 1] = logsumexp_list([alpha[n2, k + 1], a + lp])

    for k in range(T - 1, -1, -1):
        for n in range(R + 1):
            vals_total = []
            for m in (0, 1, 2):
                n2 = n + m
                if n2 > R:
                    continue
                b = beta[n2, k + 1]
                if b <= LOG_ZERO / 2:
                    continue
                obs = y[n:n2]
                if is_marker[k]:
                    lp = transition_logprob_known(obs, int(tx_syms[k]), p_ins, p_del, p_sub, vocab=vocab)
                    vals_total.append(lp + b)
                else:
                    di = int(data_symbol_indices[k])
                    inner = []
                    for s in range(vocab):
                        inner.append(
                            transition_logprob_known(obs, s, p_ins, p_del, p_sub, vocab=vocab)
                            + prior_scale * float(symbol_logpriors[di, s])
                        )
                    vals_total.append(logsumexp_list(inner) + b)
            beta[n, k] = logsumexp_list(vals_total)

    sym_logpost = np.full((T, vocab), -1e9, dtype=np.float64)
    for k in range(T):
        if is_marker[k]:
            sym_logpost[k, int(tx_syms[k])] = 0.0
            continue
        di = int(data_symbol_indices[k])
        for s in range(vocab):
            vals = []
            max_n = min(2 * k, R)
            for n in range(max_n + 1):
                a = alpha[n, k]
                if a <= LOG_ZERO / 2:
                    continue
                for m in (0, 1, 2):
                    n2 = n + m
                    if n2 > R:
                        continue
                    b = beta[n2, k + 1]
                    if b <= LOG_ZERO / 2:
                        continue
                    obs = y[n:n2]
                    lp = transition_logprob_known(obs, s, p_ins, p_del, p_sub, vocab=vocab)
                    vals.append(a + lp + prior_scale * float(symbol_logpriors[di, s]) + b)
            sym_logpost[k, s] = logsumexp_list(vals)
        z = logsumexp_list(sym_logpost[k, :].tolist())
        sym_logpost[k, :] -= z

    return alpha, beta, sym_logpost


def symbol_logpost_to_bit_llrs(sym_logpost: np.ndarray) -> np.ndarray:
    sym_logpost = np.asarray(sym_logpost, dtype=np.float64)
    llrs = np.zeros((sym_logpost.shape[0] * 2,), dtype=np.float64)
    for i, lp in enumerate(sym_logpost):
        l_b0_0 = logsumexp_list([lp[0], lp[1]])
        l_b0_1 = logsumexp_list([lp[2], lp[3]])
        l_b1_0 = logsumexp_list([lp[0], lp[2]])
        l_b1_1 = logsumexp_list([lp[1], lp[3]])
        llrs[2 * i] = l_b0_0 - l_b0_1
        llrs[2 * i + 1] = l_b1_0 - l_b1_1
    return llrs


def conv_next_state_and_output(state: int, bit: int, g=CC_G, K: int = CC_K):
    state_bits = [(state >> i) & 1 for i in range(K - 2, -1, -1)]
    reg = [bit] + state_bits
    out = []
    for poly in g:
        taps = [(poly >> i) & 1 for i in range(K - 1, -1, -1)]
        acc = 0
        for r, t in zip(reg, taps):
            if t:
                acc ^= r
        out.append(acc)
    new_state_bits = [bit] + state_bits[:-1]
    new_state = 0
    for b in new_state_bits:
        new_state = (new_state << 1) | b
    return new_state, tuple(out)


def soft_input_viterbi_decode(bit_llrs: np.ndarray,
                              msg_len: int = MSG_LEN,
                              g=CC_G,
                              K: int = CC_K,
                              terminate: bool = TERMINATE) -> np.ndarray:
    bit_llrs = np.asarray(bit_llrs, dtype=np.float64)
    n_steps = msg_len + (K - 1 if terminate else 0)
    expected_bits = 2 * n_steps
    if len(bit_llrs) != expected_bits:
        raise ValueError(f'Expected {expected_bits} coded-bit LLRs, got {len(bit_llrs)}')

    n_states = 2 ** (K - 1)
    pm = np.full((n_steps + 1, n_states), -1e100, dtype=np.float64)
    prev_state = np.full((n_steps + 1, n_states), -1, dtype=np.int64)
    prev_bit = np.full((n_steps + 1, n_states), -1, dtype=np.int64)
    pm[0, 0] = 0.0

    for t in range(n_steps):
        L0 = bit_llrs[2 * t]
        L1 = bit_llrs[2 * t + 1]
        for s in range(n_states):
            cur = pm[t, s]
            if cur <= -1e90:
                continue
            for b in (0, 1):
                if terminate and t >= msg_len and b != 0:
                    continue
                ns, out = conv_next_state_and_output(s, b, g=g, K=K)
                bm = 0.5 * ((1 - 2 * out[0]) * L0 + (1 - 2 * out[1]) * L1)
                cand = cur + bm
                if cand > pm[t + 1, ns]:
                    pm[t + 1, ns] = cand
                    prev_state[t + 1, ns] = s
                    prev_bit[t + 1, ns] = b

    end_state = 0 if terminate else int(np.argmax(pm[n_steps]))
    bits = np.zeros((n_steps,), dtype=np.int64)
    s = end_state
    for t in range(n_steps, 0, -1):
        bits[t - 1] = prev_bit[t, s]
        s = prev_state[t, s]
        if s < 0 and t > 1:
            raise RuntimeError('Viterbi traceback failed.')
    return bits[:msg_len]


def decode_one_embed_bcjr_softviterbi(msg_bits: np.ndarray,
                                      nbm_prior: EmbedNBMPrior,
                                      p_ins: float,
                                      p_del: float,
                                      p_sub: float,
                                      use_marker: bool,
                                      marker: Sequence[int],
                                      num_blocks: int,
                                      prior_scale: float = 1.0,
                                      rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    tx_syms, is_marker, data_symbol_indices, coded_bits = build_tx_with_marker_info(
        msg_bits=msg_bits,
        use_marker=use_marker,
        marker=marker,
        num_blocks=num_blocks,
    )
    tx_seq = insert_markers(bits_to_symbols_4ary(coded_bits), use_marker=use_marker, marker=marker, num_blocks=num_blocks)
    noisy_syms = ids_channel(tx_seq, p_ins=p_ins, p_del=p_del, p_sub=p_sub, rng=rng)

    code_logits = nbm_prior.code_logits_from_noisy_syms(noisy_syms)
    symbol_logpriors = symbol_logpriors_from_nbm(code_logits)

    _, _, sym_logpost = bcjr_symbol_app_with_prior(
        tx_syms=tx_syms,
        is_marker=is_marker,
        data_symbol_indices=data_symbol_indices,
        y=noisy_syms,
        p_ins=p_ins,
        p_del=p_del,
        p_sub=p_sub,
        symbol_logpriors=symbol_logpriors,
        prior_scale=prior_scale,
        vocab=4,
    )

    data_sym_logpost = sym_logpost[~is_marker]
    bit_llrs = symbol_logpost_to_bit_llrs(data_sym_logpost)
    msg_hat = soft_input_viterbi_decode(bit_llrs)
    return {
        'msg_hat': msg_hat,
        'bit_llrs': bit_llrs,
        'code_logits_nbm': code_logits,
        'symbol_logpriors': symbol_logpriors,
        'sym_logpost': sym_logpost,
        'noisy_syms': noisy_syms,
        'tx_syms': tx_syms,
        'coded_bits': coded_bits,
    }
