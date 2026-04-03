from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np
import torch

from ids_receiver.models.models_embed import FullEmbedModel
from ids_receiver.receivers.bcjr_soft_viterbi_receiver import (
    MSG_LEN,
    FB_decode as FB_decode_classical,
    convenc,
    ids_channel,
    marcode,
    rfz,
    soft_input_viterbi,
)

PAD_VALUE = 4
LOG_ZERO = -1e100


def logsumexp_list(vals: Iterable[float]) -> float:
    vals = list(vals)
    if not vals:
        return LOG_ZERO
    m = max(vals)
    if m <= LOG_ZERO / 2:
        return LOG_ZERO
    return m + math.log(sum(math.exp(v - m) for v in vals))


def symbol_logprior_from_nbm(code_logits: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    code_logits = np.asarray(code_logits, dtype=np.float64).reshape(-1)
    if len(code_logits) % 2 != 0:
        raise ValueError("Expected an even number of coded-bit logits.")
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
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = FullEmbedModel().to(self.device)
        ckpt = torch.load(nbm_ckpt, map_location="cpu")
        if "encoder_state" in ckpt:
            self.model.encoder.load_state_dict(ckpt["encoder_state"])
        if "nbm_state" in ckpt:
            self.model.nbm.load_state_dict(ckpt["nbm_state"])
        elif "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            raise KeyError("Checkpoint must contain encoder_state and nbm_state (or model_state).")
        self.model.eval()

    @torch.no_grad()
    def code_logits(self, y: np.ndarray) -> np.ndarray:
        x = torch.full((1, len(y)), PAD_VALUE, dtype=torch.long, device=self.device)
        if len(y) > 0:
            x[0, :len(y)] = torch.as_tensor(y, dtype=torch.long, device=self.device)
        lens = torch.tensor([len(y)], dtype=torch.long, device=self.device)
        code_logits, _ = self.model.forward_nbm(x, lens)
        return code_logits[0].detach().cpu().numpy().astype(np.float64)


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
    symbol_logprior: np.ndarray,
    prior_scale: float = 1.0,
) -> np.ndarray:
    p_classical = FB_decode_classical(
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
    llr_classical = np.log(np.maximum(p_classical, eps) / np.maximum(1.0 - p_classical, eps))

    sym_llr = np.asarray(symbol_logprior, dtype=np.float64)
    llr_prior = np.zeros((sym_llr.shape[0] * 2,), dtype=np.float64)
    for i, lp in enumerate(sym_llr):
        l_b0_0 = logsumexp_list([lp[0], lp[1]])
        l_b0_1 = logsumexp_list([lp[2], lp[3]])
        l_b1_0 = logsumexp_list([lp[0], lp[2]])
        l_b1_1 = logsumexp_list([lp[1], lp[3]])
        llr_prior[2 * i] = l_b0_1 - l_b0_0
        llr_prior[2 * i + 1] = l_b1_1 - l_b1_0

    llr_hybrid = llr_classical + prior_scale * llr_prior
    p_ub_1 = np.exp(llr_hybrid) / (1.0 + np.exp(llr_hybrid))
    return p_ub_1


def decode_one_embed_bcjr_softviterbi(
    msg_bits: np.ndarray,
    nbm_prior: EmbedNBMPrior,
    p_ins: float,
    p_del: float,
    p_sub: float,
    prior_scale: float = 1.0,
    rng: np.random.Generator | None = None,
    T: int = 142,
    Np: int = 7,
    l_max: int = 2,
) -> dict:
    rng = rng or np.random.default_rng()

    const_length = 3
    g = [5, 7]

    delta_step = 0.01
    xx_vec = np.arange(0.0, 10.0 + 1e-12, delta_step)
    log_map_vec = np.log(1.0 + np.exp(-xx_vec))

    if p_ins == 0:
        l_max = 0

    mu = np.array([[p_del, 1 - p_del - p_ins], [0.0, 0.0], [p_ins, 0.0]], dtype=np.float64)
    mu = mu / np.sum(mu)

    mp1 = -1 * np.ones(T, dtype=np.int64)
    mp2 = -1 * np.ones(T, dtype=np.int64)
    mp1[Np::Np] = 1
    mp1[Np + 1 :: Np] = 0
    mp2[Np::Np] = 1
    mp2[Np + 1 :: Np] = 0

    rho, f, zeta = rfz(mp1, mp2, p_sub, T)

    msg = np.asarray(msg_bits, dtype=np.int64)
    msg_tb = np.concatenate([msg, np.array([0, 0], dtype=np.int64)])

    ub = convenc(msg_tb, g, const_length)
    x = marcode(ub, mp1, mp2)
    y = ids_channel(x, T, p_ins, p_del, p_sub, l_max, rng)

    code_logits = nbm_prior.code_logits(y)
    symbol_logprior = symbol_logprior_from_nbm(code_logits)

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
        symbol_logprior=symbol_logprior,
        prior_scale=prior_scale,
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
        "code_logits": code_logits,
        "symbol_logprior": symbol_logprior,
        "x": x,
        "y": y,
        "ub": ub,
    }
