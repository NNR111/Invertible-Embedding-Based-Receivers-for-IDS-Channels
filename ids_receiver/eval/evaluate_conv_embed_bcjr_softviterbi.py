from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from tqdm import tqdm

from ids_receiver.receivers.embed_bcjr_soft_viterbi_receiver import (
    MSG_LEN,
    EmbedNBMPrior,
    decode_one_embed_bcjr_softviterbi,
)


def random_message(msg_len: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(msg_len,), dtype=np.int64)


def parse_p_sub_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def evaluate_one(
    nbm_prior: EmbedNBMPrior,
    p_ins: float,
    p_del: float,
    p_sub: float,
    n_trials: int,
    prior_scale: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    bit_errors = 0
    frame_errors = 0

    for _ in tqdm(range(n_trials), leave=False):
        msg = random_message(MSG_LEN, rng)
        out = decode_one_embed_bcjr_softviterbi(
            msg_bits=msg,
            nbm_prior=nbm_prior,
            p_ins=p_ins,
            p_del=p_del,
            p_sub=p_sub,
            prior_scale=prior_scale,
            rng=rng,
        )
        nbe = int(np.sum(out["msg_hat"] != msg))
        bit_errors += nbe
        frame_errors += int(nbe > 0)

    BER = bit_errors / (n_trials * MSG_LEN)
    FER = frame_errors / n_trials
    return BER, FER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbm_ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--prior_scale", type=float, default=1.0)
    ap.add_argument("--p_ins", type=float, default=0.03)
    ap.add_argument("--p_del", type=float, default=0.03)
    ap.add_argument("--p_sub_list", type=str, default="0.01,0.02,0.03,0.04,0.05")
    ap.add_argument("--n_trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="runs_embed_bcjr_viterbi/eval_conv_embed_bcjr_softviterbi.csv")
    args = ap.parse_args()

    nbm_prior = EmbedNBMPrior(args.nbm_ckpt, device=args.device)

    rows = []
    for i, p_sub in enumerate(parse_p_sub_list(args.p_sub_list)):
        BER, FER = evaluate_one(
            nbm_prior=nbm_prior,
            p_ins=args.p_ins,
            p_del=args.p_del,
            p_sub=p_sub,
            n_trials=args.n_trials,
            prior_scale=args.prior_scale,
            seed=args.seed + 1000 * i,
        )
        rows.append({
            "p_ins": args.p_ins,
            "p_del": args.p_del,
            "p_sub": p_sub,
            "prior_scale": args.prior_scale,
            "BER": BER,
            "FER": FER,
        })
        print(
            f"p_ins={args.p_ins:.4f} p_del={args.p_del:.4f} p_sub={p_sub:.4f} "
            f"prior_scale={args.prior_scale:.3f} BER={BER:.6f} FER={FER:.6f}"
        )

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["p_ins", "p_del", "p_sub", "prior_scale", "BER", "FER"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
