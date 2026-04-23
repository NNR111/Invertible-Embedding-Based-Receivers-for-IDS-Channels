from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
from tqdm import tqdm

from ids_receiver.receivers.embed_bcjr_soft_viterbi_receiver import (
    EmbedNBMPrior,
    MSG_LEN,
    decode_one_embed_bcjr_softviterbi,
)


# Parse comma-separated p_sub values
def parse_p_sub_list(s: str) -> list[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if len(vals) == 0:
        raise ValueError("p_sub_list must contain at least one value")
    return vals


# Append rows to CSV
def append_csv_rows(csv_path, rows, fieldnames):
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# Evaluate one p_sub setting using Python-generated data
def evaluate_one_setting(
    nbm_ckpt: str,
    p_ins: float,
    p_del: float,
    p_sub: float,
    n_trials: int,
    prior_scale: float,
    seed: int,
    device: str | None = None,
    progress_every: int = 10,
):
    # Wrapper around the trained encoder + NBM
    nbm_prior = EmbedNBMPrior(nbm_ckpt=nbm_ckpt, device=device)

    rng = np.random.default_rng(seed)

    bit_errors = 0
    frame_errors = 0

    t0 = time.time()

    for i in tqdm(range(n_trials), total=n_trials, leave=True, desc=f"p_sub={p_sub:.4f}"):
        # Generate one random 100-bit message
        msg = rng.integers(0, 2, size=(MSG_LEN,), dtype=np.int64)

        # End-to-end hybrid decode:
        # msg -> CC -> marker -> IDS -> NBM prior -> BCJR + Viterbi
        out = decode_one_embed_bcjr_softviterbi(
            msg_bits=msg,
            nbm_prior=nbm_prior,
            p_ins=p_ins,
            p_del=p_del,
            p_sub=p_sub,
            prior_scale=prior_scale,
            rng=rng,
        )

        msg_hat = np.asarray(out["msg_hat"], dtype=np.int64)
        nbe = int(np.sum(msg_hat != msg))

        bit_errors += nbe
        frame_errors += int(nbe > 0)

        if progress_every > 0 and ((i + 1) % progress_every == 0 or (i + 1) == n_trials):
            cur_ber = bit_errors / ((i + 1) * MSG_LEN)
            cur_fer = frame_errors / (i + 1)
            elapsed = time.time() - t0
            print(
                f"[{i+1}/{n_trials}] "
                f"partial BER={cur_ber:.6f} FER={cur_fer:.6f} "
                f"elapsed={elapsed:.2f}s"
            )

    ber = bit_errors / (n_trials * MSG_LEN)
    fer = frame_errors / n_trials
    total_time = time.time() - t0

    return ber, fer, total_time


def main():
    ap = argparse.ArgumentParser()

    # Hybrid receiver uses encoder+NBM checkpoint
    ap.add_argument("--nbm_ckpt", type=str, required=True)

    ap.add_argument("--p_ins", type=float, required=True)
    ap.add_argument("--p_del", type=float, required=True)
    ap.add_argument("--p_sub_list", type=str, required=True)

    ap.add_argument("--n_trials", type=int, default=3000)
    ap.add_argument("--prior_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--progress_every", type=int, default=10)

    ap.add_argument(
        "--out_csv",
        type=str,
        default="runs_embed/eval_embed_bcjr_softviterbi_python.csv",
    )

    args = ap.parse_args()

    p_sub_values = parse_p_sub_list(args.p_sub_list)

    rows = []

    for p_sub in p_sub_values:
        ber, fer, elapsed = evaluate_one_setting(
            nbm_ckpt=args.nbm_ckpt,
            p_ins=args.p_ins,
            p_del=args.p_del,
            p_sub=p_sub,
            n_trials=args.n_trials,
            prior_scale=args.prior_scale,
            seed=args.seed,
            device=args.device,
            progress_every=args.progress_every,
        )

        row = {
            "nbm_ckpt": args.nbm_ckpt,
            "p_ins": args.p_ins,
            "p_del": args.p_del,
            "p_sub": p_sub,
            "n_trials": args.n_trials,
            "prior_scale": args.prior_scale,
            "BER": ber,
            "FER": fer,
            "elapsed_sec": elapsed,
        }
        rows.append(row)

        print(
            f"p_ins={args.p_ins:.4f} "
            f"p_del={args.p_del:.4f} "
            f"p_sub={p_sub:.4f} "
            f"-> BER={ber:.6f} FER={fer:.6f}"
        )

    append_csv_rows(
        args.out_csv,
        rows,
        fieldnames=[
            "nbm_ckpt",
            "p_ins",
            "p_del",
            "p_sub",
            "n_trials",
            "prior_scale",
            "BER",
            "FER",
            "elapsed_sec",
        ],
    )

    print(f"\nResults appended to: {args.out_csv}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
