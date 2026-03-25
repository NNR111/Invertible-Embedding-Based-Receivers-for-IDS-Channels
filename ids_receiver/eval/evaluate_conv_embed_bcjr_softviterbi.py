from __future__ import annotations

import argparse
import csv
import os
from typing import List

import numpy as np
from tqdm import tqdm

from ids_receiver.data.coding import parse_marker, random_message
from ids_receiver.config import MSG_LEN, DEFAULT_MARKER, DEFAULT_NUM_BLOCKS
from ids_receiver.receivers.embed_bcjr_soft_viterbi_receiver import EmbedNBMPrior, decode_one_embed_bcjr_softviterbi


def evaluate_one(nbm_prior: EmbedNBMPrior,
                 p_ins: float,
                 p_del: float,
                 p_sub: float,
                 n_trials: int,
                 use_marker: bool,
                 marker,
                 num_blocks: int,
                 prior_scale: float,
                 seed: int):
    rng = np.random.default_rng(seed)
    total_bit_err = 0
    total_block_err = 0
    for _ in tqdm(range(n_trials), leave=False):
        msg = random_message(MSG_LEN, rng)
        out = decode_one_embed_bcjr_softviterbi(
            msg_bits=msg,
            nbm_prior=nbm_prior,
            p_ins=p_ins,
            p_del=p_del,
            p_sub=p_sub,
            use_marker=use_marker,
            marker=marker,
            num_blocks=num_blocks,
            prior_scale=prior_scale,
            rng=rng,
        )
        err = int(np.sum(out['msg_hat'] != msg))
        total_bit_err += err
        total_block_err += int(err > 0)
    ber = total_bit_err / (n_trials * MSG_LEN)
    bler = total_block_err / n_trials
    return ber, bler


def parse_p_sub_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nbm_ckpt', type=str, required=True)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--prior_scale', type=float, default=1.0)
    ap.add_argument('--p_ins', type=float, default=0.01)
    ap.add_argument('--p_del', type=float, default=0.01)
    ap.add_argument('--p_sub_list', type=str, default='0.01,0.02,0.03,0.04,0.05')
    ap.add_argument('--n_trials', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--use_marker', type=int, default=1)
    ap.add_argument('--marker', type=str, default=','.join(str(x) for x in DEFAULT_MARKER))
    ap.add_argument('--num_blocks', type=int, default=DEFAULT_NUM_BLOCKS)
    ap.add_argument('--out_csv', type=str, default='runs_embed_bcjr_viterbi/eval_conv_embed_bcjr_softviterbi.csv')
    args = ap.parse_args()

    marker = parse_marker(args.marker)
    nbm_prior = EmbedNBMPrior(args.nbm_ckpt, device=args.device)

    rows = []
    for i, p_sub in enumerate(parse_p_sub_list(args.p_sub_list)):
        ber, bler = evaluate_one(
            nbm_prior=nbm_prior,
            p_ins=args.p_ins,
            p_del=args.p_del,
            p_sub=p_sub,
            n_trials=args.n_trials,
            use_marker=bool(args.use_marker),
            marker=marker,
            num_blocks=args.num_blocks,
            prior_scale=args.prior_scale,
            seed=args.seed + 1000 * i,
        )
        rows.append({
            'p_ins': args.p_ins,
            'p_del': args.p_del,
            'p_sub': p_sub,
            'prior_scale': args.prior_scale,
            'BER': ber,
            'BLER': bler,
        })
        print(f'p_ins={args.p_ins:.4f} p_del={args.p_del:.4f} p_sub={p_sub:.4f} prior_scale={args.prior_scale:.3f} BER={ber:.6f} BLER={bler:.6f}')

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['p_ins', 'p_del', 'p_sub', 'prior_scale', 'BER', 'BLER'])
        writer.writeheader()
        writer.writerows(rows)

    print('\nCompact results:')
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
