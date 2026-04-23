from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ids_receiver.data.datasets import SyntheticIDSDataset, collate_batch
from ids_receiver.models.models_embed import FullEmbedModel
from ids_receiver.utils import set_seed, compute_ber_bler, auto_device


# Parse comma-separated substitution probabilities
def parse_p_sub_list(s: str) -> list[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if len(vals) == 0:
        raise ValueError("p_sub_list must contain at least one value")
    return vals


# Evaluate FullEmbedModel on Python-generated on-the-fly data
def evaluate_one_setting(
    model,
    n_trials: int,
    batch_size: int,
    device,
    p_ins: float,
    p_del: float,
    p_sub: float,
    use_marker: int,
    marker: tuple[int, ...],
    num_blocks: int,
    seed: int,
):
    # SyntheticIDSDataset follows the MATLAB-matching generation path:
    # msg -> terminated CC -> marcode -> IDS channel
    ds = SyntheticIDSDataset(
        n_samples=n_trials,
        p_ins=p_ins,
        p_del=p_del,
        p_sub_min=p_sub,
        p_sub_max=p_sub,
        use_marker=bool(use_marker),
        marker=marker,
        num_blocks=num_blocks,
        seed=seed,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model.eval()
    total_ber = 0.0
    total_bler = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc=f"p_sub={p_sub:.4f}"):
            # The model only receives noisy symbols at inference
            noisy_syms = batch["noisy_syms"].to(device)
            noisy_lens = batch["noisy_lens"].to(device)

            # Ground-truth information bits are used only for BER/BLER
            msg_bits = batch["msg_bits"].to(device)

            _, msg_logits, _ = model.forward_decoder(noisy_syms, noisy_lens)

            ber, bler = compute_ber_bler(msg_logits, msg_bits)
            bsz = msg_bits.size(0)

            total_ber += ber * bsz
            total_bler += bler * bsz
            count += bsz

    return total_ber / max(count, 1), total_bler / max(count, 1)


def append_results_to_csv(rows, out_csv):
    df_new = pd.DataFrame(rows)

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(out_csv)
    df_new.to_csv(out_csv, mode="a", header=not file_exists, index=False)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--p_ins", type=float, required=True)
    ap.add_argument("--p_del", type=float, required=True)
    ap.add_argument("--p_sub_list", type=str, required=True)

    ap.add_argument("--n_trials", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=256)

    # These args are kept for compatibility with the old repo commands
    # In the MATLAB-exact synthetic path, marker construction is done via mp1/mp2
    ap.add_argument("--use_marker", type=int, default=1)
    ap.add_argument("--marker", type=str, default="0,3")
    ap.add_argument("--num_blocks", type=int, default=20)

    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="runs_embed/eval_embed_python.csv")

    args = ap.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)

    p_sub_values = parse_p_sub_list(args.p_sub_list)
    marker = tuple(int(x.strip()) for x in args.marker.split(",") if x.strip() != "")

    model = FullEmbedModel().to(device)

    # Load encoder, NBM, and decoder weights from checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.encoder.load_state_dict(ckpt["encoder_state"])
    model.nbm.load_state_dict(ckpt["nbm_state"])
    model.decoder.load_state_dict(ckpt["decoder_state"])
    model.eval()

    rows = []

    for p_sub in p_sub_values:
        ber, bler = evaluate_one_setting(
            model=model,
            n_trials=args.n_trials,
            batch_size=args.batch_size,
            device=device,
            p_ins=args.p_ins,
            p_del=args.p_del,
            p_sub=p_sub,
            use_marker=args.use_marker,
            marker=marker,
            num_blocks=args.num_blocks,
            seed=args.seed,
        )

        row = {
            "ckpt": args.ckpt,
            "p_ins": args.p_ins,
            "p_del": args.p_del,
            "p_sub": p_sub,
            "n_trials": args.n_trials,
            "use_marker": args.use_marker,
            "marker": ",".join(str(x) for x in marker),
            "num_blocks": args.num_blocks,
            "BER": ber,
            "BLER": bler,
        }
        rows.append(row)

        print(
            f"p_ins={args.p_ins:.4f} "
            f"p_del={args.p_del:.4f} "
            f"p_sub={p_sub:.4f} "
            f"-> BER={ber:.6f} BLER={bler:.6f}"
        )

    append_results_to_csv(rows, args.out_csv)

    print(f"\nResults appended to: {args.out_csv}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
