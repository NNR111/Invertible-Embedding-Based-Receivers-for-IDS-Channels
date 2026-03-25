from __future__ import annotations
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ids_receiver.data.datasets import SyntheticIDSDataset, collate_batch
from ids_receiver.models.models_embed_direct import FullDirectEmbedModel
from ids_receiver.utils import set_seed, compute_ber_bler, auto_device
from ids_receiver.data.coding import parse_marker


def eval_one(model, n_trials, batch_size, p_ins, p_del, p_sub, device, seed,
             use_marker, marker, num_blocks):
    ds = SyntheticIDSDataset(
        n_trials,
        p_ins,
        p_del,
        p_sub,
        p_sub,
        use_marker=use_marker,
        marker=marker,
        num_blocks=num_blocks,
        seed=seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model.eval()
    total_ber = 0.0
    total_bler = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            noisy_syms = batch['noisy_syms'].to(device)
            noisy_lens = batch['noisy_lens'].to(device)
            msg_bits = batch['msg_bits'].to(device)

            msg_logits, _, _ = model.forward_decoder(noisy_syms, noisy_lens)
            ber, bler = compute_ber_bler(msg_logits, msg_bits)

            bsz = msg_bits.size(0)
            total_ber += ber * bsz
            total_bler += bler * bsz
            count += bsz

    return total_ber / max(count, 1), total_bler / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--p_ins', type=float, default=0.01)
    ap.add_argument('--p_del', type=float, default=0.01)
    ap.add_argument('--p_sub_list', type=str, default='0.01,0.02,0.03,0.04,0.05')
    ap.add_argument('--n_trials', type=int, default=3000)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--use_marker', type=int, default=1)
    ap.add_argument('--marker', type=str, default='0,3')
    ap.add_argument('--num_blocks', type=int, default=21)
    ap.add_argument('--out_csv', type=str, default='runs/eval_direct.csv')
    args = ap.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)
    marker = parse_marker(args.marker)

    model = FullDirectEmbedModel().to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.encoder.load_state_dict(ckpt['encoder_state'], strict=True)
    model.decoder.load_state_dict(ckpt['decoder_state'], strict=True)

    rows = []
    for p_str in args.p_sub_list.split(','):
        p_sub = float(p_str)
        ber, bler = eval_one(
            model,
            args.n_trials,
            args.batch_size,
            args.p_ins,
            args.p_del,
            p_sub,
            device,
            args.seed + int(p_sub * 10000),
            bool(args.use_marker),
            marker,
            args.num_blocks
        )
        rows.append({'p_sub': p_sub, 'BER': ber, 'BLER': bler})
        print(f'p_sub={p_sub:.4f} BER={ber:.6f} BLER={bler:.6f}')

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print('\nCompact results:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
