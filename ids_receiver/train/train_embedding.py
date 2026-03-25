from __future__ import annotations
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ids_receiver.data.datasets import SyntheticIDSDataset, collate_batch
from ids_receiver.models.models_embed import FullEmbedModel
from ids_receiver.losses import info_nce_loss
from ids_receiver.utils import set_seed, ensure_dir, auto_device, save_json
from ids_receiver.data.coding import parse_marker
from ids_receiver.arg_utils import add_common_channel_args, add_train_size_args, add_light_flag, apply_light_overrides


def run_epoch(model, loader, optim, device, train: bool, lambda_local: float):
    model.train(train)
    total = 0.0
    count = 0
    total_local = 0.0
    total_global = 0.0
    for batch in tqdm(loader, leave=False):
        clean_syms = batch['clean_syms'].to(device)
        clean_lens = batch['clean_lens'].to(device)
        noisy_syms = batch['noisy_syms'].to(device)
        noisy_lens = batch['noisy_lens'].to(device)
        coded_bits = batch['coded_bits'].to(device)

        z_clean, clean_feats = model.encode(clean_syms, clean_lens)
        z_noisy, noisy_feats = model.encode(noisy_syms, noisy_lens)
        clean_local_logits, _ = model.local_head(clean_feats, clean_lens)
        noisy_local_logits, _ = model.local_head(noisy_feats, noisy_lens)

        loss_global = info_nce_loss(z_clean, z_noisy)
        loss_local = 0.5 * (
            F.binary_cross_entropy_with_logits(clean_local_logits, coded_bits) +
            F.binary_cross_entropy_with_logits(noisy_local_logits, coded_bits)
        )
        loss = loss_global + lambda_local * loss_local
        if train:
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.local_head.parameters()), 1.0)
            optim.step()
        bsz = clean_syms.size(0)
        total += loss.item() * bsz
        total_local += loss_local.item() * bsz
        total_global += loss_global.item() * bsz
        count += bsz
    return {'loss': total / max(count, 1), 'local': total_local / max(count, 1), 'global': total_global / max(count, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--batch_size', type=int, default=160)
    ap.add_argument('--lr', type=float, default=8e-4)
    ap.add_argument('--save_dir', type=str, default='runs_embed/embed')
    ap.add_argument('--lambda_local', type=float, default=0.35)
    add_train_size_args(ap)
    add_common_channel_args(ap)
    add_light_flag(ap)
    args = apply_light_overrides(ap.parse_args(), epochs_light=2)

    set_seed(args.seed)
    ensure_dir(args.save_dir)
    save_json(vars(args), os.path.join(args.save_dir, 'args.json'))
    marker = parse_marker(args.marker)
    device = auto_device(args.device)

    train_ds = SyntheticIDSDataset(args.train_samples, args.p_ins, args.p_del, args.p_sub_train_min, args.p_sub_train_max,
                                   use_marker=bool(args.use_marker), marker=marker, num_blocks=args.num_blocks, seed=args.seed)
    val_ds = SyntheticIDSDataset(args.val_samples, args.p_ins, args.p_del, args.p_sub_train_min, args.p_sub_train_max,
                                 use_marker=bool(args.use_marker), marker=marker, num_blocks=args.num_blocks, seed=args.seed + 99991)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=args.workers)

    model = FullEmbedModel().to(device)
    optim = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.local_head.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)

    best = 1e9
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optim, device, True, args.lambda_local)
        va = run_epoch(model, val_loader, optim, device, False, args.lambda_local)
        scheduler.step(va['loss'])
        print(f"[embed] epoch {epoch:03d} train_loss={tr['loss']:.6f} train_global={tr['global']:.6f} train_local={tr['local']:.6f} "
              f"val_loss={va['loss']:.6f} val_global={va['global']:.6f} val_local={va['local']:.6f}")
        ckpt = {
            'encoder_state': model.encoder.state_dict(),
            'local_head_state': model.local_head.state_dict(),
            'args': vars(args),
            'epoch': epoch,
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pt'))
        if va['loss'] < best:
            best = va['loss']
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))


if __name__ == '__main__':
    main()
