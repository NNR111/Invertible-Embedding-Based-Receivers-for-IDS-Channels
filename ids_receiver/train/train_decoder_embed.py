from __future__ import annotations
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ids_receiver.data.datasets import SyntheticIDSDataset, collate_batch
from ids_receiver.models.models_embed import FullEmbedModel
from ids_receiver.utils import set_seed, ensure_dir, auto_device, compute_ber_bler, save_json
from ids_receiver.data.coding import parse_marker
from ids_receiver.arg_utils import add_common_channel_args, add_train_size_args, add_light_flag, apply_light_overrides


def run_epoch(model, loader, optim, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_ber = 0.0
    total_bler = 0.0
    count = 0
    for batch in tqdm(loader, leave=False):
        noisy_syms = batch['noisy_syms'].to(device)
        noisy_lens = batch['noisy_lens'].to(device)
        msg_bits = batch['msg_bits'].to(device)
        _, msg_logits, _ = model.forward_decoder(noisy_syms, noisy_lens)
        loss = F.binary_cross_entropy_with_logits(msg_logits, msg_bits)
        if train:
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
        ber, bler = compute_ber_bler(msg_logits.detach(), msg_bits)
        bsz = msg_bits.size(0)
        total_loss += loss.item() * bsz
        total_ber += ber * bsz
        total_bler += bler * bsz
        count += bsz
    return total_loss / max(count, 1), total_ber / max(count, 1), total_bler / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embed_ckpt', type=str, required=True)
    ap.add_argument('--nbm_ckpt', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=22)
    ap.add_argument('--batch_size', type=int, default=160)
    ap.add_argument('--lr', type=float, default=7e-4)
    ap.add_argument('--save_dir', type=str, default='runs_embed/decoder')
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
                                 use_marker=bool(args.use_marker), marker=marker, num_blocks=args.num_blocks, seed=args.seed + 77771)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=args.workers)

    model = FullEmbedModel().to(device)
    eckpt = torch.load(args.embed_ckpt, map_location='cpu')
    nckpt = torch.load(args.nbm_ckpt, map_location='cpu')
    model.encoder.load_state_dict(eckpt['encoder_state'])
    model.nbm.load_state_dict(nckpt['nbm_state'])
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.nbm.parameters():
        p.requires_grad = False
    for p in model.local_head.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)

    best = 1e9
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ber, tr_bler = run_epoch(model, train_loader, optim, device, True)
        va_loss, va_ber, va_bler = run_epoch(model, val_loader, optim, device, False)
        scheduler.step(va_loss)
        print(f'[decoder_embed] epoch {epoch:03d} train_loss={tr_loss:.6f} train_BER={tr_ber:.6f} train_BLER={tr_bler:.6f} val_loss={va_loss:.6f} val_BER={va_ber:.6f} val_BLER={va_bler:.6f}')
        ckpt = {
            'encoder_state': model.encoder.state_dict(),
            'nbm_state': model.nbm.state_dict(),
            'decoder_state': model.decoder.state_dict(),
            'args': vars(args),
            'epoch': epoch,
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pt'))
        if va_loss < best:
            best = va_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))


if __name__ == '__main__':
    main()
