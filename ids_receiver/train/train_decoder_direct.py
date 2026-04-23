from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ids_receiver.data.datasets import IDSDataset, collate_batch
from ids_receiver.models.models_embed_direct import FullDirectEmbedModel
from ids_receiver.utils import set_seed, ensure_dir, auto_device, compute_ber_bler, save_json


def run_epoch(model, loader, optim, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_ber = 0.0
    total_bler = 0.0
    count = 0

    for batch in tqdm(loader, leave=False):
        noisy_syms = batch["noisy_syms"].to(device)
        noisy_lens = batch["noisy_lens"].to(device)
        msg_bits = batch["msg_bits"].to(device)

        msg_logits, _, _ = model.forward_decoder(noisy_syms, noisy_lens)
        loss = F.binary_cross_entropy_with_logits(msg_logits, msg_bits)

        if train:
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optim.step()

        ber, bler = compute_ber_bler(msg_logits.detach(), msg_bits)
        bsz = msg_bits.size(0)

        total_loss += loss.item() * bsz
        total_ber += ber * bsz
        total_bler += bler * bsz
        count += bsz

    return (
        total_loss / max(count, 1),
        total_ber / max(count, 1),
        total_bler / max(count, 1),
    )


def load_init_ckpt(model, init_ckpt: str):
    ckpt = torch.load(init_ckpt, map_location="cpu")

    if "encoder_state" in ckpt:
        model.encoder.load_state_dict(ckpt["encoder_state"], strict=False)
    if "decoder_state" in ckpt:
        model.decoder.load_state_dict(ckpt["decoder_state"], strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed_ckpt", type=str, required=True)
    ap.add_argument("--freeze_embed", type=int, default=1)
    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch_size", type=int, default=160)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--save_dir", type=str, default="runs/direct_decoder")
    ap.add_argument("--train_path", type=str, default="ids_receiver/data/embed_stage1_train.mat")
    ap.add_argument("--val_path", type=str, default="ids_receiver/data/embed_stage1_val.mat")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)
    save_json(vars(args), os.path.join(args.save_dir, "args.json"))

    device = auto_device(args.device)

    train_ds = IDSDataset(args.train_path)
    val_ds = IDSDataset(args.val_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.workers,
    )

    model = FullDirectEmbedModel().to(device)

    eckpt = torch.load(args.embed_ckpt, map_location="cpu")
    model.encoder.load_state_dict(eckpt["encoder_state"], strict=False)

    if "local_head_state" in eckpt:
        try:
            model.local_head.load_state_dict(eckpt["local_head_state"], strict=False)
        except Exception:
            pass

    if args.freeze_embed == 1:
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.local_head.parameters():
            p.requires_grad = False

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2
    )

    start_epoch = 1
    best = 1e9

    if args.resume_ckpt is not None:
        rckpt = torch.load(args.resume_ckpt, map_location="cpu")

        if "encoder_state" in rckpt:
            model.encoder.load_state_dict(rckpt["encoder_state"], strict=False)
        if "decoder_state" in rckpt:
            model.decoder.load_state_dict(rckpt["decoder_state"], strict=False)
        if "optimizer_state" in rckpt:
            optim.load_state_dict(rckpt["optimizer_state"])
        if "scheduler_state" in rckpt:
            scheduler.load_state_dict(rckpt["scheduler_state"])

        start_epoch = int(rckpt.get("epoch", 0)) + 1
        best = float(rckpt.get("best_val_loss", 1e9))

    elif args.init_ckpt is not None:
        load_init_ckpt(model, args.init_ckpt)

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_ber, tr_bler = run_epoch(model, train_loader, optim, device, True)
        va_loss, va_ber, va_bler = run_epoch(model, val_loader, optim, device, False)

        scheduler.step(va_loss)

        print(
            f"[decoder_direct] epoch {epoch:03d} "
            f"train_loss={tr_loss:.6f} train_BER={tr_ber:.6f} train_BLER={tr_bler:.6f} "
            f"val_loss={va_loss:.6f} val_BER={va_ber:.6f} val_BLER={va_bler:.6f}"
        )

        ckpt = {
            "encoder_state": model.encoder.state_dict(),
            "decoder_state": model.decoder.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best,
            "args": vars(args),
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if va_loss < best:
            best = va_loss
            ckpt["best_val_loss"] = best
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()
