
from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ids_receiver.arg_utils import add_light_flag, apply_light_overrides
from ids_receiver.data.datasets import IDSDataset, collate_batch
from ids_receiver.models.models_noembed import FullNoEmbedModel
from ids_receiver.utils import auto_device, compute_ber_bler, compute_code_ber, ensure_dir, save_json, set_seed


def loss_and_metrics(model, batch, device, lambda_code: float, lambda_msg: float):
    noisy_syms = batch["noisy_syms"].to(device)
    noisy_lens = batch["noisy_lens"].to(device)
    coded_bits = batch["coded_bits"].to(device)
    msg_bits = batch["msg_bits"].to(device)

    code_logits, msg_logits, _, _ = model.forward_decoder(noisy_syms, noisy_lens)
    loss_code = F.binary_cross_entropy_with_logits(code_logits, coded_bits)
    loss_msg = F.binary_cross_entropy_with_logits(msg_logits, msg_bits)
    loss = lambda_code * loss_code + lambda_msg * loss_msg

    cber = compute_code_ber(code_logits.detach(), coded_bits)
    ber, bler = compute_ber_bler(msg_logits.detach(), msg_bits)
    return loss, loss_code.detach(), loss_msg.detach(), cber, ber, bler


def run_epoch(model, loader, optim, device, train: bool, lambda_code: float, lambda_msg: float):
    model.train(train)
    stats = {
        "loss": 0.0,
        "loss_code": 0.0,
        "loss_msg": 0.0,
        "cber": 0.0,
        "ber": 0.0,
        "bler": 0.0,
        "count": 0,
    }

    for batch in tqdm(loader, leave=False):
        loss, loss_code, loss_msg, cber, ber, bler = loss_and_metrics(
            model, batch, device, lambda_code, lambda_msg
        )
        bsz = batch["msg_bits"].size(0)

        if train:
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        stats["loss"] += loss.item() * bsz
        stats["loss_code"] += float(loss_code.item()) * bsz
        stats["loss_msg"] += float(loss_msg.item()) * bsz
        stats["cber"] += cber * bsz
        stats["ber"] += ber * bsz
        stats["bler"] += bler * bsz
        stats["count"] += bsz

    count = max(1, stats["count"])
    return {k: (v / count if k != "count" else v) for k, v in stats.items()}


def make_loaders(train_path: str, val_path: str, batch_size: int, workers: int):
    train_ds = IDSDataset(train_path)
    val_ds = IDSDataset(val_path)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=workers,
    )
    return train_loader, val_loader


def save_ckpt(path: str, model, optim, scheduler, epoch: int, best: float, args: Dict):
    torch.save(
        {
            "front_state": model.front.state_dict(),
            "backbone_state": model.backbone.state_dict(),
            "nbm_state": model.nbm.state_dict(),
            "decoder_state": model.decoder.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_msg_loss": best,
            "args": args,
            "epoch": epoch,
        },
        path,
    )


def load_pretrained(model, ckpt_path: str | None):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "front_state" in ckpt:
        model.front.load_state_dict(ckpt["front_state"], strict=False)
    if "backbone_state" in ckpt:
        model.backbone.load_state_dict(ckpt["backbone_state"], strict=False)
    if "nbm_state" in ckpt:
        model.nbm.load_state_dict(ckpt["nbm_state"], strict=False)
    if "decoder_state" in ckpt:
        model.decoder.load_state_dict(ckpt["decoder_state"], strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, required=True)
    ap.add_argument("--val_path", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="runs_noembed/joint")
    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--lr_front", type=float, default=2e-4)
    ap.add_argument("--lr_nbm", type=float, default=3e-4)
    ap.add_argument("--lr_decoder", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_code", type=float, default=0.35)
    ap.add_argument("--lambda_msg", type=float, default=1.0)
    add_light_flag(ap)
    args = apply_light_overrides(ap.parse_args(), epochs_light=2)

    set_seed(args.seed)
    ensure_dir(args.save_dir)
    save_json(vars(args), os.path.join(args.save_dir, "args.json"))
    device = auto_device(args.device)

    train_loader, val_loader = make_loaders(args.train_path, args.val_path, args.batch_size, args.workers)
    model = FullNoEmbedModel().to(device)

    start_epoch = 1
    best = 1e9
    ckpt = None

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        load_pretrained(model, args.resume_ckpt)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best_val_msg_loss", 1e9))
    else:
        load_pretrained(model, args.init_ckpt)

    optim = torch.optim.AdamW(
        [
            {"params": model.front.parameters(), "lr": args.lr_front},
            {"params": model.backbone.parameters(), "lr": args.lr_front},
            {"params": model.nbm.parameters(), "lr": args.lr_nbm},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=3
    )

    if ckpt is not None:
        if "optimizer_state" in ckpt:
            optim.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

    for epoch in range(start_epoch, args.epochs + 1):
        tr = run_epoch(model, train_loader, optim, device, True, args.lambda_code, args.lambda_msg)
        va = run_epoch(model, val_loader, None, device, False, args.lambda_code, args.lambda_msg)
        scheduler.step(va["loss_msg"])

        print(
            f"[joint_noembed] epoch {epoch:03d} "
            f"train_loss={tr['loss']:.6f} train_code_loss={tr['loss_code']:.6f} train_msg_loss={tr['loss_msg']:.6f} "
            f"train_codeBER={tr['cber']:.6f} train_BER={tr['ber']:.6f} train_BLER={tr['bler']:.6f} "
            f"val_loss={va['loss']:.6f} val_code_loss={va['loss_code']:.6f} val_msg_loss={va['loss_msg']:.6f} "
            f"val_codeBER={va['cber']:.6f} val_BER={va['ber']:.6f} val_BLER={va['bler']:.6f}"
        )

        save_ckpt(os.path.join(args.save_dir, "last.pt"), model, optim, scheduler, epoch, best, vars(args))
        if va["loss_msg"] < best:
            best = va["loss_msg"]
            save_ckpt(os.path.join(args.save_dir, "best.pt"), model, optim, scheduler, epoch, best, vars(args))


if __name__ == "__main__":
    main()
