from __future__ import annotations

import argparse

from ids_receiver.config import (
    DEFAULT_MARKER,
    DEFAULT_NUM_BLOCKS,
    TRAIN_SAMPLES_LIGHT,
    TRAIN_SAMPLES_STRONG,
    VAL_SAMPLES_LIGHT,
    VAL_SAMPLES_STRONG,
)


def add_train_size_args(ap: argparse.ArgumentParser):
    ap.add_argument('--train_samples', type=int, default=TRAIN_SAMPLES_STRONG)
    ap.add_argument('--val_samples', type=int, default=VAL_SAMPLES_STRONG)
    return ap


def add_common_channel_args(ap: argparse.ArgumentParser):
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--p_ins', type=float, default=0.03)
    ap.add_argument('--p_del', type=float, default=0.03)
    ap.add_argument('--p_sub_train_min', type=float, default=0.01)
    ap.add_argument('--p_sub_train_max', type=float, default=0.05)
    ap.add_argument('--use_marker', type=int, default=0)
    ap.add_argument('--marker', type=str, default='0,3')
    ap.add_argument('--num_blocks', type=int, default=5)
    return ap


def add_light_flag(ap: argparse.ArgumentParser):
    ap.add_argument('--light', action='store_true', help='Use a small debug configuration for quick checks.')
    return ap


def apply_light_overrides(args, epochs_light: int = 2):
    if getattr(args, 'light', False):
        args.epochs = epochs_light
        args.train_samples = TRAIN_SAMPLES_LIGHT
        args.val_samples = VAL_SAMPLES_LIGHT
        args.workers = 0
    return args
