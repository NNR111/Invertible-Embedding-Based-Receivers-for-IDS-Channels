from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from ids_receiver.data.coding import (
    random_message,
    conv_encode_bits,
    build_marker_patterns,
    marcode,
)
from ids_receiver.data.channel import ids_channel
from ids_receiver.receivers.bcjr_soft_viterbi_receiver import (
    MSG_LEN,
    FB_decode,
    rfz,
    soft_input_viterbi,
)


def decode_bcjr_softviterbi_from_sequences(
    noisy_syms: np.ndarray,
    p_ins: float,
    p_del: float,
    p_sub: float,
    T: int = 142,
    Np: int = 7,
    l_max: int = 2,
) -> dict:
    noisy_syms = np.asarray(noisy_syms, dtype=np.int64).reshape(-1)

    const_length = 3
    g = [5, 7]

    delta_step = 0.01
    xx_vec = np.arange(0.0, 10.0 + 1e-12, delta_step)
    log_map_vec = np.log(1.0 + np.exp(-xx_vec))

    if p_ins == 0:
        l_max = 0

    mu = np.array(
        [
            [p_del, 1 - p_del - p_ins],
            [0.0, 0.0],
            [p_ins, 0.0],
        ],
        dtype=np.float64,
    )
    mu = mu / np.sum(mu)

    mp1 = -1 * np.ones(T, dtype=np.int64)
    mp2 = -1 * np.ones(T, dtype=np.int64)

    mp1[Np::Np] = 1
    mp1[Np + 1::Np] = 0

    mp2[Np::Np] = 1
    mp2[Np + 1::Np] = 0

    rho, f, zeta = rfz(mp1, mp2, p_sub, T)

    p_ub_1 = FB_decode(
        y=noisy_syms,
        T=T,
        mu=mu,
        rho=rho,
        f=f,
        zeta=zeta,
        mp1=mp1,
        mp2=mp2,
        log_map_vec=log_map_vec,
        delta_step=delta_step,
        l_max=l_max,
    )

    eps = 1e-300
    llr_vec = -np.log(np.maximum(p_ub_1, eps) / np.maximum(1.0 - p_ub_1, eps))

    msg_dec = soft_input_viterbi(
        llr_vec,
        generators_octal=g,
        constraint_length=const_length,
        terminated=True,
    )

    return {
        "msg_hat": np.asarray(msg_dec[:MSG_LEN], dtype=np.int64),
        "LLR_vec": llr_vec,
        "p_ub_1": p_ub_1,
        "y": noisy_syms,
    }


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


class SyntheticIDSDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        p_ins: float,
        p_del: float,
        p_sub_min: float,
        p_sub_max: float,
        use_marker: bool = False,
        marker: tuple[int, ...] = (0, 3),
        num_blocks: int = 20,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.p_ins = p_ins
        self.p_del = p_del
        self.p_sub_min = p_sub_min
        self.p_sub_max = p_sub_max
        self.seed = seed

        self.use_marker = use_marker
        self.marker = marker
        self.num_blocks = num_blocks

        self.T = 142
        self.Np = 7
        self.mp1, self.mp2 = build_marker_patterns(T=self.T, Np=self.Np)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)

        msg = random_message(100, rng)
        coded_bits = conv_encode_bits(msg, g=(0o5, 0o7), K=3, terminate=True)
        clean_syms = marcode(coded_bits, self.mp1, self.mp2)

        p_sub = float(rng.uniform(self.p_sub_min, self.p_sub_max))

        noisy_syms = ids_channel(
            clean_syms,
            self.p_ins,
            self.p_del,
            p_sub,
            vocab=4,
            rng=rng,
            l_max=0 if self.p_ins == 0 else 2,
        )

        return {
            "msg_bits": msg.astype(np.int64),
            "noisy_syms": noisy_syms.astype(np.int64),
            "p_sub": p_sub,
        }


def evaluate_one_setting(
    n_trials: int,
    p_ins: float,
    p_del: float,
    p_sub: float,
    use_marker: int,
    marker: tuple[int, ...],
    num_blocks: int,
    seed: int,
    progress_every: int = 10,
    verbose: int = 1,
):
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

    if verbose:
        print(f"Number of samples: {len(ds)}")
        print(f"Using p_ins={p_ins:.4f}, p_del={p_del:.4f}, p_sub={p_sub:.4f}")

    bit_errors = 0
    frame_errors = 0
    t0 = time.time()

    for i in tqdm(range(len(ds)), total=len(ds), leave=True, desc=f"BCJR python p_sub={p_sub:.4f}"):
        item = ds[i]

        msg = item["msg_bits"]
        noisy_syms = item["noisy_syms"]
        cur_p_sub = float(item["p_sub"])

        out = decode_bcjr_softviterbi_from_sequences(
            noisy_syms=noisy_syms,
            p_ins=p_ins,
            p_del=p_del,
            p_sub=cur_p_sub,
        )

        msg_hat = np.asarray(out["msg_hat"]).astype(np.int64)
        nbe = int(np.sum(msg_hat != msg))

        bit_errors += nbe
        frame_errors += int(nbe > 0)

        if progress_every > 0 and ((i + 1) % progress_every == 0 or (i + 1) == len(ds)):
            cur_ber = bit_errors / ((i + 1) * MSG_LEN)
            cur_fer = frame_errors / (i + 1)
            elapsed = time.time() - t0
            print(
                f"[{i+1}/{len(ds)}] "
                f"partial BER={cur_ber:.6f} FER={cur_fer:.6f} "
                f"elapsed={elapsed:.2f}s"
            )

    ber = bit_errors / (len(ds) * MSG_LEN)
    fer = frame_errors / len(ds)
    total_time = time.time() - t0

    if verbose:
        print(f"Finished evaluation in {total_time:.2f}s")
        print(f"Final BER={ber:.6f} FER={fer:.6f}")

    return ber, fer, len(ds), total_time


def append_result_csv(out_csv: str, rows: list[dict]):
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(out_csv)

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "p_ins",
                "p_del",
                "p_sub",
                "n_trials",
                "use_marker",
                "marker",
                "num_blocks",
                "BER",
                "FER",
                "elapsed_sec",
            ],
        )
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_ins", type=float, default=0.03)
    ap.add_argument("--p_del", type=float, default=0.03)
    ap.add_argument("--p_sub_list", type=str, required=True)
    ap.add_argument("--n_trials", type=int, default=3000)

    ap.add_argument("--use_marker", type=int, default=1)
    ap.add_argument("--marker", type=str, default="0,3")
    ap.add_argument("--num_blocks", type=int, default=20)

    ap.add_argument(
        "--out_csv",
        type=str,
        default="runs_bcjr_viterbi/eval_conv_bcjr_softviterbi_python.csv",
    )
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p_sub_values = parse_p_sub_list(args.p_sub_list)
    marker = tuple(int(x.strip()) for x in args.marker.split(",") if x.strip() != "")

    rows = []

    for p_sub in p_sub_values:
        ber, fer, n_trials, total_time = evaluate_one_setting(
            n_trials=args.n_trials,
            p_ins=args.p_ins,
            p_del=args.p_del,
            p_sub=p_sub,
            use_marker=args.use_marker,
            marker=marker,
            num_blocks=args.num_blocks,
            seed=args.seed,
            progress_every=args.progress_every,
            verbose=args.verbose,
        )

        row = {
            "p_ins": args.p_ins,
            "p_del": args.p_del,
            "p_sub": p_sub,
            "n_trials": n_trials,
            "use_marker": args.use_marker,
            "marker": ",".join(str(x) for x in marker),
            "num_blocks": args.num_blocks,
            "BER": ber,
            "FER": fer,
            "elapsed_sec": total_time,
        }
        rows.append(row)

        print(
            f"p_ins={args.p_ins:.4f} "
            f"p_del={args.p_del:.4f} "
            f"p_sub={p_sub:.4f} "
            f"-> BER={ber:.6f} FER={fer:.6f}"
        )

    append_result_csv(args.out_csv, rows)

    print("\nAppended results:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
