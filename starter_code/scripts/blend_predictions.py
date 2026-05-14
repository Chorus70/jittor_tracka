#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


def read_keys(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_dir", required=True)
    parser.add_argument("--b_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--lambda_b", type=float, default=0.2)
    args = parser.parse_args()

    a_dir = Path(args.a_dir)
    b_dir = Path(args.b_dir)
    output_dir = Path(args.output_dir)
    lam = float(args.lambda_b)
    for n, key in enumerate(read_keys(args.list), start=1):
        a = np.load(a_dir / key / "denoised.npy").astype(np.float64)
        b = np.load(b_dir / key / "denoised.npy").astype(np.float64)
        out = (1.0 - lam) * a + lam * b
        out_path = output_dir / key / "denoised.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out.astype(np.float32))
        if n % 25 == 0:
            print(f"blended {n}")


if __name__ == "__main__":
    main()
