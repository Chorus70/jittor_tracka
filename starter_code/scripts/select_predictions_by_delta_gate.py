#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


def read_keys(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def delta_stats(base, candidate):
    norm = np.linalg.norm(candidate.astype(np.float64) - base.astype(np.float64), axis=1)
    return {
        "mean": float(np.mean(norm)),
        "median": float(np.median(norm)),
        "p95": float(np.percentile(norm, 95)),
        "max": float(np.max(norm)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--max_mean", type=float, default=None)
    parser.add_argument("--max_median", type=float, default=None)
    parser.add_argument("--max_p95", type=float, default=None)
    parser.add_argument("--max_max", type=float, default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    selected = 0
    total = 0
    for key in read_keys(args.list):
        base_path = base_dir / key / "denoised.npy"
        cand_path = candidate_dir / key / "denoised.npy"
        if not base_path.exists():
            raise FileNotFoundError(base_path)
        base = np.load(base_path)
        use_candidate = False
        if cand_path.exists():
            cand = np.load(cand_path)
            stats = delta_stats(base, cand)
            use_candidate = True
            if args.max_mean is not None and stats["mean"] > args.max_mean:
                use_candidate = False
            if args.max_median is not None and stats["median"] > args.max_median:
                use_candidate = False
            if args.max_p95 is not None and stats["p95"] > args.max_p95:
                use_candidate = False
            if args.max_max is not None and stats["max"] > args.max_max:
                use_candidate = False
        out = cand if use_candidate else base
        if use_candidate:
            selected += 1
        total += 1
        out_path = output_dir / key / "denoised.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out.astype(np.float32))
    print(f"selected candidate for {selected}/{total} samples")


if __name__ == "__main__":
    main()
