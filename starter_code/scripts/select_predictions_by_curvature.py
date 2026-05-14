#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_keys(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def curvature_mean(pc, k=24, max_points=512):
    pc = pc.astype(np.float64, copy=False)
    tree = cKDTree(pc)
    _, idx = tree.query(pc, k=k, workers=-1)
    step = max(1, len(idx) // max_points)
    vals = []
    for ids in idx[::step]:
        nb = pc[ids]
        center = nb.mean(axis=0)
        cov = (nb - center).T @ (nb - center) / max(len(ids), 1)
        eig = np.linalg.eigvalsh(cov)
        eig = np.maximum(eig, 1e-12)
        vals.append(float(eig[0] / eig.sum()))
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--curvature_threshold", type=float, default=0.07035767702189535)
    parser.add_argument("--curvature_k", type=int, default=24)
    parser.add_argument("--invert", action="store_true", help="Use candidate when curvature is above threshold.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    selected = 0
    keys = read_keys(args.list)
    for n, key in enumerate(keys, start=1):
        base_path = base_dir / key / "denoised.npy"
        candidate_path = candidate_dir / key / "denoised.npy"
        base = np.load(base_path)
        curv = curvature_mean(base, k=args.curvature_k)
        use_candidate = curv > args.curvature_threshold if args.invert else curv < args.curvature_threshold
        out = np.load(candidate_path) if use_candidate else base
        selected += int(use_candidate)
        out_path = output_dir / key / "denoised.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out.astype(np.float32))
        if n % 25 == 0 or n == len(keys):
            print(f"processed {n}/{len(keys)} selected={selected}")
    print(f"selected candidate for {selected}/{len(keys)} samples")


if __name__ == "__main__":
    main()
