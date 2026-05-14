#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_keys(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def local_radius(pc, k):
    tree = cKDTree(pc)
    dists, _ = tree.query(pc, k=k, workers=-1)
    return np.maximum(dists[:, -1], 1e-8)


def clipped_delta(base, candidate, k, clip_scale):
    delta = candidate - base
    radius = local_radius(base, k=k)
    limit = radius * clip_scale
    norm = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, limit / np.maximum(norm, 1e-12))
    return delta * scale[:, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--clip_k", type=int, default=24)
    parser.add_argument("--clip_scale", type=float, default=0.5)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    for n, key in enumerate(read_keys(args.list), start=1):
        base = np.load(base_dir / key / "denoised.npy").astype(np.float64)
        candidate = np.load(candidate_dir / key / "denoised.npy").astype(np.float64)
        delta = clipped_delta(base, candidate, k=args.clip_k, clip_scale=args.clip_scale)
        out = base + args.beta * delta
        out_path = output_dir / key / "denoised.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out.astype(np.float32))
        if n % 25 == 0:
            print(f"blended {n}")


if __name__ == "__main__":
    main()
