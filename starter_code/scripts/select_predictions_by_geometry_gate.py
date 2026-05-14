#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_keys(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def geometry_stats(pc, k):
    tree = cKDTree(pc)
    _, idx = tree.query(pc, k=min(k, len(pc)), workers=-1)
    nb = pc[idx]
    center = nb.mean(axis=1)
    centered = nb - center[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(idx.shape[1], 1)
    val, _ = np.linalg.eigh(cov)
    val = np.maximum(val, 1e-12)
    curvature = val[:, 0] / val.sum(axis=1)
    planarity = (val[:, 1] - val[:, 0]) / val[:, 2]
    return {
        "curv_mean": float(curvature.mean()),
        "curv_p90": float(np.percentile(curvature, 90)),
        "plan_mean": float(planarity.mean()),
        "plan_p10": float(np.percentile(planarity, 10)),
    }


def compare(value, op, threshold):
    if op == "le":
        return value <= threshold
    if op == "ge":
        return value >= threshold
    raise ValueError(f"unsupported op: {op}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--feature", choices=["curv_mean", "curv_p90", "plan_mean", "plan_p10"], required=True)
    parser.add_argument("--op", choices=["le", "ge"], required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    cand_dir = Path(args.candidate_dir)
    out_dir = Path(args.output_dir)
    selected = 0
    total = 0
    for key in read_keys(args.list):
        base = np.load(base_dir / key / "denoised.npy")
        cand = np.load(cand_dir / key / "denoised.npy")
        stats = geometry_stats(base.astype(np.float64), args.k)
        use_candidate = compare(stats[args.feature], args.op, args.threshold)
        out = cand if use_candidate else base
        if use_candidate:
            selected += 1
        total += 1
        out_path = out_dir / key / "denoised.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out.astype(np.float32))
    print(f"selected candidate for {selected}/{total} samples")


if __name__ == "__main__":
    main()
