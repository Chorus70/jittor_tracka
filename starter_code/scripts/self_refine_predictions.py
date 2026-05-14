#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from postprocess_predictions import read_keys


def pca_normals_and_stats(pc, k):
    tree = cKDTree(pc)
    dists, idx = tree.query(pc, k=k, workers=-1)
    spacing = np.maximum(dists[:, 1:].mean(axis=1), 1e-8)
    nb = pc[idx]
    centers = nb.mean(axis=1)
    centered = nb - centers[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    val, vec = np.linalg.eigh(cov)
    val = np.maximum(val, 1e-12)
    normals = vec[:, :, 0]
    planarity = np.clip((val[:, 1] - val[:, 0]) / val[:, 2], 0.0, 1.0)
    return normals, planarity, spacing, dists, idx, centers


def one_step(pc, anchor, noisy, k, plane_beta, anchor_beta, noisy_beta, repulsion_beta, max_step_scale):
    normals, planarity, spacing, dists, idx, centers = pca_normals_and_stats(pc, k)
    signed = np.einsum("ni,ni->n", pc - centers, normals)
    clamp = np.maximum(spacing * max_step_scale, 1e-8)
    signed = np.clip(signed, -clamp, clamp)
    update = -plane_beta * planarity[:, None] * signed[:, None] * normals

    if repulsion_beta > 0:
        close = idx[:, 1: min(k, 7)]
        vec = pc[:, None, :] - pc[close]
        dist = np.linalg.norm(vec, axis=2) + 1e-8
        threshold = 0.65 * spacing[:, None]
        strength = np.maximum(0.0, threshold - dist) / threshold
        rep = (vec / dist[:, :, None]) * strength[:, :, None]
        update += repulsion_beta * rep.mean(axis=1) * spacing[:, None]

    update += anchor_beta * (anchor - pc)
    if noisy_beta > 0:
        update += noisy_beta * (noisy - pc)

    step_norm = np.linalg.norm(update, axis=1)
    max_step = max_step_scale * spacing
    scale = np.minimum(1.0, max_step / (step_norm + 1e-8))
    return pc + update * scale[:, None]


def refine(noisy, pred, alpha, iters, k, plane_beta, anchor_beta, noisy_beta, repulsion_beta, max_step_scale):
    anchor = noisy + alpha * (pred - noisy)
    pc = anchor.copy()
    for _ in range(iters):
        pc = one_step(
            pc=pc,
            anchor=anchor,
            noisy=noisy,
            k=k,
            plane_beta=plane_beta,
            anchor_beta=anchor_beta,
            noisy_beta=noisy_beta,
            repulsion_beta=repulsion_beta,
            max_step_scale=max_step_scale,
        )
    return pc.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list", default=None)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--plane_beta", type=float, default=0.35)
    parser.add_argument("--anchor_beta", type=float, default=0.20)
    parser.add_argument("--noisy_beta", type=float, default=0.0)
    parser.add_argument("--repulsion_beta", type=float, default=0.03)
    parser.add_argument("--max_step_scale", type=float, default=0.35)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    keys = read_keys(args.list)
    if keys is None:
        keys = [str(p.parent.relative_to(input_dir)) for p in sorted(input_dir.rglob("noisy.npy"))]

    for n, key in enumerate(keys, start=1):
        noisy = np.load(input_dir / key / "noisy.npy").astype(np.float64)
        pred = np.load(pred_dir / key / "denoised.npy").astype(np.float64)
        out = refine(
            noisy=noisy,
            pred=pred,
            alpha=args.alpha,
            iters=args.iters,
            k=args.k,
            plane_beta=args.plane_beta,
            anchor_beta=args.anchor_beta,
            noisy_beta=args.noisy_beta,
            repulsion_beta=args.repulsion_beta,
            max_step_scale=args.max_step_scale,
        )
        out_path = output_dir / key / "denoised.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out)
        if n % 10 == 0 or n == len(keys):
            print(f"processed {n}/{len(keys)}")


if __name__ == "__main__":
    main()
