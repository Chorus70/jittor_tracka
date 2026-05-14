#!/usr/bin/env python
import argparse
import os
import shutil
import zipfile

import numpy as np
from scipy.spatial import cKDTree


def iter_keys(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            key = line.strip()
            if key:
                yield key


def zip_dir(src_dir, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                path = os.path.join(root, name)
                zf.write(path, os.path.relpath(path, src_dir))


def local_ratio(noisy, pred, k):
    disp = pred - noisy
    norm = np.linalg.norm(disp, axis=1).astype(np.float32)
    tree = cKDTree(noisy)
    dists, _ = tree.query(noisy, k=k + 1)
    radius = np.median(dists[:, 1:], axis=1).astype(np.float32)
    return norm / (radius + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bins", nargs="+", type=float, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    if len(args.alphas) != len(args.bins) + 1:
        raise ValueError("--alphas length must be len(--bins) + 1")

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)

    bins = np.asarray(args.bins, dtype=np.float32)
    alphas = np.asarray(args.alphas, dtype=np.float32)
    stats = []
    for key in iter_keys(args.list):
        noisy = np.load(os.path.join(args.noisy_dir, key, "noisy.npy")).astype(np.float32)
        pred = np.load(os.path.join(args.base_dir, key, "denoised.npy")).astype(np.float32)
        ratio = local_ratio(noisy, pred, args.k)
        idx = np.searchsorted(bins, ratio, side="right")
        alpha = alphas[idx]
        out = noisy + (pred - noisy) * alpha[:, None]
        out_path = os.path.join(args.out_dir, key, "denoised.npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, out.astype(np.float32))
        stats.append((float(alpha.mean()), float(np.linalg.norm(out - noisy, axis=1).mean())))

    stats = np.asarray(stats, dtype=np.float32)
    print(
        f"made {args.out_dir} count={len(stats)} mean_alpha={stats[:,0].mean():.4f} "
        f"mean_disp={stats[:,1].mean():.6f}",
        flush=True,
    )
    if args.zip:
        zip_dir(args.out_dir, f"{args.out_dir}.zip")


if __name__ == "__main__":
    main()
