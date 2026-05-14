#!/usr/bin/env python
import argparse
import os
import shutil
import zipfile

import numpy as np


def build_tree(points):
    try:
        from scipy.spatial import cKDTree

        return cKDTree(points)
    except Exception:
        from sklearn.neighbors import KDTree

        class _Tree:
            def __init__(self, pts):
                self.tree = KDTree(pts)

            def query(self, pts, k):
                return self.tree.query(pts, k=k, return_distance=True)

        return _Tree(points)


def query_knn_radius(points, k):
    tree = build_tree(points)
    dists, _ = tree.query(points, k=k + 1)
    dists = np.asarray(dists, dtype=np.float32)
    # Skip self-neighbor at column 0 and use a stable local scale rather than a
    # single farthest-neighbor distance.
    return np.median(dists[:, 1:], axis=1).astype(np.float32)


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


def clip_prediction(noisy, pred, radius, scale, percentile_cap, min_alpha, max_alpha):
    disp = pred - noisy
    norm = np.linalg.norm(disp, axis=1).astype(np.float32)
    cap = radius * scale
    if percentile_cap > 0:
        cap = np.minimum(cap, np.percentile(norm, percentile_cap).astype(np.float32))
    alpha = cap / (norm + 1e-12)
    alpha = np.clip(alpha, min_alpha, max_alpha).astype(np.float32)
    out = noisy + disp * alpha[:, None]
    clipped = float(np.mean(alpha < 0.999))
    return out.astype(np.float32), float(np.mean(alpha)), clipped, float(np.mean(np.linalg.norm(out - noisy, axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--scales", nargs="+", type=float, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--percentile-cap", type=float, default=0.0)
    parser.add_argument("--min-alpha", type=float, default=0.0)
    parser.add_argument("--max-alpha", type=float, default=1.0)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    keys = list(iter_keys(args.list))
    for scale in args.scales:
        tag = f"k{args.k}_s{int(round(scale * 1000)):04d}"
        if args.percentile_cap > 0:
            tag += f"_p{int(round(args.percentile_cap * 10)):03d}"
        if args.min_alpha > 0:
            tag += f"_mina{int(round(args.min_alpha * 1000)):03d}"
        out_dir = f"{args.out_prefix}_{tag}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        stats = []
        for key in keys:
            noisy_path = os.path.join(args.noisy_dir, key, "noisy.npy")
            pred_path = os.path.join(args.base_dir, key, "denoised.npy")
            noisy = np.load(noisy_path).astype(np.float32)
            pred = np.load(pred_path).astype(np.float32)
            radius = query_knn_radius(noisy, args.k)
            out, mean_alpha, clipped, mean_disp = clip_prediction(
                noisy,
                pred,
                radius,
                scale=scale,
                percentile_cap=args.percentile_cap,
                min_alpha=args.min_alpha,
                max_alpha=args.max_alpha,
            )
            out_path = os.path.join(out_dir, key, "denoised.npy")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, out)
            stats.append((mean_alpha, clipped, mean_disp))
        stats_arr = np.asarray(stats, dtype=np.float32)
        print(
            f"made {out_dir} count={len(keys)} "
            f"mean_alpha={stats_arr[:,0].mean():.4f} "
            f"clipped={stats_arr[:,1].mean():.4f} "
            f"mean_disp={stats_arr[:,2].mean():.6f}",
            flush=True,
        )
        if args.zip:
            zip_path = f"{out_dir}.zip"
            zip_dir(out_dir, zip_path)
            print(f"zipped {zip_path}", flush=True)


if __name__ == "__main__":
    main()
