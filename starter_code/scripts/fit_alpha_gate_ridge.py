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


def point_features(noisy, pred, k):
    disp = pred - noisy
    norm = np.linalg.norm(disp, axis=1).astype(np.float32)
    tree = cKDTree(noisy)
    dists, _ = tree.query(noisy, k=k + 1)
    radius = np.median(dists[:, 1:], axis=1).astype(np.float32)
    ratio = norm / (radius + 1e-12)
    centered = noisy - noisy.mean(axis=0, keepdims=True)
    radial = np.linalg.norm(centered, axis=1).astype(np.float32)
    return np.stack([ratio, norm, radius, radial], axis=1).astype(np.float32)


def expand_features(x):
    ratio = x[:, 0]
    norm = x[:, 1]
    radius = x[:, 2]
    radial = x[:, 3]
    log_ratio = np.log1p(ratio)
    return np.stack(
        [
            np.ones_like(ratio),
            ratio,
            ratio**2,
            ratio**3,
            log_ratio,
            norm,
            radius,
            radial,
            norm * ratio,
            radius * ratio,
            radial * ratio,
            norm / (np.mean(norm) + 1e-12),
            radius / (np.mean(radius) + 1e-12),
        ],
        axis=1,
    ).astype(np.float64)


def fit_ridge(x, y, reg):
    phi = expand_features(x)
    mean = phi[:, 1:].mean(axis=0)
    std = phi[:, 1:].std(axis=0) + 1e-8
    phi[:, 1:] = (phi[:, 1:] - mean) / std
    eye = np.eye(phi.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    w = np.linalg.solve(phi.T @ phi + reg * eye, phi.T @ y.astype(np.float64))
    return w, mean, std


def predict_alpha(x, w, mean, std, min_alpha, max_alpha):
    phi = expand_features(x)
    phi[:, 1:] = (phi[:, 1:] - mean) / std
    alpha = phi @ w
    return np.clip(alpha.astype(np.float32), min_alpha, max_alpha)


def collect_training(keys, base_dir, noisy_dir, gt_dir, k, max_points_per_sample, seed, target_max):
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for key in keys:
        noisy = np.load(os.path.join(noisy_dir, key, "noisy.npy")).astype(np.float32)
        pred = np.load(os.path.join(base_dir, key, "denoised.npy")).astype(np.float32)
        clean = np.load(os.path.join(gt_dir, key, "clean.npy")).astype(np.float32)
        disp = pred - noisy
        target = clean - noisy
        alpha = ((target * disp).sum(axis=1) / ((disp * disp).sum(axis=1) + 1e-12)).astype(np.float32)
        alpha = np.clip(alpha, 0.0, target_max)
        feat = point_features(noisy, pred, k)
        if max_points_per_sample > 0 and len(alpha) > max_points_per_sample:
            idx = rng.choice(len(alpha), size=max_points_per_sample, replace=False)
            feat = feat[idx]
            alpha = alpha[idx]
        xs.append(feat)
        ys.append(alpha)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--reg", type=float, default=100.0)
    parser.add_argument("--max-points-per-sample", type=int, default=12000)
    parser.add_argument("--target-max", type=float, default=1.5)
    parser.add_argument("--min-alpha", type=float, default=0.0)
    parser.add_argument("--max-alpha", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    keys = list(iter_keys(args.list))
    x, y = collect_training(
        keys,
        args.base_dir,
        args.noisy_dir,
        args.gt_dir,
        args.k,
        args.max_points_per_sample,
        args.seed,
        args.target_max,
    )
    w, mean, std = fit_ridge(x, y, args.reg)
    train_pred = predict_alpha(x, w, mean, std, args.min_alpha, args.max_alpha)
    print(
        f"fit n={len(y)} reg={args.reg} target_mean={y.mean():.4f} "
        f"pred_mean={train_pred.mean():.4f} mae={np.mean(np.abs(train_pred-y)):.4f}",
        flush=True,
    )
    print("weights", " ".join(f"{v:.6g}" for v in w), flush=True)

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    stats = []
    for key in keys:
        noisy = np.load(os.path.join(args.noisy_dir, key, "noisy.npy")).astype(np.float32)
        pred = np.load(os.path.join(args.base_dir, key, "denoised.npy")).astype(np.float32)
        feat = point_features(noisy, pred, args.k)
        alpha = predict_alpha(feat, w, mean, std, args.min_alpha, args.max_alpha)
        out = noisy + (pred - noisy) * alpha[:, None]
        out_path = os.path.join(args.out_dir, key, "denoised.npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, out.astype(np.float32))
        stats.append((float(alpha.mean()), float(np.linalg.norm(out - noisy, axis=1).mean())))
    stats = np.asarray(stats, dtype=np.float32)
    print(f"made {args.out_dir} mean_alpha={stats[:,0].mean():.4f} mean_disp={stats[:,1].mean():.6f}", flush=True)
    if args.zip:
        zip_dir(args.out_dir, f"{args.out_dir}.zip")


if __name__ == "__main__":
    main()
