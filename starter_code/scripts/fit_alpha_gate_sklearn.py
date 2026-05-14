#!/usr/bin/env python
import argparse
import os
import shutil
import zipfile

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


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


def safe_norm(x):
    return np.linalg.norm(x, axis=1).astype(np.float32)


def pca_features(noisy, k):
    tree = cKDTree(noisy)
    dists, idx = tree.query(noisy, k=k + 1)
    neigh = noisy[idx[:, 1:]]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
    eig = np.linalg.eigvalsh(cov).astype(np.float32)
    eig = np.maximum(eig, 0.0)
    s = eig.sum(axis=1) + 1e-12
    linearity = (eig[:, 2] - eig[:, 1]) / s
    planarity = (eig[:, 1] - eig[:, 0]) / s
    scattering = eig[:, 0] / s
    anisotropy = (eig[:, 2] - eig[:, 0]) / (eig[:, 2] + 1e-12)
    return dists[:, 1:].astype(np.float32), np.stack([linearity, planarity, scattering, anisotropy], axis=1)


def point_features(noisy, pred, k):
    disp = pred - noisy
    disp_norm = safe_norm(disp)
    dists, pca = pca_features(noisy, k)
    radius_med = np.median(dists, axis=1).astype(np.float32)
    radius_max = dists[:, -1].astype(np.float32)
    radius_mean = dists.mean(axis=1).astype(np.float32)
    ratio_med = disp_norm / (radius_med + 1e-12)
    ratio_max = disp_norm / (radius_max + 1e-12)
    centered = noisy - noisy.mean(axis=0, keepdims=True)
    radial = safe_norm(centered)
    abs_disp = np.abs(disp)
    return np.concatenate(
        [
            disp,
            abs_disp,
            np.stack(
                [
                    disp_norm,
                    radius_med,
                    radius_max,
                    radius_mean,
                    ratio_med,
                    ratio_max,
                    radial,
                    noisy[:, 0],
                    noisy[:, 1],
                    noisy[:, 2],
                ],
                axis=1,
            ),
            pca,
        ],
        axis=1,
    ).astype(np.float32)


def load_xy(keys, base_dir, noisy_dir, gt_dir, k, max_points_per_sample, seed, target_max):
    rng = np.random.default_rng(seed)
    xs, ys, groups = [], [], []
    for sample_idx, key in enumerate(keys):
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
        groups.append(np.full(len(alpha), sample_idx, dtype=np.int32))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def make_model(args):
    if args.model == "rf":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbose=1,
        )
    return HistGradientBoostingRegressor(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2,
        random_state=args.seed,
        verbose=1,
    )


def apply_model(keys, model, base_dir, noisy_dir, out_dir, k, min_alpha, max_alpha):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    stats = []
    for key in keys:
        noisy = np.load(os.path.join(noisy_dir, key, "noisy.npy")).astype(np.float32)
        pred = np.load(os.path.join(base_dir, key, "denoised.npy")).astype(np.float32)
        feat = point_features(noisy, pred, k)
        alpha = np.clip(model.predict(feat).astype(np.float32), min_alpha, max_alpha)
        out = noisy + (pred - noisy) * alpha[:, None]
        out_path = os.path.join(out_dir, key, "denoised.npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, out.astype(np.float32))
        stats.append((float(alpha.mean()), float(np.linalg.norm(out - noisy, axis=1).mean())))
    stats = np.asarray(stats, dtype=np.float32)
    print(f"made {out_dir} mean_alpha={stats[:,0].mean():.4f} mean_disp={stats[:,1].mean():.6f}", flush=True)


def apply_oof_models(keys, fold_models, fold_key_indices, base_dir, noisy_dir, out_dir, k, min_alpha, max_alpha):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    stats = []
    key_to_fold = {}
    for fold_id, key_indices in enumerate(fold_key_indices):
        for key_idx in key_indices:
            key_to_fold[key_idx] = fold_id
    for key_idx, key in enumerate(keys):
        model = fold_models[key_to_fold[key_idx]]
        noisy = np.load(os.path.join(noisy_dir, key, "noisy.npy")).astype(np.float32)
        pred = np.load(os.path.join(base_dir, key, "denoised.npy")).astype(np.float32)
        feat = point_features(noisy, pred, k)
        alpha = np.clip(model.predict(feat).astype(np.float32), min_alpha, max_alpha)
        out = noisy + (pred - noisy) * alpha[:, None]
        out_path = os.path.join(out_dir, key, "denoised.npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, out.astype(np.float32))
        stats.append((float(alpha.mean()), float(np.linalg.norm(out - noisy, axis=1).mean())))
    stats = np.asarray(stats, dtype=np.float32)
    print(f"made_oof {out_dir} mean_alpha={stats[:,0].mean():.4f} mean_disp={stats[:,1].mean():.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--max-points-per-sample", type=int, default=8000)
    parser.add_argument("--target-max", type=float, default=1.5)
    parser.add_argument("--min-alpha", type=float, default=0.0)
    parser.add_argument("--max-alpha", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--max-features", type=float, default=0.75)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--oof-dir", default="")
    parser.add_argument("--oof-folds", type=int, default=0)
    args = parser.parse_args()

    keys = list(iter_keys(args.list))
    x, y, groups = load_xy(
        keys,
        args.base_dir,
        args.noisy_dir,
        args.gt_dir,
        args.k,
        args.max_points_per_sample,
        args.seed,
        args.target_max,
    )
    print(f"dataset x={x.shape} y_mean={y.mean():.4f} y_median={np.median(y):.4f}", flush=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x, y, groups))
    model = make_model(args)
    model.fit(x[train_idx], y[train_idx])
    pred_train = np.clip(model.predict(x[train_idx]).astype(np.float32), args.min_alpha, args.max_alpha)
    pred_val = np.clip(model.predict(x[val_idx]).astype(np.float32), args.min_alpha, args.max_alpha)
    print(
        f"alpha_mae train={mean_absolute_error(y[train_idx], pred_train):.4f} "
        f"val={mean_absolute_error(y[val_idx], pred_val):.4f} "
        f"pred_mean={pred_val.mean():.4f}",
        flush=True,
    )
    if args.model_path:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        joblib.dump(model, args.model_path)
        print(f"saved {args.model_path}", flush=True)
    apply_model(keys, model, args.base_dir, args.noisy_dir, args.out_dir, args.k, args.min_alpha, args.max_alpha)

    if args.oof_dir and args.oof_folds > 1:
        fold_models = []
        fold_key_indices = []
        gkf = GroupKFold(n_splits=args.oof_folds)
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups)):
            print(f"oof fold {fold + 1}/{args.oof_folds} train_points={len(tr_idx)} val_points={len(va_idx)}", flush=True)
            fold_model = make_model(args)
            fold_model.fit(x[tr_idx], y[tr_idx])
            pred_va = np.clip(fold_model.predict(x[va_idx]).astype(np.float32), args.min_alpha, args.max_alpha)
            print(f"oof fold {fold + 1} val_mae={mean_absolute_error(y[va_idx], pred_va):.4f}", flush=True)
            fold_models.append(fold_model)
            fold_key_indices.append(np.unique(groups[va_idx]))
        apply_oof_models(
            keys,
            fold_models,
            fold_key_indices,
            args.base_dir,
            args.noisy_dir,
            args.oof_dir,
            args.k,
            args.min_alpha,
            args.max_alpha,
        )
    if args.zip:
        zip_dir(args.out_dir, f"{args.out_dir}.zip")


if __name__ == "__main__":
    main()
