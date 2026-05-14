#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_keys(list_path):
    if list_path is None:
        return None
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def tangent_project(pc, k, beta):
    out = pc.astype(np.float64, copy=True)
    tree = cKDTree(out)
    _, idx = tree.query(out, k=k, workers=-1)
    nb = out[idx]
    center = nb.mean(axis=1)
    centered = nb - center[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(idx.shape[1], 1)
    _, vec = np.linalg.eigh(cov)
    normal = vec[:, :, 0]
    signed = np.einsum("ni,ni->n", out - center, normal)
    projected = out - beta * signed[:, None] * normal
    return projected.astype(np.float32)


def edge_aware_tangent_project(pc, k, beta, edge_low=0.18, edge_high=0.55, clamp_scale=1.25):
    out = pc.astype(np.float64, copy=True)
    tree = cKDTree(out)
    dists, idx = tree.query(out, k=k, workers=-1)
    nb = out[idx]
    center = nb.mean(axis=1)
    centered = nb - center[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(idx.shape[1], 1)
    val, vec = np.linalg.eigh(cov)
    val = np.maximum(val, 1e-12)
    normal = vec[:, :, 0]
    edge_ratio = val[:, 1] / val[:, 2]
    confidence = np.clip((edge_ratio - edge_low) / max(edge_high - edge_low, 1e-6), 0.0, 1.0)
    signed = np.einsum("ni,ni->n", out - center, normal)
    clamp = np.maximum(dists[:, -1] * clamp_scale, 1e-8)
    signed = np.clip(signed, -clamp, clamp)
    projected = out - beta * confidence[:, None] * signed[:, None] * normal
    return projected.astype(np.float32)


def weighted_mls_project(pc, k, beta, sigma_scale=0.55, clamp_scale=1.5):
    out = pc.astype(np.float64, copy=True)
    tree = cKDTree(out)
    dists, idx = tree.query(out, k=k, workers=-1)
    nb = out[idx]
    sigma = np.maximum(dists[:, -1:] * sigma_scale, 1e-8)
    w = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    center = (nb * w[:, :, None]).sum(axis=1)
    centered = nb - center[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered * w[:, :, None], centered)
    _, vec = np.linalg.eigh(cov)
    normal = vec[:, :, 0]
    signed = np.einsum("ni,ni->n", out - center, normal)
    clamp = np.maximum(dists[:, -1] * clamp_scale, 1e-8)
    signed = np.clip(signed, -clamp, clamp)
    projected = out - beta * signed[:, None] * normal
    return projected.astype(np.float32)


def estimate_normals(pc, k):
    out = pc.astype(np.float64, copy=False)
    tree = cKDTree(out)
    _, idx = tree.query(out, k=k, workers=-1)
    normals = np.empty_like(out)
    for i, ids in enumerate(idx):
        nb = out[ids]
        center = nb.mean(axis=0)
        centered = nb - center
        cov = centered.T @ centered / max(len(ids), 1)
        _, vec = np.linalg.eigh(cov)
        normals[i] = vec[:, 0]
    return normals


def filter_normals(pc, normals, k, sigma_spatial_scale=0.55, sigma_normal_scale=0.35, iters=1):
    out = pc.astype(np.float64, copy=False)
    filt = normals.astype(np.float64, copy=True)
    tree = cKDTree(out)
    dists, idx = tree.query(out, k=k, workers=-1)
    for _ in range(iters):
        next_normals = np.empty_like(filt)
        for i, ids in enumerate(idx):
            center_n = filt[i]
            nb_n = filt[ids].copy()
            signs = np.sign(nb_n @ center_n)
            signs[signs == 0] = 1.0
            nb_n *= signs[:, None]

            radius = max(float(dists[i, -1]), 1e-8)
            sigma_spatial = max(radius * sigma_spatial_scale, 1e-8)
            sigma_normal = max(float(sigma_normal_scale), 1e-8)
            cos_dist = np.maximum(0.0, 1.0 - np.clip(nb_n @ center_n, -1.0, 1.0))
            w_spatial = np.exp(-(dists[i] ** 2) / (2.0 * sigma_spatial ** 2))
            w_normal = np.exp(-(cos_dist ** 2) / (2.0 * sigma_normal ** 2))
            w = w_spatial * w_normal
            n = (w[:, None] * nb_n).sum(axis=0)
            n_norm = np.linalg.norm(n)
            next_normals[i] = center_n if n_norm < 1e-12 else n / n_norm
        filt = next_normals
    return filt


def bilateral_project(pc, k, beta, normal_k=None, sigma_spatial_scale=0.55, sigma_normal_scale=0.45, clamp_scale=1.2):
    out = pc.astype(np.float64, copy=True)
    normal_k = normal_k or k
    normals = estimate_normals(out, normal_k)

    tree = cKDTree(out)
    dists, idx = tree.query(out, k=k, workers=-1)
    projected = out.copy()
    for i, ids in enumerate(idx):
        nb = out[ids]
        nb_normals = normals[ids]
        vec_to_plane = nb - out[i]
        signed = np.einsum("ij,ij->i", vec_to_plane, nb_normals)

        radius = max(float(dists[i, -1]), 1e-8)
        sigma_spatial = max(radius * sigma_spatial_scale, 1e-8)
        sigma_normal = max(radius * sigma_normal_scale, 1e-8)
        w_spatial = np.exp(-(dists[i] ** 2) / (2.0 * sigma_spatial ** 2))
        w_normal = np.exp(-(signed ** 2) / (2.0 * sigma_normal ** 2))
        w = w_spatial * w_normal
        w_sum = float(w.sum())
        if w_sum < 1e-12:
            continue

        clamp = max(radius * clamp_scale, 1e-8)
        signed = np.clip(signed, -clamp, clamp)
        correction = (w[:, None] * signed[:, None] * nb_normals).sum(axis=0) / w_sum
        projected[i] = out[i] + beta * correction
    return projected.astype(np.float32)


def normal_filter_project(
    pc,
    k,
    beta,
    normal_k=None,
    filter_k=None,
    filter_iters=1,
    sigma_spatial_scale=0.55,
    sigma_normal_scale=0.35,
    clamp_scale=1.2,
):
    out = pc.astype(np.float64, copy=True)
    normal_k = normal_k or k
    filter_k = filter_k or normal_k
    normals = estimate_normals(out, normal_k)
    normals = filter_normals(
        out,
        normals,
        k=filter_k,
        sigma_spatial_scale=sigma_spatial_scale,
        sigma_normal_scale=sigma_normal_scale,
        iters=filter_iters,
    )

    tree = cKDTree(out)
    dists, idx = tree.query(out, k=k, workers=-1)
    projected = out.copy()
    for i, ids in enumerate(idx):
        nb = out[ids]
        nb_normals = normals[ids]
        center = nb.mean(axis=0)
        signed = np.einsum("ij,ij->i", out[i][None, :] - nb, nb_normals)

        radius = max(float(dists[i, -1]), 1e-8)
        sigma_spatial = max(radius * sigma_spatial_scale, 1e-8)
        w_spatial = np.exp(-(dists[i] ** 2) / (2.0 * sigma_spatial ** 2))
        center_offset = np.linalg.norm(nb - center, axis=1)
        w_center = np.exp(-(center_offset ** 2) / (2.0 * sigma_spatial ** 2))
        w = w_spatial * w_center
        w_sum = float(w.sum())
        if w_sum < 1e-12:
            continue

        clamp = max(radius * clamp_scale, 1e-8)
        signed = np.clip(signed, -clamp, clamp)
        correction = (w[:, None] * signed[:, None] * nb_normals).sum(axis=0) / w_sum
        projected[i] = out[i] - beta * correction
    return projected.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing noisy.npy")
    parser.add_argument("--pred_dir", required=True, help="Directory containing denoised.npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list", default=None)
    parser.add_argument("--alpha", type=float, default=1.1)
    parser.add_argument("--project_k", type=int, default=48)
    parser.add_argument("--project_beta", type=float, default=0.6)
    parser.add_argument("--project_iters", type=int, default=1)
    parser.add_argument("--project_method", choices=["tangent", "edge_tangent", "weighted_mls", "bilateral", "normal_filter"], default="tangent")
    parser.add_argument("--edge_low", type=float, default=0.18)
    parser.add_argument("--edge_high", type=float, default=0.55)
    parser.add_argument("--edge_clamp_scale", type=float, default=1.25)
    parser.add_argument("--mls_sigma_scale", type=float, default=0.55)
    parser.add_argument("--mls_clamp_scale", type=float, default=1.5)
    parser.add_argument("--bilateral_normal_k", type=int, default=0)
    parser.add_argument("--bilateral_sigma_spatial_scale", type=float, default=0.55)
    parser.add_argument("--bilateral_sigma_normal_scale", type=float, default=0.45)
    parser.add_argument("--bilateral_clamp_scale", type=float, default=1.2)
    parser.add_argument("--normal_filter_k", type=int, default=0)
    parser.add_argument("--normal_filter_iters", type=int, default=1)
    parser.add_argument("--normal_filter_sigma_spatial_scale", type=float, default=0.55)
    parser.add_argument("--normal_filter_sigma_normal_scale", type=float, default=0.35)
    parser.add_argument("--normal_filter_clamp_scale", type=float, default=1.2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    keys = read_keys(args.list)
    if keys is None:
        keys = [str(p.parent.relative_to(input_dir)) for p in sorted(input_dir.rglob("noisy.npy"))]

    for n, key in enumerate(keys, start=1):
        noisy_path = input_dir / key / "noisy.npy"
        pred_path = pred_dir / key / "denoised.npy"
        out_path = output_dir / key / "denoised.npy"
        noisy = np.load(noisy_path).astype(np.float64)
        pred = np.load(pred_path).astype(np.float64)
        denoised = noisy + args.alpha * (pred - noisy)
        for _ in range(args.project_iters):
            if args.project_method == "tangent":
                denoised = tangent_project(denoised, k=args.project_k, beta=args.project_beta)
            elif args.project_method == "edge_tangent":
                denoised = edge_aware_tangent_project(
                    denoised,
                    k=args.project_k,
                    beta=args.project_beta,
                    edge_low=args.edge_low,
                    edge_high=args.edge_high,
                    clamp_scale=args.edge_clamp_scale,
                )
            elif args.project_method == "weighted_mls":
                denoised = weighted_mls_project(
                    denoised,
                    k=args.project_k,
                    beta=args.project_beta,
                    sigma_scale=args.mls_sigma_scale,
                    clamp_scale=args.mls_clamp_scale,
                )
            elif args.project_method == "bilateral":
                denoised = bilateral_project(
                    denoised,
                    k=args.project_k,
                    beta=args.project_beta,
                    normal_k=args.bilateral_normal_k or None,
                    sigma_spatial_scale=args.bilateral_sigma_spatial_scale,
                    sigma_normal_scale=args.bilateral_sigma_normal_scale,
                    clamp_scale=args.bilateral_clamp_scale,
                )
            else:
                denoised = normal_filter_project(
                    denoised,
                    k=args.project_k,
                    beta=args.project_beta,
                    normal_k=args.bilateral_normal_k or None,
                    filter_k=args.normal_filter_k or None,
                    filter_iters=args.normal_filter_iters,
                    sigma_spatial_scale=args.normal_filter_sigma_spatial_scale,
                    sigma_normal_scale=args.normal_filter_sigma_normal_scale,
                    clamp_scale=args.normal_filter_clamp_scale,
                )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, denoised)
        if n % 10 == 0 or n == len(keys):
            print(f"processed {n}/{len(keys)}")


if __name__ == "__main__":
    main()
