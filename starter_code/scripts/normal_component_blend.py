#!/usr/bin/env python
import argparse
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_keys(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def estimate_normals(pc, k):
    pc64 = pc.astype(np.float64, copy=False)
    tree = cKDTree(pc64)
    _, idx = tree.query(pc64, k=k, workers=-1)
    nb = pc64[idx]
    center = nb.mean(axis=1)
    centered = nb - center[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    _, vec = np.linalg.eigh(cov)
    return vec[:, :, 0]


def zip_dir(src_dir, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                path = os.path.join(root, name)
                zf.write(path, os.path.relpath(path, src_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--k", nargs="+", type=int, required=True)
    parser.add_argument("--normal-alpha", nargs="+", type=float, required=True)
    parser.add_argument("--tangent-alpha", nargs="+", type=float, default=[0.0])
    args = parser.parse_args()

    keys = read_keys(args.list)
    noisy_dir = Path(args.noisy_dir)
    pred_dir = Path(args.pred_dir)
    for k in args.k:
        for normal_alpha in args.normal_alpha:
            for tangent_alpha in args.tangent_alpha:
                tag = f"k{k}_n{int(round(normal_alpha * 1000)):03d}_t{int(round(tangent_alpha * 1000)):03d}"
                out_dir = Path(f"{args.out_prefix}_{tag}")
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                disp_stats = []
                for key in keys:
                    noisy = np.load(noisy_dir / key / "noisy.npy").astype(np.float32)
                    pred = np.load(pred_dir / key / "denoised.npy").astype(np.float32)
                    normals = estimate_normals(noisy, k=min(k, len(noisy)))
                    disp = pred.astype(np.float64) - noisy.astype(np.float64)
                    normal_scalar = np.einsum("ni,ni->n", disp, normals)
                    normal_disp = normal_scalar[:, None] * normals
                    tangent_disp = disp - normal_disp
                    out = noisy.astype(np.float64) + normal_alpha * normal_disp + tangent_alpha * tangent_disp
                    out_path = out_dir / key / "denoised.npy"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, out.astype(np.float32))
                    disp_stats.append(float(np.linalg.norm(out - noisy, axis=1).mean()))
                zip_path = f"{out_dir}.zip"
                zip_dir(str(out_dir), zip_path)
                print(
                    f"made {out_dir.name} count={len(keys)} mean_disp={np.mean(disp_stats):.6f} zip={zip_path}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
