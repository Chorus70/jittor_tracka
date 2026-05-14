#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.utils import sample_surface


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)


def add_noise(pc, sigma, noise_type, outlier_p, rng):
    if noise_type == "gaussian":
        noise = rng.normal(0.0, sigma, size=pc.shape)
    elif noise_type == "laplace":
        noise = rng.laplace(0.0, sigma, size=pc.shape)
    elif noise_type == "uniform":
        noise = rng.uniform(-np.sqrt(3.0) * sigma, np.sqrt(3.0) * sigma, size=pc.shape)
    elif noise_type == "anisotropic":
        scale = rng.uniform(0.4, 1.8, size=(1, 3))
        noise = rng.normal(0.0, sigma, size=pc.shape) * scale
    else:
        raise ValueError(f"unsupported noise type: {noise_type}")
    if outlier_p > 0:
        mask = rng.random((pc.shape[0], 1)) < outlier_p
        outliers = rng.normal(0.0, sigma * 3.0, size=pc.shape)
        noise = np.where(mask, outliers, noise)
    return pc + noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", default="./dataset_train")
    parser.add_argument("--datalist", default="./datalist/validate.txt")
    parser.add_argument("--output_dir", default="./local_eval")
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sigmas", default="0.005,0.010,0.015,0.020")
    parser.add_argument("--noise_types", default="gaussian,laplace,uniform,anisotropic")
    parser.add_argument("--outlier_p", type=float, default=0.005)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    gt_dir = output_dir / "gt"
    noisy_dir = output_dir / "noisy"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    rels = [x.strip() for x in open(args.datalist, "r").readlines() if x.strip()]
    if args.limit > 0:
        rels = rels[:args.limit]

    noise_types = [x.strip() for x in args.noise_types.split(",") if x.strip()]
    sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]
    written = []

    for idx, rel in enumerate(rels):
        mesh_path = Path(args.mesh_dir) / rel / "models/model_normalized.obj"
        vertices, faces = load_mesh(str(mesh_path))
        clean, _, _, _ = sample_surface(args.num_points, vertices, faces)
        sigma = sigmas[idx % len(sigmas)]
        noise_type = noise_types[(idx // len(sigmas)) % len(noise_types)]
        noisy = add_noise(clean, sigma=sigma, noise_type=noise_type, outlier_p=args.outlier_p, rng=rng)

        gt_sample_dir = gt_dir / rel
        noisy_sample_dir = noisy_dir / rel
        gt_sample_dir.mkdir(parents=True, exist_ok=True)
        noisy_sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(gt_sample_dir / "clean.npy", clean.astype(np.float32))
        np.save(noisy_sample_dir / "noisy.npy", noisy.astype(np.float32))
        written.append(rel)

    (output_dir / "test.txt").write_text("\n".join(written) + "\n")
    print(f"prepared {len(written)} local eval samples under {output_dir}")


if __name__ == "__main__":
    main()
