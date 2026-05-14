#!/usr/bin/env python
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import chamfer_distance, load_mesh_vf, metric_to_score, point_to_surface_distance
from scripts.postprocess_predictions import (
    edge_aware_tangent_project,
    tangent_project,
    weighted_mls_project,
)


def read_keys(path):
    return [line.strip() for line in open(path, "r", encoding="utf-8") if line.strip()]


def project(pc, method, k, beta):
    if method == "tangent":
        return tangent_project(pc, k=k, beta=beta)
    if method == "edge_tangent":
        return edge_aware_tangent_project(pc, k=k, beta=beta)
    if method == "weighted_mls":
        return weighted_mls_project(pc, k=k, beta=beta)
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--mesh_dir", default="dataset_train")
    parser.add_argument("--list", required=True)
    parser.add_argument("--methods", default="tangent,edge_tangent,weighted_mls")
    parser.add_argument("--ks", default="16,24,32,40,48,64")
    parser.add_argument("--betas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.8")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    mesh_dir = Path(args.mesh_dir)
    keys = read_keys(args.list)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    betas = [float(x) for x in args.betas.split(",") if x.strip()]

    cache = []
    for key in keys:
        noisy = np.load(eval_dir / "noisy" / key / "noisy.npy").astype(np.float64)
        clean = np.load(eval_dir / "gt" / key / "clean.npy").astype(np.float64)
        mesh_v, mesh_f = load_mesh_vf(str(mesh_dir / key / "models/model_normalized.obj"))
        cd_noisy = chamfer_distance(noisy, clean, normalize=True)
        p2s_noisy = point_to_surface_distance(noisy, mesh_v, mesh_f, normalize_ref_pc=clean)
        cache.append((key, noisy, clean, mesh_v, mesh_f, cd_noisy, p2s_noisy))

    rows = []
    for method, k, beta in itertools.product(methods, ks, betas):
        cd_scores = []
        p2s_scores = []
        cd_preds = []
        p2s_preds = []
        for _, noisy, clean, mesh_v, mesh_f, cd_noisy, p2s_noisy in cache:
            pred = project(noisy, method, k, beta).astype(np.float64)
            cd_pred = chamfer_distance(pred, clean, normalize=True)
            p2s_pred = point_to_surface_distance(pred, mesh_v, mesh_f, normalize_ref_pc=clean)
            cd_scores.append(metric_to_score(cd_pred, cd_noisy))
            p2s_scores.append(metric_to_score(p2s_pred, p2s_noisy))
            cd_preds.append(cd_pred)
            p2s_preds.append(p2s_pred)
        cd = float(np.mean(cd_scores))
        p2s = float(np.mean(p2s_scores))
        rows.append((0.5 * (cd + p2s), cd, p2s, float(np.mean(cd_preds)), float(np.mean(p2s_preds)), method, k, beta))
        print(
            f"score={rows[-1][0]:.3f} cd={cd:.3f} p2s={p2s:.3f} "
            f"mean_cd={rows[-1][3]:.8f} mean_p2s={rows[-1][4]:.8f} "
            f"method={method} k={k} beta={beta}",
            flush=True,
        )

    print("\nTOP")
    for row in sorted(rows, reverse=True)[: args.topk]:
        score, cd, p2s, mean_cd, mean_p2s, method, k, beta = row
        print(
            f"score={score:.3f} cd={cd:.3f} p2s={p2s:.3f} "
            f"mean_cd={mean_cd:.8f} mean_p2s={mean_p2s:.8f} "
            f"method={method} k={k} beta={beta}"
        )


if __name__ == "__main__":
    main()
