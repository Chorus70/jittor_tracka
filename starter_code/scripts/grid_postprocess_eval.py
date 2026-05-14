#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import evaluate_single
from postprocess_predictions import read_keys, tangent_project


def write_predictions(input_dir, pred_dir, output_dir, keys, alpha, project_k, project_beta, project_iters):
    input_dir = Path(input_dir)
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)
    for key in keys:
        noisy = np.load(input_dir / key / "noisy.npy").astype(np.float64)
        pred = np.load(pred_dir / key / "denoised.npy").astype(np.float64)
        denoised = noisy + alpha * (pred - noisy)
        for _ in range(project_iters):
            denoised = tangent_project(denoised, k=project_k, beta=project_beta)
        out_path = output_dir / key / "denoised.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, denoised.astype(np.float32))


def evaluate_dir(pred_dir, gt_dir, noisy_dir, mesh_dir, keys):
    rows = []
    for key in keys:
        rows.append(evaluate_single((
            key,
            str(Path(pred_dir) / key / "denoised.npy"),
            str(Path(gt_dir) / key / "clean.npy"),
            str(Path(noisy_dir) / key / "noisy.npy"),
            str(Path(mesh_dir) / key / "models/model_normalized.obj"),
        )))
    cd_scores = [r[3] for r in rows]
    p2s_scores = [r[6] for r in rows if r[6] is not None]
    cd = float(np.mean(cd_scores))
    p2s = float(np.mean(p2s_scores))
    return cd, p2s, 0.5 * (cd + p2s)


def parse_list(values, cast):
    return [cast(x) for x in values.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--alphas", default="1.0,1.1,1.2")
    parser.add_argument("--ks", default="40,48,64")
    parser.add_argument("--betas", default="0.45,0.6,0.75")
    parser.add_argument("--iters", default="1,2")
    args = parser.parse_args()

    keys = read_keys(args.list)
    assert keys
    best = None
    for alpha in parse_list(args.alphas, float):
        for k in parse_list(args.ks, int):
            for beta in parse_list(args.betas, float):
                for iters in parse_list(args.iters, int):
                    name = f"a{alpha:g}_k{k}_b{beta:g}_i{iters}"
                    out_dir = Path(args.output_root) / name
                    write_predictions(
                        input_dir=args.input_dir,
                        pred_dir=args.pred_dir,
                        output_dir=out_dir,
                        keys=keys,
                        alpha=alpha,
                        project_k=k,
                        project_beta=beta,
                        project_iters=iters,
                    )
                    cd, p2s, score = evaluate_dir(out_dir, args.gt_dir, args.input_dir, args.mesh_dir, keys)
                    print(f"{name}: final={score:.4f} cd={cd:.4f} p2s={p2s:.4f}", flush=True)
                    if best is None or score > best[0]:
                        best = (score, cd, p2s, name)
    if best is not None:
        print(f"BEST {best[3]} final={best[0]:.4f} cd={best[1]:.4f} p2s={best[2]:.4f}")


if __name__ == "__main__":
    main()
