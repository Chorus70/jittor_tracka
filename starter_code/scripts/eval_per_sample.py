#!/usr/bin/env python
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate import evaluate_single, filter_samples, find_meshes, find_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dirs", nargs="+", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    gt = filter_samples(find_samples(args.gt_dir, "clean.npy"), args.list)
    noisy = filter_samples(find_samples(args.noisy_dir, "noisy.npy"), args.list)
    meshes = find_meshes(args.mesh_dir)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pred_dir",
                "name",
                "key",
                "category",
                "model_id",
                "cd_score",
                "p2s_score",
                "final_score",
                "cd_pred",
                "cd_noisy",
                "p2s_pred",
                "p2s_noisy",
            ],
        )
        writer.writeheader()
        for pred_dir in args.pred_dirs:
            pred = filter_samples(find_samples(pred_dir, "denoised.npy"), args.list)
            name = pred_dir.rstrip("/").split("/")[-1]
            keys = sorted(set(pred) & set(gt) & set(noisy))
            for key in keys:
                row = evaluate_single((key, pred[key], gt[key], noisy[key], meshes.get(key)))
                _, cd_pred, cd_noisy, cd_score, p2s_pred, p2s_noisy, p2s_score = row
                final = None if p2s_score is None else 0.5 * (cd_score + p2s_score)
                parts = key.split("/")
                writer.writerow(
                    {
                        "pred_dir": pred_dir,
                        "name": name,
                        "key": key,
                        "category": parts[0] if parts else "",
                        "model_id": parts[1] if len(parts) > 1 else "",
                        "cd_score": cd_score,
                        "p2s_score": p2s_score,
                        "final_score": final,
                        "cd_pred": cd_pred,
                        "cd_noisy": cd_noisy,
                        "p2s_pred": p2s_pred,
                        "p2s_noisy": p2s_noisy,
                    }
                )


if __name__ == "__main__":
    main()
