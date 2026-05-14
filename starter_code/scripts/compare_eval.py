import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate import (
    evaluate_single,
    filter_samples,
    find_meshes,
    find_samples,
)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dirs", nargs="+", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--list", required=True)
    args = parser.parse_args()

    gt = filter_samples(find_samples(args.gt_dir, "clean.npy"), args.list)
    noisy = filter_samples(find_samples(args.noisy_dir, "noisy.npy"), args.list)
    meshes = find_meshes(args.mesh_dir)

    print("name,final,cd,p2s,valid")
    for pred_dir in args.pred_dirs:
        pred = filter_samples(find_samples(pred_dir, "denoised.npy"), args.list)
        keys = sorted(set(pred) & set(gt) & set(noisy))
        rows = []
        for key in keys:
            rows.append(evaluate_single((key, pred[key], gt[key], noisy[key], meshes.get(key))))
        cd = mean([r[3] for r in rows])
        p2s = mean([r[6] for r in rows])
        final = None if cd is None or p2s is None else 0.5 * (cd + p2s)
        name = pred_dir.rstrip("/").split("/")[-1]
        if final is None:
            print(f"{name},nan,nan,nan,{len(rows)}")
        else:
            print(f"{name},{final:.4f},{cd:.4f},{p2s:.4f},{len(rows)}")


if __name__ == "__main__":
    main()
