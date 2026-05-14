import argparse
import os
from pathlib import Path

import numpy as np


def read_list(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(line)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--save_name", default="denoised.npy")
    args = parser.parse_args()

    pred_dirs = [Path(p) for p in args.pred_dirs]
    if args.weights is None:
        weights = np.ones((len(pred_dirs),), dtype=np.float64)
    else:
        assert len(args.weights) == len(pred_dirs), "weights must match pred_dirs"
        weights = np.asarray(args.weights, dtype=np.float64)
    weights = weights / weights.sum()

    output_dir = Path(args.output_dir)
    for rel in read_list(args.list):
        arrays = []
        for pred_dir in pred_dirs:
            path = pred_dir / rel / args.save_name
            if not path.exists():
                raise FileNotFoundError(path)
            arrays.append(np.load(path).astype(np.float64))
        shape = arrays[0].shape
        if any(a.shape != shape for a in arrays):
            raise ValueError(f"shape mismatch for {rel}")
        merged = sum(w * a for w, a in zip(weights, arrays)).astype(np.float32)
        out_path = output_dir / rel / args.save_name
        os.makedirs(out_path.parent, exist_ok=True)
        np.save(out_path, merged)

    print(f"ensembled {len(read_list(args.list))} samples from {len(pred_dirs)} dirs")


if __name__ == "__main__":
    main()
