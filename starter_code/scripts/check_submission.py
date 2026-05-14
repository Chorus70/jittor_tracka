#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./dataset_test_noisy")
    parser.add_argument("--pred_dir", default="./results/dataset_test_noisy")
    parser.add_argument("--list", default=None)
    args = parser.parse_args()

    if args.list is None:
        inputs = sorted(Path(args.input_dir).rglob("noisy.npy"))
    else:
        with open(args.list, "r", encoding="utf-8") as f:
            inputs = [
                Path(args.input_dir) / line.strip() / "noisy.npy"
                for line in f
                if line.strip()
            ]
    missing = []
    bad = []
    for noisy_path in inputs:
        rel_dir = noisy_path.parent.relative_to(args.input_dir)
        pred_path = Path(args.pred_dir) / rel_dir / "denoised.npy"
        if not pred_path.exists():
            missing.append(str(rel_dir))
            continue
        noisy = np.load(noisy_path)
        pred = np.load(pred_path)
        if pred.dtype != np.float32 or pred.shape != noisy.shape or pred.ndim != 2 or pred.shape[1] != 3:
            bad.append((str(rel_dir), pred.shape, str(pred.dtype), noisy.shape, str(noisy.dtype)))
        elif not np.isfinite(pred).all():
            bad.append((str(rel_dir), "non-finite", str(pred.dtype), noisy.shape, str(noisy.dtype)))

    print(f"inputs={len(inputs)} missing={len(missing)} bad={len(bad)}")
    if missing:
        print("missing examples:", missing[:10])
    if bad:
        print("bad examples:", bad[:10])
    if missing or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
