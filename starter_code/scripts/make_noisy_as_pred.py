#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy_dir", required=True, help="Directory containing **/noisy.npy")
    parser.add_argument("--output_dir", required=True, help="Directory to write **/denoised.npy")
    parser.add_argument("--list", default=None, help="Optional relative-key list")
    args = parser.parse_args()

    noisy_dir = Path(args.noisy_dir)
    output_dir = Path(args.output_dir)
    if args.list:
        keys = [line.strip() for line in open(args.list, "r", encoding="utf-8") if line.strip()]
        noisy_paths = [noisy_dir / key / "noisy.npy" for key in keys]
    else:
        noisy_paths = sorted(noisy_dir.rglob("noisy.npy"))

    written = 0
    for noisy_path in noisy_paths:
        if not noisy_path.exists():
            raise FileNotFoundError(noisy_path)
        rel = noisy_path.parent.relative_to(noisy_dir)
        out_path = output_dir / rel / "denoised.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.load(noisy_path).astype(np.float32))
        written += 1

    print(f"written={written} output_dir={output_dir}")


if __name__ == "__main__":
    main()
