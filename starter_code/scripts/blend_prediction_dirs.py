#!/usr/bin/env python
import argparse
import os
import shutil
import zipfile

import numpy as np


def iter_pred_files(pred_dir):
    for root, _, files in os.walk(pred_dir):
        if "denoised.npy" in files:
            rel = os.path.relpath(root, pred_dir)
            yield rel, os.path.join(root, "denoised.npy")


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
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--refine-dir", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--betas", nargs="+", type=float, required=True)
    args = parser.parse_args()

    for beta in args.betas:
        tag = f"b{int(round(beta * 1000)):03d}"
        out_dir = f"{args.out_prefix}_{tag}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        count = 0
        disp = []
        for rel, refine_path in iter_pred_files(args.refine_dir):
            base_path = os.path.join(args.base_dir, rel, "denoised.npy")
            if not os.path.exists(base_path):
                raise FileNotFoundError(base_path)
            base = np.load(base_path).astype(np.float32)
            refine = np.load(refine_path).astype(np.float32)
            out = base + beta * (refine - base)
            out_path = os.path.join(out_dir, rel, "denoised.npy")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, out.astype(np.float32))
            disp.append(float(np.linalg.norm(out - base, axis=1).mean()))
            count += 1
        zip_path = f"{out_dir}.zip"
        zip_dir(out_dir, zip_path)
        print(f"made beta={beta:.3f} count={count} mean_delta_from_base={np.mean(disp):.6f} zip={zip_path}")


if __name__ == "__main__":
    main()
