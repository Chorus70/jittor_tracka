#!/usr/bin/env python
import argparse
import os
import shutil
import zipfile

import numpy as np


def iter_pred_files(pred_dir):
    for root, _, files in os.walk(pred_dir):
        if "denoised.npy" in files:
            pred_path = os.path.join(root, "denoised.npy")
            rel_dir = os.path.relpath(root, pred_dir)
            yield rel_dir, pred_path


def zip_dir(src_dir, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                path = os.path.join(root, name)
                arc = os.path.relpath(path, src_dir)
                zf.write(path, arc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--replace-result-alpha", type=float, default=None)
    args = parser.parse_args()

    root = os.getcwd()
    made = []
    for alpha in args.alphas:
        tag = f"a{int(round(alpha * 1000)):03d}"
        out_dir = f"{args.out_prefix}_{tag}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        count = 0
        disp_norms = []
        for rel_dir, pred_path in iter_pred_files(args.pred_dir):
            noisy_path = os.path.join(args.noisy_dir, rel_dir, "noisy.npy")
            if not os.path.exists(noisy_path):
                raise FileNotFoundError(noisy_path)
            pred = np.load(pred_path).astype(np.float32)
            noisy = np.load(noisy_path).astype(np.float32)
            if pred.shape != noisy.shape:
                raise ValueError(f"shape mismatch: {pred_path} {pred.shape} vs {noisy_path} {noisy.shape}")
            blended = noisy + alpha * (pred - noisy)
            out_path = os.path.join(out_dir, rel_dir, "denoised.npy")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, blended.astype(np.float32))
            disp_norms.append(float(np.linalg.norm(blended - noisy, axis=1).mean()))
            count += 1
        zip_path = f"{out_dir}.zip"
        zip_dir(out_dir, zip_path)
        made.append((alpha, out_dir, zip_path, count, float(np.mean(disp_norms))))
        print(f"made alpha={alpha:.3f} count={count} mean_disp={made[-1][4]:.6f} zip={zip_path}")

    if args.replace_result_alpha is not None:
        chosen = min(made, key=lambda x: abs(x[0] - args.replace_result_alpha))
        backup = "result_backup_before_official_blend.zip"
        if os.path.exists("result.zip") and not os.path.exists(backup):
            shutil.copy2("result.zip", backup)
        shutil.copy2(chosen[2], "result.zip")
        print(f"result.zip replaced with {chosen[2]}")


if __name__ == "__main__":
    main()
