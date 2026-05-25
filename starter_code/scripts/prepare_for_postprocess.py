#!/usr/bin/env python3
"""准备预测目录并在需要时调用后处理脚本。

用法示例：
1) 只准备目录结构（对每个 noisy.npy 创建相同路径的 predictions 目录）：
   python3 prepare_for_postprocess.py --input_dir dataset_test_noisy --pred_dir starter_code/predictions --prepare

2) 用 noisy 作为占位预测（拷贝 noisy -> denoised.npy）：
   python3 prepare_for_postprocess.py --input_dir dataset_test_noisy --pred_dir starter_code/predictions --copy-noisy

3) 如果已经把同学的 denoised.npy 放到 predictions 相应目录，直接运行后处理：
   python3 prepare_for_postprocess.py --input_dir dataset_test_noisy --pred_dir starter_code/predictions --run-postprocess --output_dir starter_code/predictions_postprocessed --list datalist/test.txt

脚本不会覆盖已有的 denoised.npy（除非使用 --overwrite）。
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path


def iter_noisy_keys(input_dir):
    for p in sorted(Path(input_dir).rglob("noisy.npy")):
        # relative key like shapenet/<synset>/<model>
        yield str(p.parent.relative_to(input_dir))


def prepare_dirs(input_dir, pred_dir, copy_noisy=False, overwrite=False):
    input_dir = Path(input_dir)
    pred_dir = Path(pred_dir)
    n = 0
    for key in iter_noisy_keys(input_dir):
        src = input_dir / key / "noisy.npy"
        dst_dir = pred_dir / key
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "denoised.npy"
        if copy_noisy:
            if dst.exists() and not overwrite:
                # skip existing
                pass
            else:
                shutil.copy2(src, dst)
                n += 1
        else:
            n += 1
    print(f"prepared {n} prediction entries under {pred_dir}")


def run_postprocess(input_dir, pred_dir, output_dir, list_path=None, extra_args=None):
    cmd = [
        "python3",
        str(Path(__file__).parent / "postprocess_predictions.py"),
        "--input_dir",
        str(input_dir),
        "--pred_dir",
        str(pred_dir),
        "--output_dir",
        str(output_dir),
    ]
    if list_path:
        cmd += ["--list", str(list_path)]
    if extra_args:
        cmd += extra_args
    print("running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="test_noisy input dir (root)")
    parser.add_argument("--pred_dir", required=True, help="where predictions will live")
    parser.add_argument("--prepare", action="store_true", help="only ensure prediction dirs exist")
    parser.add_argument("--copy-noisy", action="store_true", help="copy noisy.npy -> denoised.npy as placeholder")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing denoised.npy when copying")
    parser.add_argument("--run-postprocess", action="store_true", help="after predictions exist, run postprocess_predictions.py")
    parser.add_argument("--output_dir", default=None, help="postprocess output dir")
    parser.add_argument("--list", default=None, help="datalist file relative to repo root or absolute")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="extra args forwarded to postprocess_predictions.py")
    args = parser.parse_args()

    if args.prepare or args.copy_noisy:
        prepare_dirs(args.input_dir, args.pred_dir, copy_noisy=args.copy_noisy, overwrite=args.overwrite)

    if args.run_postprocess:
        if args.output_dir is None:
            raise SystemExit("--output_dir required when --run-postprocess")
        extra = args.extra or []
        run_postprocess(args.input_dir, args.pred_dir, args.output_dir, list_path=args.list, extra_args=extra)


if __name__ == "__main__":
    main()
