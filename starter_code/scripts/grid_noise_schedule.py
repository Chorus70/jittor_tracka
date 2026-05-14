#!/usr/bin/env python
import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_eval(text):
    patterns = {
        "mean_cd_noisy": r"平均 CD_noisy:\s+([0-9.]+)",
        "mean_p2s_noisy": r"平均 P2S_noisy:\s+([0-9.]+)",
        "cd_score": r"CD 得分:\s+([0-9.]+)",
        "p2s_score": r"P2S 得分:\s+([0-9.]+)",
        "final_score": r"最终得分 .*?:\s+([0-9.]+)",
    }
    out = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        out[key] = float(m.group(1)) if m else None
    return out


def run(cmd):
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    return proc.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", default="./dataset_train")
    parser.add_argument("--datalist", default="./datalist/validate.txt")
    parser.add_argument("--output_root", default="./local_eval_schedules")
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=99)
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--sigma_sets", required=True, help="Semicolon-separated sigma lists, e.g. '0.003,0.004;0.004,0.005'")
    parser.add_argument("--noise_type_sets", default="gaussian,anisotropic")
    parser.add_argument("--outlier_ps", default="0.0")
    parser.add_argument("--target_cd", type=float, default=0.000246)
    parser.add_argument("--target_p2s", type=float, default=0.000196)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", default="./noise_schedule_results.md")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    sigma_sets = [x.strip() for x in args.sigma_sets.split(";") if x.strip()]
    noise_type_sets = [x.strip() for x in args.noise_type_sets.split(";") if x.strip()]
    outlier_ps = [float(x) for x in args.outlier_ps.split(",") if x.strip()]

    rows = []
    for idx, (seed, sigmas, noise_types, outlier_p) in enumerate(
        itertools.product(seeds, sigma_sets, noise_type_sets, outlier_ps), start=1
    ):
        name = f"s{seed}_sig{sigmas.replace(',', '-')}_nt{noise_types.replace(',', '-')}_out{outlier_p:g}"
        eval_dir = output_root / name
        print(f"[{idx}] prepare {name}", flush=True)
        run(
            [
                sys.executable,
                "scripts/prepare_local_eval.py",
                "--mesh_dir",
                args.mesh_dir,
                "--datalist",
                args.datalist,
                "--output_dir",
                str(eval_dir),
                "--num_points",
                str(args.num_points),
                "--limit",
                str(args.limit),
                "--seed",
                str(seed),
                "--sigmas",
                sigmas,
                "--noise_types",
                noise_types,
                "--outlier_p",
                str(outlier_p),
            ]
        )
        noisy_as_pred = eval_dir / "noisy_as_pred"
        run(
            [
                sys.executable,
                "scripts/make_noisy_as_pred.py",
                "--noisy_dir",
                str(eval_dir / "noisy"),
                "--output_dir",
                str(noisy_as_pred),
                "--list",
                str(eval_dir / "test.txt"),
            ]
        )
        text = run(
            [
                sys.executable,
                "evaluate.py",
                "--pred_dir",
                str(noisy_as_pred),
                "--gt_dir",
                str(eval_dir / "gt"),
                "--noisy_dir",
                str(eval_dir / "noisy"),
                "--mesh_dir",
                args.mesh_dir,
                "--list",
                str(eval_dir / "test.txt"),
                "--workers",
                str(args.workers),
            ]
        )
        metrics = parse_eval(text)
        cd_err = abs(metrics["mean_cd_noisy"] - args.target_cd) / args.target_cd
        p2s_err = abs(metrics["mean_p2s_noisy"] - args.target_p2s) / args.target_p2s
        score = cd_err + p2s_err
        row = {
            "name": name,
            "eval_dir": str(eval_dir),
            "seed": seed,
            "sigmas": sigmas,
            "noise_types": noise_types,
            "outlier_p": outlier_p,
            "mean_cd_noisy": metrics["mean_cd_noisy"],
            "mean_p2s_noisy": metrics["mean_p2s_noisy"],
            "cd_rel_err": cd_err,
            "p2s_rel_err": p2s_err,
            "match_score": score,
        }
        rows.append(row)
        print(
            f"{name}: mean_cd_noisy={row['mean_cd_noisy']:.8f} "
            f"mean_p2s_noisy={row['mean_p2s_noisy']:.8f} "
            f"cd_err={cd_err:.3f} p2s_err={p2s_err:.3f}",
            flush=True,
        )

    rows.sort(key=lambda x: x["match_score"])
    report = ROOT / args.report
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Noise Schedule Search Results\n\n")
        f.write(f"Target mean_CD_noisy={args.target_cd:.8f}, mean_P2S_noisy={args.target_p2s:.8f}\n\n")
        f.write("| rank | eval_dir | sigmas | noise_types | outlier_p | mean_CD_noisy | mean_P2S_noisy | CD err | P2S err |\n")
        f.write("|---:|---|---|---|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(rows, start=1):
            f.write(
                f"| {rank} | `{row['eval_dir']}` | `{row['sigmas']}` | `{row['noise_types']}` | "
                f"{row['outlier_p']:.4f} | {row['mean_cd_noisy']:.8f} | {row['mean_p2s_noisy']:.8f} | "
                f"{row['cd_rel_err']:.3f} | {row['p2s_rel_err']:.3f} |\n"
            )
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
