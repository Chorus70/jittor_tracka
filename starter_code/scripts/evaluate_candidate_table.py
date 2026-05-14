#!/usr/bin/env python
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_eval(text):
    patterns = {
        "mean_cd_pred": r"平均 CD_pred:\s+([0-9.]+)",
        "mean_cd_noisy": r"平均 CD_noisy:\s+([0-9.]+)",
        "cd_score": r"CD 得分:\s+([0-9.]+)",
        "mean_p2s_pred": r"平均 P2S_pred:\s+([0-9.]+)",
        "mean_p2s_noisy": r"平均 P2S_noisy:\s+([0-9.]+)",
        "p2s_score": r"P2S 得分:\s+([0-9.]+)",
        "final_score": r"最终得分 .*?:\s+([0-9.]+)",
    }
    out = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        out[key] = float(m.group(1)) if m else None
    return out


def fmt_metric(value, digits=2):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def mean_displacement(pred_dir, noisy_dir, list_path):
    pred_dir = Path(pred_dir)
    noisy_dir = Path(noisy_dir)
    keys = [line.strip() for line in open(list_path, "r", encoding="utf-8") if line.strip()]
    vals = []
    for key in keys:
        pred = np.load(pred_dir / key / "denoised.npy")
        noisy = np.load(noisy_dir / key / "noisy.npy")
        vals.append(np.linalg.norm(pred - noisy, axis=1).mean())
    return float(np.mean(vals)) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_dirs", required=True, help="Comma-separated prediction directories")
    parser.add_argument("--names", default="", help="Optional comma-separated display names")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--csv", default="candidate_baseline_table.csv")
    parser.add_argument("--md", default="candidate_baseline_table.md")
    args = parser.parse_args()

    candidate_dirs = [x.strip() for x in args.candidate_dirs.split(",") if x.strip()]
    names = [x.strip() for x in args.names.split(",") if x.strip()]
    if not names:
        names = [Path(x).name for x in candidate_dirs]
    if len(names) != len(candidate_dirs):
        raise ValueError("--names must match --candidate_dirs length")

    rows = []
    for name, pred_dir in zip(names, candidate_dirs):
        cmd = [
            sys.executable,
            "evaluate.py",
            "--pred_dir",
            pred_dir,
            "--gt_dir",
            args.gt_dir,
            "--noisy_dir",
            args.noisy_dir,
            "--mesh_dir",
            args.mesh_dir,
            "--list",
            args.list,
            "--workers",
            str(args.workers),
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        metrics = parse_eval(proc.stdout)
        metrics["name"] = name
        metrics["pred_dir"] = pred_dir
        metrics["mean_disp"] = mean_displacement(pred_dir, args.noisy_dir, args.list)
        rows.append(metrics)
        print(
            f"{name}: final={fmt_metric(metrics['final_score'])} cd={fmt_metric(metrics['cd_score'])} "
            f"p2s={fmt_metric(metrics['p2s_score'])} mean_disp={metrics['mean_disp']:.6f}",
            flush=True,
        )

    csv_path = ROOT / args.csv
    md_path = ROOT / args.md
    fields = [
        "name",
        "pred_dir",
        "final_score",
        "cd_score",
        "p2s_score",
        "mean_cd_pred",
        "mean_cd_noisy",
        "mean_p2s_pred",
        "mean_p2s_noisy",
        "mean_disp",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Candidate Baseline Table\n\n")
        f.write("| name | final | CD | P2S | mean_CD_pred | mean_CD_noisy | mean_P2S_pred | mean_P2S_noisy | mean_disp |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda x: x["final_score"], reverse=True):
            f.write(
                f"| {row['name']} | {fmt_metric(row['final_score'])} | {fmt_metric(row['cd_score'])} | "
                f"{fmt_metric(row['p2s_score'])} | {fmt_metric(row['mean_cd_pred'], 8)} | "
                f"{fmt_metric(row['mean_cd_noisy'], 8)} | {fmt_metric(row['mean_p2s_pred'], 8)} | "
                f"{fmt_metric(row['mean_p2s_noisy'], 8)} | {row['mean_disp']:.6f} |\n"
            )
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
