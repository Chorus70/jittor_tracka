#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path


def read_keys(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_mapping(items):
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"mapping must be CATEGORY=DIR, got {item}")
        key, value = item.split("=", 1)
        mapping[key] = Path(value)
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--map", action="append", default=[], help="CATEGORY=prediction_dir")
    parser.add_argument("--filename", default="denoised.npy")
    args = parser.parse_args()

    default_dir = Path(args.default_dir)
    output_dir = Path(args.output_dir)
    mapping = parse_mapping(args.map)
    selected = {}
    for key in read_keys(args.list):
        parts = key.split("/")
        if len(parts) < 2:
            raise ValueError(f"cannot parse category from key: {key}")
        category = parts[1]
        src_dir = mapping.get(category, default_dir)
        src = src_dir / key / args.filename
        if not src.exists():
            raise FileNotFoundError(f"missing prediction for {key}: {src}")
        dst = output_dir / key / args.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        selected[category] = selected.get(category, 0) + 1
    for category, count in sorted(selected.items()):
        src_dir = mapping.get(category, default_dir)
        print(f"{category}: {count} <- {src_dir}")


if __name__ == "__main__":
    main()
