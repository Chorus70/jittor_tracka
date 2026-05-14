#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import jittor as jt
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.parse import get_model


def load_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--transform_config", required=True)
    parser.add_argument("--source_ckpt", required=True)
    parser.add_argument("--output_ckpt", required=True)
    args = parser.parse_args()

    model = get_model(
        model_config=load_yaml(args.model_config),
        transform_config=load_yaml(args.transform_config),
    )
    source = jt.load(args.source_ckpt)
    target = model.state_dict()

    copied = []
    skipped_shape = []
    missing = []
    for name, value in target.items():
        if name not in source:
            missing.append(name)
            continue
        if tuple(source[name].shape) != tuple(value.shape):
            skipped_shape.append((name, tuple(source[name].shape), tuple(value.shape)))
            continue
        value.assign(source[name])
        copied.append(name)

    model.save(args.output_ckpt)
    print(f"saved {args.output_ckpt}")
    print(f"copied: {len(copied)}")
    print(f"missing_in_source: {len(missing)}")
    print(f"shape_mismatch: {len(skipped_shape)}")
    if skipped_shape:
        print("shape mismatches:")
        for item in skipped_shape[:40]:
            print(item)


if __name__ == "__main__":
    main()
