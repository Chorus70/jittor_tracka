#!/usr/bin/env python
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    import point_cloud_utils as pcu
except ImportError as exc:
    raise SystemExit("point_cloud_utils is required for mesh projection") from exc


def read_keys(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def category_of(key):
    parts = key.split("/")
    if len(parts) < 2:
        raise ValueError(f"cannot parse category from {key}")
    return parts[1]


def model_id_of(key):
    parts = key.split("/")
    if len(parts) < 3:
        raise ValueError(f"cannot parse model id from {key}")
    return parts[2]


def load_mesh(mesh_path):
    # TinyObjReader prints missing-material warnings directly for many
    # ShapeNet OBJ files; geometry still loads correctly.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        v, f = pcu.load_mesh_vf(str(mesh_path))
    return v.astype(np.float64), f.astype(np.int32)


def sample_rows(x, n, seed):
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]


def chamfer(a, b):
    ta = cKDTree(a)
    tb = cKDTree(b)
    db, _ = ta.query(b, k=1, workers=-1)
    da, _ = tb.query(a, k=1, workers=-1)
    return float((da * da).mean() + (db * db).mean())


def closest_points_on_mesh(points, vertices, faces):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _, face_ids, bc = pcu.closest_points_on_mesh(
            points.astype(np.float32),
            vertices.astype(np.float32),
            faces,
        )
    tri = vertices[faces[face_ids]]
    closest = (tri * bc[:, :, None]).sum(axis=1)
    finite = np.isfinite(closest).all(axis=1)
    if not finite.all():
        closest = closest.copy()
        closest[~finite] = points[~finite]
    return closest


def build_candidates(train_dir, category, max_candidates, exclude_model_id=None, seed=0, mesh_points=2048):
    meshes = sorted((Path(train_dir) / "shapenet" / category).glob("*/models/model_normalized.obj"))
    if exclude_model_id is not None:
        meshes = [p for p in meshes if p.parts[-3] != exclude_model_id]
    if max_candidates > 0 and len(meshes) > max_candidates:
        rng = np.random.default_rng(seed)
        meshes = [meshes[i] for i in rng.choice(len(meshes), size=max_candidates, replace=False)]
    candidates = []
    for i, mesh in enumerate(meshes):
        v, f = load_mesh(mesh)
        probe = sample_rows(v, mesh_points, seed + i)
        candidates.append((mesh, v, f, probe))
    return candidates


def retrieve(noisy_probe, candidates):
    best = None
    for mesh, v, f, probe in candidates:
        score = chamfer(noisy_probe, probe)
        if best is None or score < best[0]:
            best = (score, mesh, v, f)
    if best is None:
        raise RuntimeError("no retrieval candidates")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing noisy.npy")
    parser.add_argument("--pred_dir", required=True, help="Directory containing denoised.npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_dir", default="./dataset_train")
    parser.add_argument("--list", required=True)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--retrieve_points", type=int, default=1024)
    parser.add_argument("--mesh_points", type=int, default=2048)
    parser.add_argument("--surface_beta", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--exclude_same_id", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    cache = {}
    for n, key in enumerate(read_keys(args.list), start=1):
        category = category_of(key)
        exclude = model_id_of(key) if args.exclude_same_id else None
        cache_key = (category, exclude)
        if cache_key not in cache:
            cache[cache_key] = build_candidates(
                args.train_dir,
                category,
                max_candidates=args.max_candidates,
                exclude_model_id=exclude,
                seed=args.seed,
                mesh_points=args.mesh_points,
            )

        noisy = np.load(input_dir / key / "noisy.npy").astype(np.float64)
        pred = np.load(pred_dir / key / "denoised.npy").astype(np.float64)
        noisy_probe = sample_rows(noisy, args.retrieve_points, args.seed + n)
        score, mesh, v, f = retrieve(noisy_probe, cache[cache_key])
        surface = closest_points_on_mesh(pred, v, f)
        out = pred + args.surface_beta * (surface - pred)

        out_path = output_dir / key / "denoised.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out.astype(np.float32))
        if n % 5 == 0 or n == 1:
            print(f"processed {n}: {key} <- {mesh.parts[-3]} score={score:.6g}")


if __name__ == "__main__":
    main()
