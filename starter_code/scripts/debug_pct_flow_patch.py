#!/usr/bin/env python
import math
import os
import sys

import jittor as jt
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.parse import get_model
from src.model.vm import farthest_point_sampling, knn_points


def main():
    jt.flags.use_cuda = 1
    model_config = OmegaConf.to_container(OmegaConf.load("configs/model/vm_pct_flow_conditioned.yaml"))
    transform_config = OmegaConf.to_container(OmegaConf.load("configs/transform/finetune_pct_direct.yaml"))
    model = get_model(model_config=model_config, transform_config=transform_config)
    model.load("experiments/vm_pct_flow_conditioned/checkpoint_e0_s100.pkl")
    model.eval()

    path = "local_eval_testcats_100/noisy/shapenet/02691156/1e44b99c8de5eb01ebc54ed98d6399b2/noisy.npy"
    pc = np.load(path).astype("float32")
    x = jt.array(pc)
    n, d = x.shape
    patch_size = 1024
    seed_k = 8
    weight_gamma = 4.5
    noise_std = 0.0125
    num_patches = int(seed_k * n / patch_size)
    seed_pnts, _ = farthest_point_sampling(x.unsqueeze(0), num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, x.unsqueeze(0), patch_size)
    patches = patches[0]
    patch_dists = patch_dists[0]
    point_idxs = point_idxs[0]
    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)

    outs = []
    step = int(math.ceil(n / patch_size))
    i = 0
    while i < num_patches:
        out, _ = model.denoise_langevin_dynamics(patches[i : i + step], num_steps=3, noise_std=noise_std)
        outs.append(out)
        i += step
    den = jt.concat(outs, dim=0) + seed_expand

    point_idxs_np = point_idxs.detach().numpy()
    den_np = den.detach().numpy()
    patch_dists_np = patch_dists.detach().numpy()
    patch_weights_np = np.exp(-weight_gamma * patch_dists_np).astype(np.float64)

    print("idx", point_idxs_np.dtype, point_idxs_np.shape, point_idxs_np.min(), point_idxs_np.max(), point_idxs_np[0, :8])
    raw_disp = den_np[0].astype(np.float64) - pc[point_idxs_np[0]].astype(np.float64)
    print("raw first patch", np.linalg.norm(raw_disp, axis=1).mean(), np.linalg.norm(raw_disp, axis=1).max())
    print("weights first patch", patch_weights_np[0].min(), patch_weights_np[0].max(), patch_weights_np[0].mean())
    test_sum = np.zeros((n, 1), dtype=np.float64)
    np.add.at(test_sum, point_idxs_np[0], patch_weights_np[0][:, None])
    print("test add", test_sum.min(), test_sum.max(), test_sum.sum(), (test_sum[:, 0] > 0).sum())

    pcl_sum_np = np.zeros((n, d), dtype=np.float64)
    weight_sum_np = np.zeros((n, 1), dtype=np.float64)
    max_disp = 2.0 * noise_std
    for patch_id in range(num_patches):
        point_ids = point_idxs_np[patch_id]
        w = patch_weights_np[patch_id][:, None]
        patch_base = pc[point_ids].astype(np.float64)
        patch_pred = den_np[patch_id].astype(np.float64)
        disp = patch_pred - patch_base
        disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
        over = np.maximum(disp_norm / max(max_disp, 1e-8) - 1.0, 0.0)
        w = w * np.exp(-10.0 * over * over)
        disp = disp * np.minimum(1.0, max_disp / (disp_norm + 1e-8))
        value = patch_base + disp
        np.add.at(pcl_sum_np, point_ids, value * w)
        np.add.at(weight_sum_np, point_ids, w)

    valid = weight_sum_np[:, 0] > 1e-12
    print("valid", valid.mean(), weight_sum_np.min(), weight_sum_np.max(), weight_sum_np.mean())
    out_np = np.empty((n, d), dtype=np.float32)
    out_np[valid] = (pcl_sum_np[valid] / weight_sum_np[valid]).astype(np.float32)
    out_np[~valid] = pc[~valid]
    final_disp = out_np - pc
    print("final", np.linalg.norm(final_disp, axis=1).mean(), np.linalg.norm(final_disp, axis=1).max(), np.abs(final_disp).max())


if __name__ == "__main__":
    main()
