from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .asset import Asset
from .spec import ConfigSpec
from .utils import random_euler_rotation, sample_vertex_groups

@dataclass(frozen=True)
class Augment(ConfigSpec):
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Augment':
        pass
    
    @abstractmethod
    def apply(self, asset: Asset, **kwargs):
        pass

@dataclass(frozen=True)
class AugmentSample(Augment):
    
    num_samples: int # total number of vertices on the face to be sampled
    
    num_vertex_samples: int=0 # number of vertices to be chosen
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentSample':
        cls.check_keys(kwargs)
        return AugmentSample(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        assert asset.vertices is not None
        assert asset.faces is not None
        sampled_vertices, sampled_normals, sampled_vertex_groups, hidden_states = sample_vertex_groups(
            vertices=asset.vertices,
            faces=asset.faces,
            num_samples=self.num_samples,
            num_vertex_samples=self.num_vertex_samples,
        )
        asset.sampled_vertices = sampled_vertices

@dataclass(frozen=True)
class AugmentNormalizePC(Augment):
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizePC':
        cls.check_keys(kwargs)
        return AugmentNormalizePC(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentNormalizePC"
        p_max = pc.max(axis=0)
        p_min = pc.min(axis=0)
        center = (p_max + p_min) / 2
        pc = pc - center
        scale = np.sqrt((pc**2).sum(axis=1).max()).max()
        asset.sampled_vertices = pc / scale

@dataclass(frozen=True)
class AugmentAddNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float

    distribution: Union[str, List[str]]="laplace"

    anisotropic_p: float=0.0

    outlier_p: float=0.0

    outlier_std_multiplier: float=3.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)

        distributions = self.distribution
        if isinstance(distributions, str):
            if distributions == "mixed":
                distributions = ["gaussian", "laplace", "uniform"]
            else:
                distributions = [distributions]
        distribution = np.random.choice(distributions)

        if distribution == "gaussian":
            noise = np.random.normal(0, noise_std, size=pc.shape)
        elif distribution == "laplace":
            noise = np.random.laplace(0, noise_std, size=pc.shape)
        elif distribution == "uniform":
            bound = np.sqrt(3.0) * noise_std
            noise = np.random.uniform(-bound, bound, size=pc.shape)
        else:
            raise ValueError(f"unsupported noise distribution: {distribution}")

        if self.anisotropic_p > 0 and np.random.rand() < self.anisotropic_p:
            axis_scale = np.random.uniform(0.4, 1.8, size=(1, 3))
            noise = noise * axis_scale

        if self.outlier_p > 0:
            mask = np.random.rand(pc.shape[0], 1) < self.outlier_p
            outliers = np.random.normal(
                0,
                noise_std * self.outlier_std_multiplier,
                size=pc.shape,
            )
            noise = np.where(mask, outliers, noise)

        asset.sampled_vertices_noisy = pc + noise
        if asset.meta is None:
            asset.meta = {}
        asset.meta['noise_std'] = np.float32(noise_std)
        asset.meta['noise_rms'] = np.float32(np.sqrt(np.mean(noise.astype(np.float64) ** 2.0)))

@dataclass(frozen=True)
class AugmentLinear(Augment):
    
    scale: Tuple[float, float]=(1.0, 1.0)
    
    rotate_x_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_y_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_z_range: Tuple[float, float]=(0.0, 0.0)
    
    scale_p: float=0.0
    
    rotate_p: float=0.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinear':
        cls.check_keys(kwargs)
        return AugmentLinear(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            r = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans_vertex = r @ trans_vertex
        if np.random.rand() < self.scale_p:
            scale = np.zeros((4, 4), dtype=np.float32)
            scale[0, 0] = np.random.uniform(self.scale[0], self.scale[1])
            scale[1, 1] = np.random.uniform(self.scale[0], self.scale[1])
            scale[2, 2] = np.random.uniform(self.scale[0], self.scale[1])
            scale[3, 3] = 1.0
            trans_vertex = scale @ trans_vertex
        asset.transform(trans_vertex)
        if asset.meta is not None and ('noise_std' in asset.meta or 'noise_rms' in asset.meta):
            scale_rms = np.sqrt(np.mean(np.diag(trans_vertex[:3, :3]).astype(np.float64) ** 2.0))
            if 'noise_std' in asset.meta:
                asset.meta['noise_std'] = np.float32(asset.meta['noise_std'] * scale_rms)
            if 'noise_rms' in asset.meta:
                asset.meta['noise_rms'] = np.float32(asset.meta['noise_rms'] * scale_rms)

@dataclass(frozen=True)
class AugmentPatch(Augment):
    
    patch_size: int
    
    num_patches: int
    
    train_cvm_network: bool

    compute_normals: bool=False

    normal_k: int=32

    compute_geometry: bool=False

    geometry_k: int=32

    mix_strategy: str="linear"

    score_mix_prob: float=0.0

    score_sigma_min: float=0.004

    score_sigma_max: float=0.024

    score_distribution: Union[str, List[str]]="mixed"
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        pc_noisy = asset.sampled_vertices_noisy
        
        assert pc is not None
        assert pc_noisy is not None
        
        N = pc_noisy.shape[0]
        
        seed_idx = np.random.permutation(N)[:self.num_patches]   # (P,)
        seed_points = pc_noisy[seed_idx]                         # (P, 3)
        
        tree = cKDTree(pc_noisy, compact_nodes=False, balanced_tree=False)
        nn_dist, nn_idx = tree.query(seed_points, k=self.patch_size)   # (P, M)
        nn_dist = nn_dist.astype(np.float32)
        nn_dist = nn_dist / (nn_dist[:, -1:] + 1e-8)

        pat_A = pc_noisy[nn_idx]  # (P, M, 3)
        pat_B = pc[nn_idx]        # (P, M, 3)
        pat_normals = None
        if self.compute_normals:
            pat_normals = np.stack(
                [estimate_patch_normals(pat_B[i], k=self.normal_k) for i in range(self.num_patches)],
                axis=0,
            )

        l1, l2 = 1e-8, 1.0
        t = np.random.rand(self.num_patches, self.patch_size, 1)
        t = (l2 - l1) * t + l1

        pat_linear = t * pat_B + (1 - t) * pat_A
        seed_points_linear = (
            t[:, 0:1, :] * pc[seed_idx][:, None, :] +
            (1 - t[:, 0:1, :]) * pc_noisy[seed_idx][:, None, :]
        )
        score_mask = np.zeros((self.num_patches, 1, 1), dtype=np.float32)
        score_sigma_full = None
        if self.mix_strategy not in ("linear", "score", "hybrid"):
            raise ValueError(f"unsupported patch mix_strategy: {self.mix_strategy}")
        if self.mix_strategy in ("score", "hybrid"):
            if self.mix_strategy == "score":
                score_mask[...] = 1.0
            else:
                score_mask = (np.random.rand(self.num_patches, 1, 1) < self.score_mix_prob).astype(np.float32)
            sigma = np.random.uniform(self.score_sigma_min, self.score_sigma_max, size=(self.num_patches, 1, 1))
            score_sigma_full = np.broadcast_to(sigma, (self.num_patches, self.patch_size, 1)).astype(np.float32)
            distributions = self.score_distribution
            if isinstance(distributions, str):
                if distributions == "mixed":
                    distributions = ["gaussian", "laplace", "uniform"]
                else:
                    distributions = [distributions]
            distribution = np.random.choice(distributions)
            if distribution == "gaussian":
                score_noise = np.random.normal(0, sigma, size=pat_B.shape)
            elif distribution == "laplace":
                score_noise = np.random.laplace(0, sigma, size=pat_B.shape)
            elif distribution == "uniform":
                bound = np.sqrt(3.0) * sigma
                score_noise = np.random.uniform(-bound, bound, size=pat_B.shape)
            else:
                raise ValueError(f"unsupported score distribution: {distribution}")
            pat_score = pat_B + score_noise
            seed_points_score = pc[seed_idx][:, None, :]
            pat_t = np.where(score_mask > 0, pat_score, pat_linear)
            seed_points_t = np.where(score_mask > 0, seed_points_score, seed_points_linear)
            t = np.where(score_mask > 0, np.ones_like(t), t)
        else:
            pat_t = pat_linear
            seed_points_t = seed_points_linear
        
        pat_A = pat_A - seed_points_t
        pat_B = pat_B - seed_points_t
        pat_t = pat_t - seed_points_t
        
        if asset.meta is None:
            asset.meta = {}
        asset.meta['pc_noisy'] = pat_A
        asset.meta['pc_clean'] = pat_B
        asset.meta['pc_mix'] = pat_t
        asset.meta['pc_t'] = t.astype(np.float32)
        asset.meta['pc_score_mask'] = np.broadcast_to(score_mask, (self.num_patches, self.patch_size, 1)).astype(np.float32)
        asset.meta['pc_seed_dist'] = nn_dist[:, :, None]
        if pat_normals is not None:
            asset.meta['pc_normal'] = pat_normals
        if self.compute_geometry:
            asset.meta['pc_geom'] = np.stack(
                [estimate_patch_geometry(pat_t[i], k=self.geometry_k) for i in range(self.num_patches)],
                axis=0,
            )
        if 'noise_rms' in asset.meta:
            noise = np.full((self.num_patches, self.patch_size, 1), asset.meta['noise_rms'], dtype=np.float32)
            if score_sigma_full is not None:
                score_mask_full = asset.meta['pc_score_mask']
                noise = np.where(score_mask_full > 0, score_sigma_full, noise)
            asset.meta['pc_noise_std'] = noise


def estimate_patch_geometry(pc: np.ndarray, k: int) -> np.ndarray:
    out = pc.astype(np.float64, copy=False)
    tree = cKDTree(out, compact_nodes=False, balanced_tree=False)
    dists, idx = tree.query(out, k=min(k, len(out)))
    geom = np.empty((len(out), 4), dtype=np.float32)
    for i, ids in enumerate(idx):
        nb = out[ids]
        center = nb.mean(axis=0)
        centered = nb - center
        cov = centered.T @ centered / max(len(ids), 1)
        val, _ = np.linalg.eigh(cov)
        val = np.maximum(val, 1e-12)
        l0, l1, l2 = val
        total = l0 + l1 + l2
        curvature = l0 / total
        linearity = (l2 - l1) / l2
        planarity = (l1 - l0) / l2
        local_scale = float(dists[i, -1]) if np.ndim(dists) == 2 else 0.0
        geom[i] = (curvature, linearity, planarity, local_scale)
    return geom


def estimate_patch_normals(pc: np.ndarray, k: int) -> np.ndarray:
    k = min(k, pc.shape[0])
    tree = cKDTree(pc, compact_nodes=False, balanced_tree=False)
    _, nn_idx = tree.query(pc, k=k)
    normals = np.empty_like(pc, dtype=np.float32)
    for i, ids in enumerate(nn_idx):
        nb = pc[ids]
        center = nb.mean(axis=0)
        centered = nb - center
        cov = centered.T @ centered / max(len(ids), 1)
        _, vec = np.linalg.eigh(cov)
        normals[i] = vec[:, 0]
    return normals

def get_augments(*args) -> List[Augment]:
    MAP = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "add_noise": AugmentAddNoise,
        "linear": AugmentLinear,
        "patch": AugmentPatch,
    }
    MAP: Dict[str, type[Augment]]
    augments = []
    for (i, config) in enumerate(args):
        __target__ = config.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in augment of position {i}"
        c = deepcopy(config)
        del c['__target__']
        augments.append(MAP[__target__].parse(**c))
    return augments
