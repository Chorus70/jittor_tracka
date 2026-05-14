from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from .feature import Decoder, EdgePCTHybridFeatureExtraction, FeatureExtraction, PCTNeighborFeatureExtraction
from .spec import ModelSpec
from .vm import VelocityModule, farthest_point_sampling, knn_points
from ..data.asset import Asset
from ..data.augment import estimate_patch_geometry


def _safe_norm(x, dim=-1, keepdims=True, eps=1e-8):
    return jt.sqrt((x ** 2.0).sum(dim=dim, keepdims=keepdims) + eps)


def _softplus(x):
    return jt.log(jt.exp(-jt.abs(x)) + 1.0) + jt.maximum(x, 0.0)


class LowNoiseAdaptiveRefiner(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.num_train_points = cfg.get("num_train_points", 256)
        self.frame_knn = cfg.get("frame_knn", 24)
        self.local_radius_k = cfg.get("local_radius_k", 16)
        self.max_step_scale = cfg.get("max_step_scale", 1.2)
        self.huber_delta = cfg.get("huber_delta", 0.004)
        self.patch_cd_k = cfg.get("patch_cd_k", 1)
        self.repulsion_k = cfg.get("repulsion_k", 6)
        self.repulsion_h = cfg.get("repulsion_h", 0.01)
        self.base_blend = cfg.get("base_blend", 1.0)
        self.predict_patch_size = cfg.get("predict_patch_size", 1200)
        self.predict_seed_k = cfg.get("predict_seed_k", 5)
        self.predict_seed_k_alpha = cfg.get("predict_seed_k_alpha", 1)
        self.predict_noise_estimate_k = cfg.get("predict_noise_estimate_k", 24)
        self.predict_compute_geometry = cfg.get("predict_compute_geometry", False)
        self.base_ckpt = cfg.get("base_ckpt", None)
        self.base_model = None

        input_dim = 14
        encoder_type = cfg.get("encoder_type", "pct_neighbor")
        encoder_kwargs = dict(
            k=self.frame_knn,
            input_dim=input_dim,
            embedding_dim=cfg.get("feat_embedding_dim", 384),
            use_global_feature=cfg.get("use_global_feature", True),
            use_input_coords=cfg.get("use_input_coords", True),
            knn_coord_dim=3,
        )
        if encoder_type == "pct_neighbor":
            self.encoder = PCTNeighborFeatureExtraction(
                **encoder_kwargs,
                attention_channels=cfg.get("pct_attention_channels", 96),
                num_attention_layers=cfg.get("pct_num_attention_layers", 4),
                use_position_embedding=cfg.get("pct_use_position_embedding", True),
            )
        elif encoder_type == "edge_pct_hybrid":
            self.encoder = EdgePCTHybridFeatureExtraction(
                **encoder_kwargs,
                attention_channels=cfg.get("pct_attention_channels", 96),
                num_attention_layers=cfg.get("pct_num_attention_layers", 2),
                use_position_embedding=cfg.get("pct_use_position_embedding", True),
            )
        elif encoder_type == "edgeconv":
            self.encoder = FeatureExtraction(**encoder_kwargs)
        else:
            raise ValueError(f"unsupported encoder_type: {encoder_type}")

        hidden = cfg.get("decoder_hidden_dim", 128)
        self.dir_head = Decoder(self.encoder.embedding_dim, dim=3, out_dim=3, hidden_size=hidden)
        self.step_lin_1 = nn.Linear(self.encoder.embedding_dim, hidden)
        self.step_lin_2 = nn.Linear(hidden, 1)
        self.step_act = nn.ReLU()
        self.gate_lin_1 = nn.Linear(self.encoder.embedding_dim, cfg.get("gate_hidden_dim", hidden))
        self.gate_lin_2 = nn.Linear(cfg.get("gate_hidden_dim", hidden), 1)
        self.gate_act = nn.ReLU()

        if cfg.get("zero_init_step_gate", True):
            self.step_lin_2.weight.assign(jt.zeros_like(self.step_lin_2.weight))
            self.step_lin_2.bias.assign(jt.zeros_like(self.step_lin_2.bias))
            self.gate_lin_2.weight.assign(jt.zeros_like(self.gate_lin_2.weight))
            self.gate_lin_2.bias.assign(jt.zeros_like(self.gate_lin_2.bias))

        if self.base_ckpt is not None:
            base_cfg = cfg.get("base_model_config", None)
            if base_cfg is None:
                raise ValueError("base_ckpt requires base_model_config")
            if "__target__" in base_cfg:
                base_cfg = dict(base_cfg)
                base_cfg.pop("__target__", None)
            base_model = VelocityModule(base_cfg, self.transform_config)
            base_model.load(self.base_ckpt)
            base_model.eval()
            for p in base_model.parameters():
                p.stop_grad()
            object.__setattr__(self, "base_model", base_model)

    def _base_predict(self, pc_noisy):
        if self.base_model is None:
            return pc_noisy
        with jt.no_grad():
            pred = self.base_model._decode_direction(self.base_model._encoder_input(pc_noisy))
            scale = self.base_model._condition_scale(None)
            if scale is not None:
                pred = pred * scale
            pc_base = pc_noisy + pred
        pc_base.stop_grad()
        return pc_base

    def _local_radius(self, pc):
        k = min(self.local_radius_k + 1, pc.shape[1])
        if k <= 1:
            return jt.ones((pc.shape[0], pc.shape[1], 1)) * 0.001
        dist = ((pc.unsqueeze(2) - pc.unsqueeze(1)) ** 2.0).sum(dim=-1)
        dist_k, _ = jt.topk(dist, k=k, dim=-1, largest=False)
        return jt.sqrt(dist_k[:, :, -1:] + 1e-8)

    def _sigma_feature(self, pc_noisy, pc_noise_std):
        B, N, _ = pc_noisy.shape
        if pc_noise_std is not None:
            return pc_noise_std
        radius = self._local_radius(pc_noisy)
        denom = jt.mean(radius, dim=1, keepdims=True) + 1e-8
        return (radius / denom).broadcast((B, N, 1))

    def _geom_feature(self, pc_noisy, pc_geom):
        B, N, _ = pc_noisy.shape
        if pc_geom is not None:
            return pc_geom
        return jt.zeros((B, N, 4))

    def _build_input(self, pc_noisy, pc_base, pc_geom=None, pc_noise_std=None):
        base_disp = pc_base - pc_noisy
        geom = self._geom_feature(pc_noisy, pc_geom)
        sigma = self._sigma_feature(pc_noisy, pc_noise_std)
        return jt.concat([pc_noisy, pc_base, base_disp, geom, sigma], dim=-1)

    def refine(self, pc_noisy, pc_base=None, pc_geom=None, pc_noise_std=None):
        if pc_base is None:
            pc_base = self._base_predict(pc_noisy)
        B, N, _ = pc_noisy.shape
        feat = self.encoder(self._build_input(pc_noisy, pc_base, pc_geom, pc_noise_std))
        flat = feat.reshape(B * N, -1)
        direction = self.dir_head(c=flat, B=B, N=N).reshape(B, N, 3)
        direction = direction / _safe_norm(direction)
        step = self.step_lin_1(flat)
        step = self.step_act(step)
        step = _softplus(self.step_lin_2(step)).reshape(B, N, 1)
        max_step = self.max_step_scale * self._local_radius(pc_noisy)
        step = jt.minimum(step, max_step)
        gate = self.gate_lin_1(flat)
        gate = self.gate_act(gate)
        gate = jt.sigmoid(self.gate_lin_2(gate)).reshape(B, N, 1)
        residual_disp = gate * step * direction
        base_disp = pc_base - pc_noisy
        total_disp = self.base_blend * base_disp + residual_disp
        return {
            "pc_denoised": pc_noisy + total_disp,
            "disp": total_disp,
            "residual_disp": residual_disp,
            "gate": gate,
            "step": step,
            "max_step": max_step,
        }

    def _sample_idx(self, n):
        if self.num_train_points <= 0 or self.num_train_points >= n:
            return None
        return jt.array(np.random.permutation(n)[: self.num_train_points]).int32()

    def _point_huber(self, err):
        abs_err = jt.abs(err)
        quadratic = jt.minimum(abs_err, self.huber_delta)
        linear = abs_err - quadratic
        return (0.5 * quadratic ** 2.0 + self.huber_delta * linear).sum(dim=-1).mean()

    def _patch_cd_loss(self, pc_pred, pc_clean):
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        pred_to_clean = jt.min(dist, dim=2)
        clean_to_pred = jt.min(dist, dim=1)
        return pred_to_clean.mean() + clean_to_pred.mean()

    def _normal_tangent_losses(self, disp, target_disp, normal):
        if normal is None:
            z = jt.array(0.0)
            return z, z
        normal = normal / _safe_norm(normal)
        disp_n = (disp * normal).sum(dim=-1, keepdims=True) * normal
        target_n = (target_disp * normal).sum(dim=-1, keepdims=True) * normal
        normal_loss = ((disp_n - target_n) ** 2.0).sum(dim=-1).mean()
        tangent = disp - disp_n
        tangent_loss = (tangent ** 2.0).sum(dim=-1).mean()
        return normal_loss, tangent_loss

    def _repulsion_loss(self, pc_pred):
        k = min(self.repulsion_k + 1, pc_pred.shape[1])
        if k <= 1:
            return jt.array(0.0)
        dist = ((pc_pred.unsqueeze(2) - pc_pred.unsqueeze(1)) ** 2.0).sum(dim=-1)
        dist_k, _ = jt.topk(dist, k=k, dim=-1, largest=False)
        dist_k = dist_k[:, :, 1:]
        return jt.exp(-dist_k / max(self.repulsion_h ** 2, 1e-8)).mean()

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        pc_base = None
        if self.base_model is None:
            pc_base = pc_noisy
        pc_normal = batch.get("pc_normal", None)
        if pc_normal is not None:
            pc_normal = pc_normal.reshape(-1, patch_size, 3)
        pc_geom = batch.get("pc_geom", None)
        if pc_geom is not None:
            pc_geom = pc_geom.reshape(-1, patch_size, 4)
        pc_noise_std = batch.get("pc_noise_std", None)
        if pc_noise_std is not None:
            pc_noise_std = pc_noise_std.reshape(-1, patch_size, 1)

        idx = self._sample_idx(patch_size)
        if idx is not None:
            pc_noisy = pc_noisy[:, idx, :]
            pc_clean = pc_clean[:, idx, :]
            pc_base = None if pc_base is None else pc_base[:, idx, :]
            pc_normal = None if pc_normal is None else pc_normal[:, idx, :]
            pc_geom = None if pc_geom is None else pc_geom[:, idx, :]
            pc_noise_std = None if pc_noise_std is None else pc_noise_std[:, idx, :]

        if pc_base is None:
            pc_base = self._base_predict(pc_noisy)
        out = self.refine(pc_noisy, pc_base=pc_base, pc_geom=pc_geom, pc_noise_std=pc_noise_std)
        pc_pred = out["pc_denoised"]
        disp = out["disp"]
        residual_target = pc_clean - (pc_noisy + self.base_blend * (pc_base - pc_noisy))
        target_disp = pc_clean - pc_noisy
        normal_loss, tangent_loss = self._normal_tangent_losses(disp, target_disp, pc_normal)
        gate_target = jt.minimum(_safe_norm(residual_target) / (out["max_step"] + 1e-8), 1.0)
        return {
            "loss_point": self._point_huber(pc_pred - pc_clean),
            "loss_patch_cd": self._patch_cd_loss(pc_pred, pc_clean),
            "loss_surface": normal_loss,
            "loss_tangent": tangent_loss,
            "loss_gate": ((out["gate"] - gate_target) ** 2.0).mean(),
            "loss_repulsion": self._repulsion_loss(pc_pred),
            "loss_anchor": jt.maximum(_safe_norm(out["residual_disp"]) - out["max_step"], 0.0).mean(),
        }

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    def _predict_one(self, pc_noisy):
        N = pc_noisy.shape[0]
        if N <= self.predict_patch_size:
            return self.refine(pc_noisy.unsqueeze(0))["pc_denoised"].squeeze(0)
        num_patches = max(1, int(self.predict_seed_k * N / self.predict_patch_size))
        pc_batch = pc_noisy.unsqueeze(0)
        seed_pnts, _ = farthest_point_sampling(pc_batch, num_patches)
        patch_dists, point_idxs, patches = knn_points(seed_pnts, pc_batch, self.predict_patch_size)
        patches = patches[0]
        point_idxs = point_idxs[0]
        patch_dists = patch_dists[0]
        seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
        patches_centered = patches - seed_expand

        denoised = []
        i = 0
        patch_step = max(1, int(ceil(N / (self.predict_seed_k_alpha * self.predict_patch_size))))
        while i < num_patches:
            curr = patches_centered[i : i + patch_step]
            geom = None
            if self.predict_compute_geometry:
                geom_np = np.stack(
                    [estimate_patch_geometry(curr[j].detach().numpy(), k=self.predict_noise_estimate_k) for j in range(curr.shape[0])],
                    axis=0,
                ).astype(np.float32)
                geom = jt.array(geom_np)
            pred = self.refine(curr, pc_geom=geom)["pc_denoised"] + seed_expand[i : i + patch_step]
            denoised.append(pred)
            i += patch_step
        denoised = jt.concat(denoised, dim=0)

        patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
        weights = jt.exp(-4.0 * patch_dists)
        out = jt.zeros((N, 3))
        denom = jt.zeros((N, 1))
        for p in range(num_patches):
            idx = point_idxs[p]
            w = weights[p].unsqueeze(-1)
            out[idx] += denoised[p] * w
            denom[idx] += w
        return out / (denom + 1e-8)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        res = []
        for pc_noisy in batch["pc_noisy"]:
            res.append({"pc_denoised": self._predict_one(pc_noisy).detach().numpy()})
        return res

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append(
                    {
                        "pc_noisy": b.meta["pc_noisy"],
                        "pc_clean": b.meta["pc_clean"],
                        "pc_mix": b.meta["pc_mix"],
                        **({"pc_normal": b.meta["pc_normal"]} if "pc_normal" in b.meta else {}),
                        **({"pc_noise_std": b.meta["pc_noise_std"]} if "pc_noise_std" in b.meta else {}),
                        **({"pc_geom": b.meta["pc_geom"]} if "pc_geom" in b.meta else {}),
                    }
                )
            else:
                res.append({"pc_noisy": b.sampled_vertices_noisy})
        return res


class AlphaGateRefiner(LowNoiseAdaptiveRefiner):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.alpha_min = cfg.get("alpha_min", 0.0)
        self.alpha_max = cfg.get("alpha_max", 1.5)
        self.alpha_target_max = cfg.get("alpha_target_max", 1.5)
        self.alpha_loss_weight = cfg.get("alpha_loss_weight", 1.0)

    def refine(self, pc_noisy, pc_base=None, pc_geom=None, pc_noise_std=None):
        if pc_base is None:
            pc_base = self._base_predict(pc_noisy)
        B, N, _ = pc_noisy.shape
        feat = self.encoder(self._build_input(pc_noisy, pc_base, pc_geom, pc_noise_std))
        flat = feat.reshape(B * N, -1)
        gate = self.gate_lin_1(flat)
        gate = self.gate_act(gate)
        alpha01 = jt.sigmoid(self.gate_lin_2(gate)).reshape(B, N, 1)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha01
        base_disp = pc_base - pc_noisy
        total_disp = alpha * base_disp
        return {
            "pc_denoised": pc_noisy + total_disp,
            "disp": total_disp,
            "base_disp": base_disp,
            "alpha": alpha,
            "gate": alpha01,
        }

    def _alpha_target(self, pc_noisy, pc_base, pc_clean):
        base_disp = pc_base - pc_noisy
        target_disp = pc_clean - pc_noisy
        denom = (base_disp ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
        alpha = (target_disp * base_disp).sum(dim=-1, keepdims=True) / denom
        return jt.minimum(jt.maximum(alpha, self.alpha_min), self.alpha_target_max)

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        pc_normal = batch.get("pc_normal", None)
        if pc_normal is not None:
            pc_normal = pc_normal.reshape(-1, patch_size, 3)
        pc_geom = batch.get("pc_geom", None)
        if pc_geom is not None:
            pc_geom = pc_geom.reshape(-1, patch_size, 4)
        pc_noise_std = batch.get("pc_noise_std", None)
        if pc_noise_std is not None:
            pc_noise_std = pc_noise_std.reshape(-1, patch_size, 1)

        idx = self._sample_idx(patch_size)
        if idx is not None:
            pc_noisy = pc_noisy[:, idx, :]
            pc_clean = pc_clean[:, idx, :]
            pc_normal = None if pc_normal is None else pc_normal[:, idx, :]
            pc_geom = None if pc_geom is None else pc_geom[:, idx, :]
            pc_noise_std = None if pc_noise_std is None else pc_noise_std[:, idx, :]

        pc_base = self._base_predict(pc_noisy)
        out = self.refine(pc_noisy, pc_base=pc_base, pc_geom=pc_geom, pc_noise_std=pc_noise_std)
        pc_pred = out["pc_denoised"]
        disp = out["disp"]
        target_disp = pc_clean - pc_noisy
        alpha_target = self._alpha_target(pc_noisy, pc_base, pc_clean)
        normal_loss, tangent_loss = self._normal_tangent_losses(disp, target_disp, pc_normal)
        return {
            "loss_point": self._point_huber(pc_pred - pc_clean),
            "loss_patch_cd": self._patch_cd_loss(pc_pred, pc_clean),
            "loss_surface": normal_loss,
            "loss_tangent": tangent_loss,
            "loss_alpha": ((out["alpha"] - alpha_target) ** 2.0).mean(),
            "loss_repulsion": self._repulsion_loss(pc_pred),
        }
