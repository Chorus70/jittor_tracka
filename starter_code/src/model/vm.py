from math import ceil
from copy import deepcopy
from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from .feature import EdgePCTHybridFeatureExtraction, EdgePCTResidualFeatureExtraction, FeatureExtraction, PCTDenoiseFeatureExtraction, PCTFeatureExtraction, PCTHierarchicalFeatureExtraction, PCTNeighborFeatureExtraction, PCTRefineLocalFeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset
from ..data.augment import estimate_patch_geometry

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

def get_center_focused_indices(n, m, center_count):
    assert m < n
    center_count = min(max(center_count, 0), m)
    if center_count == 0:
        return get_random_indices(n, m)
    prefix = np.arange(center_count, dtype=np.int32)
    rest_pool = np.arange(center_count, n, dtype=np.int32)
    rest_count = m - center_count
    if rest_count > 0:
        rest = np.random.permutation(rest_pool)[:rest_count]
        idx = np.concatenate([prefix, rest], axis=0)
        idx = np.random.permutation(idx)
    else:
        idx = prefix
    return jt.array(idx).int32()

class VelocityModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        self.train_center_focus_points = cfg.get('train_center_focus_points', 0)
        
        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']
        self.target_mode = cfg.get('target_mode', 'clean_minus_noisy')
        self.loss_type = cfg.get('loss_type', 'mse')
        self.huber_delta = cfg.get('huber_delta', 0.01)
        self.smooth_weight = cfg.get('smooth_weight', 0.0)
        self.smooth_k = cfg.get('smooth_k', 8)
        self.normal_loss_weight = cfg.get('normal_loss_weight', 0.0)
        self.tangent_damping_weight = cfg.get('tangent_damping_weight', 0.0)
        self.target_tangent_weight = cfg.get('target_tangent_weight', 1.0)
        self.surface_loss_weight = cfg.get('surface_loss_weight', 0.0)
        self.surface_planarity_power = cfg.get('surface_planarity_power', 0.0)
        self.surface_curvature_suppress = cfg.get('surface_curvature_suppress', 0.0)
        self.score_target_blend = cfg.get('score_target_blend', 0.0)
        self.loss_center_gamma = cfg.get('loss_center_gamma', 0.0)
        self.predict_num_steps = cfg.get('predict_num_steps', 1)
        self.predict_dynamics = cfg.get('predict_dynamics', 'euler')
        self.predict_sigma_min = cfg.get('predict_sigma_min', cfg.get('noise_condition_min', 0.004))
        self.predict_sigma_max = cfg.get('predict_sigma_max', cfg.get('noise_condition_max', 0.024))
        self.predict_score_step_scale = cfg.get('predict_score_step_scale', 0.6)
        self.predict_score_momentum = cfg.get('predict_score_momentum', 0.0)
        self.predict_patch_size = cfg.get('predict_patch_size', 1000)
        self.predict_seed_k = cfg.get('predict_seed_k', 6)
        self.predict_seed_k_alpha = cfg.get('predict_seed_k_alpha', 1)
        self.predict_patch_aggregation = cfg.get('predict_patch_aggregation', 'best')
        self.predict_patch_weight_gamma = cfg.get('predict_patch_weight_gamma', 4.0)
        self.predict_disp_confidence_gamma = cfg.get('predict_disp_confidence_gamma', 0.0)
        self.predict_disp_confidence_floor = cfg.get('predict_disp_confidence_floor', 0.0)
        self.predict_disp_max_scale = cfg.get('predict_disp_max_scale', 4.0)
        self.use_noise_condition = cfg.get('use_noise_condition', False)
        self.noise_condition_ref = cfg.get('noise_condition_ref', 0.0125)
        self.noise_condition_min = cfg.get('noise_condition_min', 0.004)
        self.noise_condition_max = cfg.get('noise_condition_max', 0.024)
        self.predict_noise_std = cfg.get('predict_noise_std', None)
        self.predict_noise_estimate_k = cfg.get('predict_noise_estimate_k', 24)
        self.predict_noise_estimate_samples = cfg.get('predict_noise_estimate_samples', 8192)
        self.use_condition_input = cfg.get('use_condition_input', False)
        self.use_latent_condition = cfg.get('use_latent_condition', False)
        self.latent_condition_hidden_dim = cfg.get('latent_condition_hidden_dim', 32)
        self.use_geometry_input = cfg.get('use_geometry_input', False)
        self.geometry_dim = cfg.get('geometry_dim', 4)
        self.multi_scale_knns = cfg.get('multi_scale_knns', [])
        self.multi_scale_weight = cfg.get('multi_scale_weight', 0.15)
        self.repulsion_weight = cfg.get('repulsion_weight', 0.0)
        self.repulsion_k = cfg.get('repulsion_k', 6)
        self.repulsion_h = cfg.get('repulsion_h', 0.018)
        self.use_reliability = cfg.get('use_reliability', False)
        self.reliability_hidden_dim = cfg.get('reliability_hidden_dim', 128)
        self.reliability_loss_weight = cfg.get('reliability_loss_weight', 0.0)
        self.reliability_sigma_scale = cfg.get('reliability_sigma_scale', 2.0)
        self.unreliable_anchor_weight = cfg.get('unreliable_anchor_weight', 0.0)
        self.manifold_loss_weight = cfg.get('manifold_loss_weight', 0.0)
        self.manifold_cover_weight = cfg.get('manifold_cover_weight', 0.0)
        self.manifold_k = cfg.get('manifold_k', 8)
        self.manifold_edge_low = cfg.get('manifold_edge_low', 0.18)
        self.manifold_edge_high = cfg.get('manifold_edge_high', 0.55)
        self.manifold_curvature_suppress = cfg.get('manifold_curvature_suppress', 20.0)
        self.teacher_ckpt = cfg.get('teacher_ckpt', None)
        self.teacher_distill_weight = cfg.get('teacher_distill_weight', 0.0)
        self.teacher_distill_score_weight = cfg.get('teacher_distill_score_weight', 1.0)
        self.teacher_model = None
        self.zero_init_decoder = cfg.get('zero_init_decoder', False)
        self.freeze_edge_pct_residual_base = cfg.get('freeze_edge_pct_residual_base', False)
        self.refine_ckpt = cfg.get('refine_ckpt', None)
        self.refine_step_scale = cfg.get('refine_step_scale', 1.0)
        self.refine_scale_consistent_loss = cfg.get('refine_scale_consistent_loss', False)
        self.refine_loss_type = cfg.get('refine_loss_type', self.loss_type)
        self.refine_dsm_sigma = cfg.get('refine_dsm_sigma', self.dsm_sigma)
        self.refine_huber_delta = cfg.get('refine_huber_delta', self.huber_delta)
        self.refine_zero_init = cfg.get('refine_zero_init', True)
        self.refine_use_gate = cfg.get('refine_use_gate', False)
        self.refine_gate_weight = cfg.get('refine_gate_weight', 0.0)
        self.refine_corr_anchor_weight = cfg.get('refine_corr_anchor_weight', 0.0)
        self.refine_surface_loss_weight = cfg.get('refine_surface_loss_weight', 0.0)
        self.refine_target_tangent_weight = cfg.get('refine_target_tangent_weight', 1.0)
        self.refine_tangent_damping_weight = cfg.get('refine_tangent_damping_weight', 0.0)
        self.refine_normal_loss_weight = cfg.get('refine_normal_loss_weight', 0.0)
        self.refine_gate_tau = cfg.get('refine_gate_tau', 0.01)
        self.refine_gate_pointwise = cfg.get('refine_gate_pointwise', False)
        self.refine_gate_target_mode = cfg.get('refine_gate_target_mode', 'target_norm')
        self.refine_gate_noisy_margin = cfg.get('refine_gate_noisy_margin', 0.9)
        self.use_refiner = self.refine_ckpt is not None
        
        # networks
        encoder_input_dim = 3
        if self.use_geometry_input:
            encoder_input_dim += self.geometry_dim
        if self.use_condition_input:
            encoder_input_dim += 2
        def build_encoder(encoder_type, k, input_dim, embedding_dim, prefix='', default_attention_layers=4):
            if encoder_type == 'pct':
                return PCTFeatureExtraction(
                    input_dim=input_dim,
                    embedding_dim=embedding_dim,
                    attention_channels=cfg.get(f'{prefix}pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get(f'{prefix}pct_num_attention_layers', cfg.get('pct_num_attention_layers', default_attention_layers)),
                    use_global_feature=cfg.get(f'{prefix}use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=cfg.get(f'{prefix}use_input_coords', cfg.get('use_input_coords', False)),
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get(f'{prefix}pct_use_position_embedding', cfg.get('pct_use_position_embedding', False)),
                )
            elif encoder_type == 'pct_neighbor':
                return PCTNeighborFeatureExtraction(
                    k=k,
                    input_dim=input_dim,
                    embedding_dim=embedding_dim,
                    attention_channels=cfg.get(f'{prefix}pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get(f'{prefix}pct_num_attention_layers', cfg.get('pct_num_attention_layers', default_attention_layers)),
                    use_global_feature=cfg.get(f'{prefix}use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=cfg.get(f'{prefix}use_input_coords', cfg.get('use_input_coords', False)),
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get(f'{prefix}pct_use_position_embedding', cfg.get('pct_use_position_embedding', False)),
                )
            elif encoder_type == 'pct_denoise':
                return PCTDenoiseFeatureExtraction(
                    k=k,
                    input_dim=input_dim,
                    embedding_dim=embedding_dim,
                    attention_channels=cfg.get(f'{prefix}pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get(f'{prefix}pct_num_attention_layers', cfg.get('pct_num_attention_layers', default_attention_layers)),
                    use_global_feature=cfg.get(f'{prefix}use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=cfg.get(f'{prefix}use_input_coords', cfg.get('use_input_coords', False)),
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get(f'{prefix}pct_use_position_embedding', cfg.get('pct_use_position_embedding', True)),
                )
            elif encoder_type == 'pct_hier':
                return PCTHierarchicalFeatureExtraction(
                    k=k,
                    input_dim=input_dim,
                    embedding_dim=embedding_dim,
                    attention_channels=cfg.get(f'{prefix}pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get(f'{prefix}pct_num_attention_layers', cfg.get('pct_num_attention_layers', default_attention_layers)),
                    num_tokens=cfg.get(f'{prefix}pct_num_tokens', cfg.get('pct_num_tokens', 128)),
                    interp_k=cfg.get(f'{prefix}pct_interp_k', cfg.get('pct_interp_k', 3)),
                    use_global_feature=cfg.get(f'{prefix}use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=cfg.get(f'{prefix}use_input_coords', cfg.get('use_input_coords', False)),
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get(f'{prefix}pct_use_position_embedding', cfg.get('pct_use_position_embedding', True)),
                )
            elif encoder_type == 'edge_pct_hybrid':
                return EdgePCTHybridFeatureExtraction(
                    k=k,
                    input_dim=input_dim,
                    embedding_dim=embedding_dim,
                    attention_channels=cfg.get(f'{prefix}pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get(f'{prefix}pct_num_attention_layers', cfg.get('pct_num_attention_layers', 2)),
                    use_global_feature=cfg.get(f'{prefix}use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=cfg.get(f'{prefix}use_input_coords', cfg.get('use_input_coords', False)),
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get(f'{prefix}pct_use_position_embedding', cfg.get('pct_use_position_embedding', True)),
                )
            elif encoder_type == 'edge_pct_residual':
                return EdgePCTResidualFeatureExtraction(
                    k=k,
                    input_dim=input_dim,
                    embedding_dim=embedding_dim,
                    num_attention_layers=cfg.get(f'{prefix}pct_num_attention_layers', cfg.get('pct_num_attention_layers', 1)),
                    use_global_feature=cfg.get(f'{prefix}use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=cfg.get(f'{prefix}use_input_coords', cfg.get('use_input_coords', False)),
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get(f'{prefix}pct_use_position_embedding', cfg.get('pct_use_position_embedding', True)),
                )
            elif encoder_type == 'edgeconv':
                return FeatureExtraction(
                    k=k,
                    input_dim=input_dim,
                    embedding_dim=embedding_dim,
                    use_global_feature=cfg.get(f'{prefix}use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=cfg.get(f'{prefix}use_input_coords', cfg.get('use_input_coords', False)),
                    knn_coord_dim=3,
                )
            else:
                raise ValueError(f"unsupported encoder_type: {encoder_type}")

        encoder_type = cfg.get('encoder_type', 'edgeconv')
        self.encoder = build_encoder(
            encoder_type=encoder_type,
            k=self.frame_knn,
            input_dim=encoder_input_dim,
            embedding_dim=cfg['feat_embedding_dim'],
        )
        
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )
        if self.zero_init_decoder:
            self.decoder.lin_3.weight.assign(jt.zeros_like(self.decoder.lin_3.weight))
            self.decoder.lin_3.bias.assign(jt.zeros_like(self.decoder.lin_3.bias))
        if self.use_latent_condition:
            self.condition_lin_1 = nn.Linear(2, self.latent_condition_hidden_dim)
            self.condition_lin_2 = nn.Linear(self.latent_condition_hidden_dim, self.encoder.embedding_dim)
            self.condition_act = nn.ReLU()
            self.condition_lin_2.weight.assign(jt.zeros_like(self.condition_lin_2.weight))
            self.condition_lin_2.bias.assign(jt.zeros_like(self.condition_lin_2.bias))
        if self.use_reliability:
            self.reliability_lin_1 = nn.Linear(self.encoder.embedding_dim, self.reliability_hidden_dim)
            self.reliability_lin_2 = nn.Linear(self.reliability_hidden_dim, 1)
            self.reliability_act = nn.ReLU()

        for i, k in enumerate(self.multi_scale_knns):
            encoder = build_encoder(
                encoder_type=cfg.get('multi_scale_encoder_type', 'edgeconv'),
                k=k,
                input_dim=encoder_input_dim,
                embedding_dim=cfg.get('multi_scale_embedding_dim', cfg['feat_embedding_dim']),
                prefix='multi_scale_',
                default_attention_layers=cfg.get('pct_num_attention_layers', 2),
            )
            decoder = Decoder(
                z_dim=encoder.embedding_dim,
                dim=3,
                out_dim=3,
                hidden_size=cfg['decoder_hidden_dim'],
            )
            setattr(self, f"multi_encoder_{i}", encoder)
            setattr(self, f"multi_decoder_{i}", decoder)

        if self.use_refiner:
            refine_encoder_type = cfg.get('refine_encoder_type', 'edgeconv')
            if refine_encoder_type == 'pct':
                self.refine_encoder = PCTFeatureExtraction(
                    input_dim=9,
                    embedding_dim=cfg.get('refine_feat_embedding_dim', cfg['feat_embedding_dim']),
                    attention_channels=cfg.get('refine_pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get('refine_pct_num_attention_layers', cfg.get('pct_num_attention_layers', 4)),
                    use_global_feature=cfg.get('refine_use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=True,
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get('refine_pct_use_position_embedding', cfg.get('pct_use_position_embedding', False)),
                )
            elif refine_encoder_type == 'pct_neighbor':
                self.refine_encoder = PCTNeighborFeatureExtraction(
                    k=cfg.get('refine_frame_knn', self.frame_knn),
                    input_dim=9,
                    embedding_dim=cfg.get('refine_feat_embedding_dim', cfg['feat_embedding_dim']),
                    attention_channels=cfg.get('refine_pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get('refine_pct_num_attention_layers', cfg.get('pct_num_attention_layers', 4)),
                    use_global_feature=cfg.get('refine_use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=True,
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get('refine_pct_use_position_embedding', cfg.get('pct_use_position_embedding', False)),
                )
            elif refine_encoder_type == 'pct_denoise':
                self.refine_encoder = PCTDenoiseFeatureExtraction(
                    k=cfg.get('refine_frame_knn', self.frame_knn),
                    input_dim=9,
                    embedding_dim=cfg.get('refine_feat_embedding_dim', cfg['feat_embedding_dim']),
                    attention_channels=cfg.get('refine_pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get('refine_pct_num_attention_layers', cfg.get('pct_num_attention_layers', 4)),
                    use_global_feature=cfg.get('refine_use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=True,
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get('refine_pct_use_position_embedding', cfg.get('pct_use_position_embedding', True)),
                )
            elif refine_encoder_type == 'pct_hier':
                self.refine_encoder = PCTHierarchicalFeatureExtraction(
                    k=cfg.get('refine_frame_knn', self.frame_knn),
                    input_dim=9,
                    embedding_dim=cfg.get('refine_feat_embedding_dim', cfg['feat_embedding_dim']),
                    attention_channels=cfg.get('refine_pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get('refine_pct_num_attention_layers', cfg.get('pct_num_attention_layers', 4)),
                    num_tokens=cfg.get('refine_pct_num_tokens', cfg.get('pct_num_tokens', 128)),
                    interp_k=cfg.get('refine_pct_interp_k', cfg.get('pct_interp_k', 3)),
                    use_global_feature=cfg.get('refine_use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=True,
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get('refine_pct_use_position_embedding', cfg.get('pct_use_position_embedding', True)),
                )
            elif refine_encoder_type == 'pct_refine_local':
                self.refine_encoder = PCTRefineLocalFeatureExtraction(
                    k=cfg.get('refine_frame_knn', self.frame_knn),
                    input_dim=9,
                    embedding_dim=cfg.get('refine_feat_embedding_dim', cfg['feat_embedding_dim']),
                    attention_channels=cfg.get('refine_pct_attention_channels', cfg.get('pct_attention_channels', 96)),
                    num_attention_layers=cfg.get('refine_pct_num_attention_layers', cfg.get('pct_num_attention_layers', 2)),
                    use_global_feature=cfg.get('refine_use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=True,
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get('refine_pct_use_position_embedding', cfg.get('pct_use_position_embedding', True)),
                )
            elif refine_encoder_type == 'edge_pct_hybrid':
                self.refine_encoder = EdgePCTHybridFeatureExtraction(
                    k=cfg.get('refine_frame_knn', self.frame_knn),
                    input_dim=9,
                    embedding_dim=cfg.get('refine_feat_embedding_dim', cfg['feat_embedding_dim']),
                    attention_channels=cfg.get('refine_pct_attention_channels', cfg.get('pct_attention_channels', 128)),
                    num_attention_layers=cfg.get('refine_pct_num_attention_layers', cfg.get('pct_num_attention_layers', 2)),
                    use_global_feature=cfg.get('refine_use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=True,
                    knn_coord_dim=3,
                    use_position_embedding=cfg.get('refine_pct_use_position_embedding', cfg.get('pct_use_position_embedding', True)),
                )
            elif refine_encoder_type == 'edgeconv':
                self.refine_encoder = FeatureExtraction(
                    k=cfg.get('refine_frame_knn', self.frame_knn),
                    input_dim=9,
                    embedding_dim=cfg.get('refine_feat_embedding_dim', cfg['feat_embedding_dim']),
                    use_global_feature=cfg.get('refine_use_global_feature', cfg.get('use_global_feature', False)),
                    use_input_coords=True,
                    knn_coord_dim=3,
                )
            else:
                raise ValueError(f"unsupported refine_encoder_type: {refine_encoder_type}")
            self.refine_decoder = Decoder(
                z_dim=self.refine_encoder.embedding_dim,
                dim=3,
                out_dim=3,
                hidden_size=cfg.get('refine_decoder_hidden_dim', cfg['decoder_hidden_dim']),
            )
            if self.refine_zero_init:
                self.refine_decoder.lin_3.weight.assign(jt.zeros_like(self.refine_decoder.lin_3.weight))
                self.refine_decoder.lin_3.bias.assign(jt.zeros_like(self.refine_decoder.lin_3.bias))
            if self.refine_use_gate:
                if self.refine_gate_pointwise:
                    gate_hidden = cfg.get('refine_gate_hidden_dim', cfg.get('refine_decoder_hidden_dim', cfg['decoder_hidden_dim']))
                    self.refine_gate_lin_1 = nn.Linear(self.refine_encoder.embedding_dim, gate_hidden)
                    self.refine_gate_lin_2 = nn.Linear(gate_hidden, 1)
                    self.refine_gate_act = nn.ReLU()
                else:
                    self.refine_gate_decoder = Decoder(
                        z_dim=self.refine_encoder.embedding_dim,
                        dim=3,
                        out_dim=1,
                        hidden_size=cfg.get('refine_gate_hidden_dim', cfg.get('refine_decoder_hidden_dim', cfg['decoder_hidden_dim'])),
                    )
                if self.refine_zero_init:
                    if self.refine_gate_pointwise:
                        self.refine_gate_lin_2.weight.assign(jt.zeros_like(self.refine_gate_lin_2.weight))
                        self.refine_gate_lin_2.bias.assign(jt.zeros_like(self.refine_gate_lin_2.bias))
                    else:
                        self.refine_gate_decoder.lin_3.weight.assign(jt.zeros_like(self.refine_gate_decoder.lin_3.weight))
                        self.refine_gate_decoder.lin_3.bias.assign(jt.zeros_like(self.refine_gate_decoder.lin_3.bias))
            base_cfg = deepcopy(cfg)
            for key in (
                'refine_ckpt',
                'refine_step_scale',
                'refine_scale_consistent_loss',
                'refine_loss_type',
                'refine_dsm_sigma',
                'refine_huber_delta',
                'refine_zero_init',
                'refine_use_gate',
                'refine_gate_weight',
                'refine_corr_anchor_weight',
                'refine_target_tangent_weight',
                'refine_tangent_damping_weight',
                'refine_normal_loss_weight',
                'refine_gate_tau',
                'refine_gate_hidden_dim',
                'refine_frame_knn',
                'refine_feat_embedding_dim',
                'refine_use_global_feature',
                'refine_decoder_hidden_dim',
                'teacher_ckpt',
                'teacher_distill_weight',
                'teacher_distill_score_weight',
            ):
                base_cfg.pop(key, None)
            base_model = VelocityModule(base_cfg, self.transform_config)
            base_model.load(self.refine_ckpt)
            base_model.eval()
            object.__setattr__(self, 'base_model', base_model)
            for module in [self.encoder, self.decoder]:
                for p in module.parameters():
                    p.stop_grad()
            for i in range(len(self.multi_scale_knns)):
                for p in getattr(self, f"multi_encoder_{i}").parameters():
                    p.stop_grad()
                for p in getattr(self, f"multi_decoder_{i}").parameters():
                    p.stop_grad()

        if self.teacher_ckpt is not None and self.teacher_distill_weight > 0:
            teacher_cfg = deepcopy(cfg)
            for key in (
                'teacher_ckpt',
                'teacher_distill_weight',
                'teacher_distill_score_weight',
            ):
                teacher_cfg.pop(key, None)
            teacher = VelocityModule(teacher_cfg, self.transform_config)
            teacher.load(self.teacher_ckpt)
            teacher.eval()
            # Keep the teacher out of this module's trainable parameter tree.
            object.__setattr__(self, 'teacher_model', teacher)

        if self.freeze_edge_pct_residual_base:
            if not isinstance(self.encoder, EdgePCTResidualFeatureExtraction):
                raise ValueError("freeze_edge_pct_residual_base requires encoder_type=edge_pct_residual")
            for module in [self.encoder.conv1, self.encoder.conv2, self.encoder.conv3, self.decoder]:
                for p in module.parameters():
                    p.stop_grad()
            for i in range(len(self.multi_scale_knns)):
                for p in getattr(self, f"multi_encoder_{i}").parameters():
                    p.stop_grad()
                for p in getattr(self, f"multi_decoder_{i}").parameters():
                    p.stop_grad()

    def _predict_direction_from_model(
        self,
        model,
        pc,
        pnt_idx=None,
        pc_t=None,
        pc_noise_std=None,
        pc_geom=None,
    ):
        pred = model._decode_direction(
            model._encoder_input(
                pc,
                pc_t=pc_t,
                pc_noise_std=pc_noise_std,
                pc_geom=pc_geom,
            ),
            pnt_idx=pnt_idx,
            pc_t=pc_t,
            pc_noise_std=pc_noise_std,
        )
        if pc_noise_std is not None and pnt_idx is not None:
            pc_noise_std = pc_noise_std[:, pnt_idx, :]
        scale = model._condition_scale(pc_noise_std)
        if scale is not None:
            pred = pred * scale
        return pred

    def _decode_refine_direction(self, refiner_input, pnt_idx=None):
        feat = self.refine_encoder(refiner_input)
        if pnt_idx is not None:
            feat = feat[:, pnt_idx, :]
        B, N, F_dim = feat.shape
        corr = self.refine_decoder(c=feat.reshape(-1, F_dim)).reshape(B, N, 3)
        gate = None
        if self.refine_use_gate:
            if self.refine_gate_pointwise:
                gate = self.refine_gate_lin_1(feat.reshape(-1, F_dim))
                gate = self.refine_gate_act(gate)
                gate = self.refine_gate_lin_2(gate)
                gate = jt.sigmoid(gate).reshape(B, N, 1)
            else:
                gate = self.refine_gate_decoder(c=feat.reshape(-1, F_dim), B=B, N=N)
        return corr, gate
    
    def _condition_scale(self, pc_noise_std):
        if pc_noise_std is None or not self.use_noise_condition:
            return None
        scale = pc_noise_std / self.noise_condition_ref
        scale = jt.maximum(scale, self.noise_condition_min / self.noise_condition_ref)
        scale = jt.minimum(scale, self.noise_condition_max / self.noise_condition_ref)
        return scale

    def _encoder_input(self, pc, pc_t=None, pc_noise_std=None, pc_geom=None):
        if not self.use_condition_input and not self.use_geometry_input:
            return pc
        B, N, _ = pc.shape
        features = [pc]
        if self.use_geometry_input:
            if pc_geom is None:
                pc_geom = jt.zeros((B, N, self.geometry_dim))
            features.append(pc_geom)
        if pc_t is None:
            pc_t = jt.zeros((B, N, 1))
        if pc_noise_std is None:
            pc_noise_std = jt.ones((B, N, 1)) * self.noise_condition_ref
        if self.use_condition_input:
            noise = pc_noise_std / self.noise_condition_ref
            features.extend([pc_t, noise])
        return jt.concat(features, dim=-1)

    def _point_mean(self, value, weight=None):
        if weight is None:
            return value.mean()
        while len(weight.shape) < len(value.shape):
            weight = weight.unsqueeze(-1)
        weight = weight.broadcast(value.shape)
        return (value * weight).sum() / (weight.sum() + 1e-8)

    def _condition_bias(self, pc_t, pc_noise_std, B, N, pnt_idx=None):
        if not self.use_latent_condition:
            return None
        if pc_t is None:
            pc_t = jt.zeros((B, N, 1))
        if pc_noise_std is None:
            pc_noise_std = jt.ones((B, N, 1)) * self.noise_condition_ref
        if pnt_idx is not None:
            pc_t = pc_t[:, pnt_idx, :]
            pc_noise_std = pc_noise_std[:, pnt_idx, :]
        cond = jt.concat([pc_t, pc_noise_std / self.noise_condition_ref], dim=-1)
        cond = self.condition_lin_1(cond.reshape(-1, 2))
        cond = self.condition_act(cond)
        cond = self.condition_lin_2(cond)
        return cond.reshape(cond.shape[0] // pc_t.shape[1], pc_t.shape[1], -1)

    def _decode_direction(self, encoder_input, pnt_idx=None, pc_t=None, pc_noise_std=None):
        feat = self.encoder(encoder_input)
        if pnt_idx is not None:
            feat = feat[:, pnt_idx, :]
        B, N, F_dim = feat.shape
        cond_bias = self._condition_bias(pc_t, pc_noise_std, encoder_input.shape[0], encoder_input.shape[1], pnt_idx=pnt_idx)
        if cond_bias is not None:
            feat = feat + cond_bias
        pred = self.decoder(c=feat.reshape(-1, F_dim)).reshape(B, N, 3)

        if self.multi_scale_knns:
            for i in range(len(self.multi_scale_knns)):
                encoder = getattr(self, f"multi_encoder_{i}")
                decoder = getattr(self, f"multi_decoder_{i}")
                ms_feat = encoder(encoder_input)
                if pnt_idx is not None:
                    ms_feat = ms_feat[:, pnt_idx, :]
                _, _, ms_dim = ms_feat.shape
                ms_pred = decoder(c=ms_feat.reshape(-1, ms_dim)).reshape(B, N, 3)
                pred = pred + self.multi_scale_weight * ms_pred
        return pred

    def _decode_reliability(self, encoder_input, pnt_idx=None, pc_t=None, pc_noise_std=None):
        if not self.use_reliability:
            return None
        feat = self.encoder(encoder_input)
        if pnt_idx is not None:
            feat = feat[:, pnt_idx, :]
        cond_bias = self._condition_bias(pc_t, pc_noise_std, encoder_input.shape[0], encoder_input.shape[1], pnt_idx=pnt_idx)
        if cond_bias is not None:
            feat = feat + cond_bias
        B, N, F_dim = feat.shape
        conf = self.reliability_lin_1(feat.reshape(-1, F_dim))
        conf = self.reliability_act(conf)
        conf = self.reliability_lin_2(conf)
        return jt.sigmoid(conf).reshape(B, N, 1)

    def _repulsion_loss(self, pc_pred):
        if self.repulsion_weight <= 0:
            return None
        dist = ((pc_pred.unsqueeze(2) - pc_pred.unsqueeze(1)) ** 2).sum(dim=-1)
        k = min(self.repulsion_k + 1, pc_pred.shape[1])
        dist_k, _ = jt.topk(dist, k=k, dim=-1, largest=False)
        dist_k = dist_k[:, :, 1:]
        h2 = max(self.repulsion_h * self.repulsion_h, 1e-8)
        return jt.exp(-dist_k / h2).mean()

    def get_supervised_loss(
        self,
        pc_noisy,
        pc_mix,
        pc_clean,
        pc_normal=None,
        pc_noise_std=None,
        pc_seed_dist=None,
        pc_t=None,
        pc_geom=None,
        pc_score_mask=None,
    ):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        if self.use_refiner:
            return self.get_refiner_loss(
                pc_noisy=pc_noisy,
                pc_clean=pc_clean,
                pc_normal=pc_normal,
                pc_seed_dist=pc_seed_dist,
            )

        B, N_noisy, d = pc_mix.shape
        
        pnt_idx = get_center_focused_indices(
            N_noisy,
            self.num_train_points,
            self.train_center_focus_points,
        )
        pc_mix_full = pc_mix
        pc_clean_full = pc_clean
        pc_normal_full = pc_normal
        pc_geom_full = pc_geom
        pc_t_full = pc_t
        pc_noise_std_full = pc_noise_std

        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        if pc_t is not None:
            pc_t = pc_t[:, pnt_idx, :]
        if pc_normal is not None:
            pc_normal = pc_normal[:, pnt_idx, :]
        if pc_noise_std is not None:
            pc_noise_std = pc_noise_std[:, pnt_idx, :]
        if pc_geom is not None:
            pc_geom = pc_geom[:, pnt_idx, :]
        if pc_score_mask is not None:
            pc_score_mask = pc_score_mask[:, pnt_idx, :]
        point_weight = None
        if pc_seed_dist is not None and self.loss_center_gamma > 0:
            pc_seed_dist = pc_seed_dist[:, pnt_idx, :]
            point_weight = jt.exp(-self.loss_center_gamma * pc_seed_dist).squeeze(-1)
        
        # target
        if self.target_mode == 'clean_minus_noisy':
            grad_dir_t_target = pc_clean - pc_noisy
        elif self.target_mode == 'clean_minus_mix':
            grad_dir_t_target = pc_clean - pc_mix
        elif self.target_mode == 'clean_minus_noisy_or_score_mix':
            target_noisy = pc_clean - pc_noisy
            target_score = pc_clean - pc_mix
            if self.score_target_blend > 0:
                target_score = (
                    self.score_target_blend * target_noisy +
                    (1.0 - self.score_target_blend) * target_score
                )
            if pc_score_mask is None:
                grad_dir_t_target = target_noisy
            else:
                grad_dir_t_target = target_noisy * (1.0 - pc_score_mask) + target_score * pc_score_mask
        elif self.target_mode == 'sigma_score':
            if pc_noise_std is None:
                raise ValueError("target_mode sigma_score requires pc_noise_std")
            grad_dir_t_target = (pc_clean - pc_mix) / jt.maximum(pc_noise_std, 1e-6)
        else:
            raise ValueError(f"unsupported target_mode: {self.target_mode}")

        if pc_normal is not None and self.target_tangent_weight < 1.0:
            normal_norm = jt.sqrt((pc_normal ** 2.0).sum(dim=-1, keepdims=True) + 1e-8)
            normal = pc_normal / normal_norm
            target_normal_scalar = (grad_dir_t_target * normal).sum(dim=-1, keepdims=True)
            target_normal = target_normal_scalar * normal
            target_tangent = grad_dir_t_target - target_normal
            grad_dir_t_target = target_normal + self.target_tangent_weight * target_tangent
        
        # decoder
        pred_dir = self._predict_direction_from_model(
            self,
                pc_mix_full,
                pnt_idx=pnt_idx,
                pc_t=pc_t_full,
                pc_noise_std=pc_noise_std_full,
                pc_geom=pc_geom_full,
        )
        
        err = pred_dir - grad_dir_t_target
        if self.loss_type == 'mse':
            loss = self._point_mean(((err ** 2.0) / self.dsm_sigma).sum(dim=-1), point_weight)
        elif self.loss_type == 'huber':
            abs_err = jt.abs(err)
            quadratic = jt.minimum(abs_err, self.huber_delta)
            linear = abs_err - quadratic
            loss = self._point_mean(
                ((0.5 * quadratic ** 2.0 + self.huber_delta * linear) / self.dsm_sigma).sum(dim=-1),
                point_weight,
            )
        else:
            raise ValueError(f"unsupported loss_type: {self.loss_type}")

        if pc_normal is not None and (self.normal_loss_weight > 0 or self.tangent_damping_weight > 0):
            normal_norm = jt.sqrt((pc_normal ** 2.0).sum(dim=-1, keepdims=True) + 1e-8)
            normal = pc_normal / normal_norm
            pred_normal_scalar = (pred_dir * normal).sum(dim=-1, keepdims=True)
            target_normal_scalar = (grad_dir_t_target * normal).sum(dim=-1, keepdims=True)
            pred_normal = pred_normal_scalar * normal
            target_normal = target_normal_scalar * normal
            if self.normal_loss_weight > 0:
                normal_loss = self._point_mean(
                    (((pred_normal - target_normal) ** 2.0) / self.dsm_sigma).sum(dim=-1),
                    point_weight,
                )
                loss = loss + self.normal_loss_weight * normal_loss
            if self.tangent_damping_weight > 0:
                pred_tangent = pred_dir - pred_normal
                tangent_loss = self._point_mean(
                    ((pred_tangent ** 2.0) / self.dsm_sigma).sum(dim=-1),
                    point_weight,
                )
                loss = loss + self.tangent_damping_weight * tangent_loss

        if pc_normal is not None and self.surface_loss_weight > 0:
            normal_norm = jt.sqrt((pc_normal ** 2.0).sum(dim=-1, keepdims=True) + 1e-8)
            normal = pc_normal / normal_norm
            pc_pred = pc_mix + pred_dir
            plane_residual = ((pc_pred - pc_clean) * normal).sum(dim=-1)
            surface_weight = point_weight
            if pc_geom is not None and (self.surface_planarity_power > 0 or self.surface_curvature_suppress > 0):
                curvature = pc_geom[:, :, 0]
                planarity = pc_geom[:, :, 2]
                geom_weight = jt.ones_like(planarity)
                if self.surface_planarity_power > 0:
                    geom_weight = geom_weight * jt.maximum(planarity, 0.0) ** self.surface_planarity_power
                if self.surface_curvature_suppress > 0:
                    geom_weight = geom_weight * jt.exp(-self.surface_curvature_suppress * curvature)
                surface_weight = geom_weight if surface_weight is None else surface_weight * geom_weight
            surface_loss = self._point_mean((plane_residual ** 2.0) / self.dsm_sigma, surface_weight)
            loss = loss + self.surface_loss_weight * surface_loss

        if self.smooth_weight > 0:
            dist = ((pc_mix.unsqueeze(2) - pc_mix.unsqueeze(1)) ** 2).sum(dim=-1)
            _, idx = jt.topk(dist, k=self.smooth_k + 1, dim=-1, largest=False)
            smooth_terms = []
            for b in range(B):
                nbr = pred_dir[b][idx[b, :, 1:]]
                center = pred_dir[b].unsqueeze(1).broadcast(nbr.shape)
                smooth_terms.append(((center - nbr) ** 2.0).sum(dim=-1).mean())
            loss = loss + self.smooth_weight * jt.stack(smooth_terms).mean()

        repulsion_loss = self._repulsion_loss(pc_mix + pred_dir)
        if repulsion_loss is not None:
            loss = loss + self.repulsion_weight * repulsion_loss

        pred_conf = None
        if self.use_reliability:
            pred_conf = self._decode_reliability(
                self._encoder_input(
                    pc_mix_full,
                    pc_t=pc_t_full,
                    pc_noise_std=pc_noise_std_full,
                    pc_geom=pc_geom_full,
                ),
                pnt_idx=pnt_idx,
                pc_t=pc_t_full,
                pc_noise_std=pc_noise_std_full,
            )
            if pc_noise_std is not None and self.reliability_loss_weight > 0:
                target_norm2 = ((pc_noisy - pc_clean) ** 2.0).sum(dim=-1, keepdims=True)
                sigma2 = (self.reliability_sigma_scale * pc_noise_std) ** 2.0 + 1e-8
                conf_target = jt.exp(-target_norm2 / (2.0 * sigma2))
                conf_loss = self._point_mean(((pred_conf - conf_target) ** 2.0).squeeze(-1), point_weight)
                loss = loss + self.reliability_loss_weight * conf_loss
            if self.unreliable_anchor_weight > 0:
                anchor_weight = 1.0 - pred_conf.squeeze(-1)
                if point_weight is not None:
                    anchor_weight = anchor_weight * point_weight
                anchor_loss = self._point_mean(((pred_dir ** 2.0) / self.dsm_sigma).sum(dim=-1), anchor_weight)
                loss = loss + self.unreliable_anchor_weight * anchor_loss

        if (
            pc_normal_full is not None
            and (self.manifold_loss_weight > 0 or self.manifold_cover_weight > 0)
        ):
            pc_pred = pc_mix + pred_dir
            k = min(self.manifold_k, pc_clean_full.shape[1])
            dist = ((pc_pred.unsqueeze(2) - pc_clean_full.unsqueeze(1)) ** 2.0).sum(dim=-1)
            dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
            clean_nb = []
            normal_nb = []
            for b in range(B):
                clean_nb.append(pc_clean_full[b][idx[b]])
                normal_nb.append(pc_normal_full[b][idx[b]])
            clean_nb = jt.stack(clean_nb, dim=0)
            normal_nb = jt.stack(normal_nb, dim=0)
            normal_norm = jt.sqrt((normal_nb ** 2.0).sum(dim=-1, keepdims=True) + 1e-8)
            normal_nb = normal_nb / normal_norm
            plane_residual = ((pc_pred.unsqueeze(2) - clean_nb) * normal_nb).sum(dim=-1)
            h2 = dist_k[:, :, -1:] + 1e-8
            nb_weight = jt.exp(-dist_k / (2.0 * h2))
            nb_weight = nb_weight / (nb_weight.sum(dim=-1, keepdims=True) + 1e-8)

            manifold_weight = point_weight
            if pred_conf is not None:
                conf_gate = pred_conf.squeeze(-1)
                conf_gate.stop_grad()
                manifold_weight = conf_gate if manifold_weight is None else manifold_weight * conf_gate
            if pc_geom is not None:
                curvature = pc_geom[:, :, 0]
                planarity = pc_geom[:, :, 2]
                edge_gate = (planarity - self.manifold_edge_low) / max(self.manifold_edge_high - self.manifold_edge_low, 1e-6)
                edge_gate = jt.maximum(jt.minimum(edge_gate, 1.0), 0.0)
                edge_gate = edge_gate * jt.exp(-self.manifold_curvature_suppress * curvature)
                manifold_weight = edge_gate if manifold_weight is None else manifold_weight * edge_gate

            if self.manifold_loss_weight > 0:
                manifold_per_point = ((nb_weight * (plane_residual ** 2.0)).sum(dim=-1)) / self.dsm_sigma
                manifold_loss = self._point_mean(manifold_per_point, manifold_weight)
                loss = loss + self.manifold_loss_weight * manifold_loss
            if self.manifold_cover_weight > 0:
                cover_loss = self._point_mean(dist_k[:, :, 0] / self.dsm_sigma, manifold_weight)
                loss = loss + self.manifold_cover_weight * cover_loss

        if self.teacher_model is not None and self.teacher_distill_weight > 0:
            with jt.no_grad():
                teacher_pred = self._predict_direction_from_model(
                    self.teacher_model,
                    pc_mix_full,
                    pnt_idx=pnt_idx,
                    pc_t=pc_t_full,
                    pc_noise_std=pc_noise_std_full,
                    pc_geom=pc_geom_full,
                )
            teacher_pred.stop_grad()
            distill_err = pred_dir - teacher_pred
            distill_loss_per_point = ((distill_err ** 2.0) / self.dsm_sigma).sum(dim=-1)
            distill_weight = point_weight
            if pc_score_mask is not None and self.teacher_distill_score_weight != 1.0:
                score_weight = (
                    (1.0 - pc_score_mask.squeeze(-1)) +
                    self.teacher_distill_score_weight * pc_score_mask.squeeze(-1)
                )
                distill_weight = score_weight if distill_weight is None else distill_weight * score_weight
            distill_loss = self._point_mean(distill_loss_per_point, distill_weight)
            loss = loss + self.teacher_distill_weight * distill_loss
        
        return loss

    def get_refiner_loss(self, pc_noisy, pc_clean, pc_normal=None, pc_seed_dist=None):
        B, N_noisy, _ = pc_noisy.shape
        pnt_idx = get_center_focused_indices(
            N_noisy,
            self.num_train_points,
            self.train_center_focus_points,
        )
        point_weight = None
        if pc_seed_dist is not None and self.loss_center_gamma > 0:
            pc_seed_dist = pc_seed_dist[:, pnt_idx, :]
            point_weight = jt.exp(-self.loss_center_gamma * pc_seed_dist).squeeze(-1)

        with jt.no_grad():
            base_dir_full = self._predict_direction_from_model(
                self.base_model,
                pc_noisy,
                pnt_idx=None,
            )
        base_dir_full.stop_grad()
        base_clean_full = pc_noisy + base_dir_full
        base_clean_full.stop_grad()

        refiner_input = jt.concat([pc_noisy, base_clean_full, base_dir_full], dim=-1)
        pred_corr, pred_gate = self._decode_refine_direction(refiner_input, pnt_idx=pnt_idx)
        target_corr = pc_clean[:, pnt_idx, :] - base_clean_full[:, pnt_idx, :]
        normal = None
        if pc_normal is not None:
            normal = pc_normal[:, pnt_idx, :]
            normal_norm = jt.sqrt((normal ** 2.0).sum(dim=-1, keepdims=True) + 1e-8)
            normal = normal / normal_norm
            if self.refine_target_tangent_weight < 1.0:
                target_normal_scalar = (target_corr * normal).sum(dim=-1, keepdims=True)
                target_normal = target_normal_scalar * normal
                target_tangent = target_corr - target_normal
                target_corr = target_normal + self.refine_target_tangent_weight * target_tangent
        if pred_gate is not None:
            pred_corr_effective = pred_corr * pred_gate
        else:
            pred_corr_effective = pred_corr

        pred_corr_for_loss = pred_corr_effective
        if self.refine_scale_consistent_loss:
            pred_corr_for_loss = self.refine_step_scale * pred_corr_effective

        err = pred_corr_for_loss - target_corr
        if self.refine_loss_type == 'mse':
            loss = self._point_mean(((err ** 2.0) / self.refine_dsm_sigma).sum(dim=-1), point_weight)
        elif self.refine_loss_type == 'huber':
            abs_err = jt.abs(err)
            quadratic = jt.minimum(abs_err, self.refine_huber_delta)
            linear = abs_err - quadratic
            loss = self._point_mean(
                ((0.5 * quadratic ** 2.0 + self.refine_huber_delta * linear) / self.refine_dsm_sigma).sum(dim=-1),
                point_weight,
            )
        else:
            raise ValueError(f"unsupported refine_loss_type: {self.refine_loss_type}")

        if self.refine_corr_anchor_weight > 0:
            corr_anchor = self._point_mean(
                ((pred_corr_for_loss ** 2.0) / self.refine_dsm_sigma).sum(dim=-1),
                point_weight,
            )
            loss = loss + self.refine_corr_anchor_weight * corr_anchor
        if normal is not None and (self.refine_normal_loss_weight > 0 or self.refine_tangent_damping_weight > 0):
            pred_normal_scalar = (pred_corr_for_loss * normal).sum(dim=-1, keepdims=True)
            target_normal_scalar = (target_corr * normal).sum(dim=-1, keepdims=True)
            pred_normal = pred_normal_scalar * normal
            target_normal = target_normal_scalar * normal
            if self.refine_normal_loss_weight > 0:
                normal_loss = self._point_mean(
                    (((pred_normal - target_normal) ** 2.0) / self.refine_dsm_sigma).sum(dim=-1),
                    point_weight,
                )
                loss = loss + self.refine_normal_loss_weight * normal_loss
            if self.refine_tangent_damping_weight > 0:
                pred_tangent = pred_corr_for_loss - pred_normal
                tangent_loss = self._point_mean(
                    ((pred_tangent ** 2.0) / self.refine_dsm_sigma).sum(dim=-1),
                    point_weight,
                )
                loss = loss + self.refine_tangent_damping_weight * tangent_loss
        if normal is not None and self.refine_surface_loss_weight > 0:
            pc_pred = base_clean_full[:, pnt_idx, :] + self.refine_step_scale * pred_corr_effective
            plane_residual = ((pc_pred - pc_clean[:, pnt_idx, :]) * normal).sum(dim=-1)
            surface_loss = self._point_mean((plane_residual ** 2.0) / self.refine_dsm_sigma, point_weight)
            loss = loss + self.refine_surface_loss_weight * surface_loss
        if pred_gate is not None and self.refine_gate_weight > 0:
            if self.refine_gate_target_mode == 'improvement':
                base_err = jt.sqrt(((base_clean_full[:, pnt_idx, :] - pc_clean[:, pnt_idx, :]) ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
                noisy_err = jt.sqrt(((pc_noisy[:, pnt_idx, :] - pc_clean[:, pnt_idx, :]) ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
                improve_need = jt.maximum(base_err - self.refine_gate_noisy_margin * noisy_err, 0.0)
                gate_target = jt.minimum(improve_need / max(self.refine_gate_tau, 1e-8), 1.0)
            elif self.refine_gate_target_mode == 'target_norm':
                target_norm = jt.sqrt((target_corr ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
                gate_target = jt.minimum(target_norm / max(self.refine_gate_tau, 1e-8), 1.0)
            else:
                raise ValueError(f"unsupported refine_gate_target_mode: {self.refine_gate_target_mode}")
            gate_loss = self._point_mean(((pred_gate - gate_target) ** 2.0).squeeze(-1), point_weight)
            loss = loss + self.refine_gate_weight * gate_loss
        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=4, noise_std=None, t_value=None, pc_geom=None):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            if self.use_refiner:
                base_dir = self._predict_direction_from_model(self.base_model, pcl_noisy)
                base_clean = pcl_noisy + base_dir
                refiner_input = jt.concat([pcl_noisy, base_clean, base_dir], dim=-1)
                corr, gate = self._decode_refine_direction(refiner_input)
                if gate is not None:
                    corr = corr * gate
                return base_clean + self.refine_step_scale * corr, None

            pcl_next = pcl_noisy.clone()
            momentum = jt.zeros_like(pcl_next)
            for it in range(num_steps):
                if self.predict_dynamics == 'annealed_sigma_score':
                    denom = max(float(num_steps) - 1.0, 1.0)
                    frac = float(it) / denom
                    sigma_value = self.predict_sigma_max * ((self.predict_sigma_min / self.predict_sigma_max) ** frac)
                    step_t = 1.0
                elif t_value is None:
                    step_t = float(it) / max(float(num_steps), 1.0)
                    sigma_value = noise_std
                else:
                    step_t = t_value
                    sigma_value = noise_std
                pc_t = jt.ones((B, N, 1)) * step_t
                pc_noise_std = None
                if sigma_value is not None:
                    if not isinstance(sigma_value, jt.Var):
                        sigma_value = jt.array(sigma_value).float32()
                    if len(sigma_value.shape) == 0:
                        sigma_value = sigma_value.reshape(1, 1, 1)
                    elif len(sigma_value.shape) == 1:
                        sigma_value = sigma_value.reshape(-1, 1, 1)
                    pc_noise_std = sigma_value.broadcast((B, N, 1))
                pred_dir = self._decode_direction(
                    self._encoder_input(pcl_next, pc_t=pc_t, pc_noise_std=pc_noise_std, pc_geom=pc_geom),
                    pc_t=pc_t,
                    pc_noise_std=pc_noise_std,
                )
                if self.use_noise_condition and pc_noise_std is not None:
                    scale = self._condition_scale(pc_noise_std)
                    if scale is not None:
                        pred_dir = pred_dir * scale
                
                if self.predict_dynamics == 'annealed_sigma_score':
                    step = self.predict_score_step_scale * pc_noise_std * pred_dir
                    if self.predict_score_momentum > 0:
                        momentum = self.predict_score_momentum * momentum + step
                        step = momentum
                    pcl_next = pcl_next + step
                else:
                    pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        pc_normal = None
        if 'pc_normal' in batch:
            pc_normal = batch['pc_normal'].reshape(-1, patch_size, 3)
        pc_noise_std = None
        if 'pc_noise_std' in batch:
            pc_noise_std = batch['pc_noise_std'].reshape(-1, patch_size, 1)
        pc_t = None
        if 'pc_t' in batch:
            pc_t = batch['pc_t'].reshape(-1, patch_size, 1)
        pc_geom = None
        if 'pc_geom' in batch:
            pc_geom = batch['pc_geom'].reshape(-1, patch_size, self.geometry_dim)
        pc_seed_dist = None
        if 'pc_seed_dist' in batch:
            pc_seed_dist = batch['pc_seed_dist'].reshape(-1, patch_size, 1)
        pc_score_mask = None
        if 'pc_score_mask' in batch:
            pc_score_mask = batch['pc_score_mask'].reshape(-1, patch_size, 1)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
            pc_normal=pc_normal,
            pc_noise_std=pc_noise_std,
            pc_seed_dist=pc_seed_dist,
            pc_t=pc_t,
            pc_geom=pc_geom,
            pc_score_mask=pc_score_mask,
        )
        return {"loss": loss}
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)
    
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            noise_std = self.predict_noise_std
            if self.use_noise_condition and noise_std is None:
                noise_std = estimate_noise_std_numpy(
                    pc_noisy.detach().numpy(),
                    k=self.predict_noise_estimate_k,
                    max_samples=self.predict_noise_estimate_samples,
                    lo=self.noise_condition_min,
                    hi=self.noise_condition_max,
                )
            pc_next = patch_based_denoise(
                model=self,
                pcl_noisy=pc_next,
                patch_size=self.predict_patch_size,
                seed_k=self.predict_seed_k,
                seed_k_alpha=self.predict_seed_k_alpha,
                aggregation=self.predict_patch_aggregation,
                weight_gamma=self.predict_patch_weight_gamma,
                noise_std=noise_std,
                num_steps=self.predict_num_steps,
            )
            pc_denoised = pc_next.detach().numpy()
            res.append({"pc_denoised": pc_denoised})
        return res
    
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                    **({"pc_normal": b.meta["pc_normal"]} if "pc_normal" in b.meta else {}),
                    **({"pc_noise_std": b.meta["pc_noise_std"]} if "pc_noise_std" in b.meta else {}),
                    **({"pc_t": b.meta["pc_t"]} if "pc_t" in b.meta else {}),
                    **({"pc_geom": b.meta["pc_geom"]} if "pc_geom" in b.meta else {}),
                    **({"pc_seed_dist": b.meta["pc_seed_dist"]} if "pc_seed_dist" in b.meta else {}),
                    **({"pc_score_mask": b.meta["pc_score_mask"]} if "pc_score_mask" in b.meta else {}),
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res

def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]  # (N, 3)
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]  # (3,)
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices

def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return:
        dist: (B, P, k)
        idx:  (B, P, k)
        nn:   (B, P, k, 3)
    """
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def patch_based_denoise(
    model: VelocityModule,
    pcl_noisy,
    patch_size=1000,
    seed_k=6,
    seed_k_alpha=1,
    aggregation='best',
    weight_gamma=4.0,
    noise_std=None,
    num_steps=1,
) -> jt.Var:
    """
    pcl_noisy: (N, 3)
    """
    assert len(pcl_noisy.shape) == 2
    
    N, d = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy_orig_np = pcl_noisy.detach().numpy()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    
    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)
    
    from ..data.asset import Exporter
    pts = patches[0].reshape(-1, 3).detach().numpy()
    
    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)
    
    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    
    all_dists = jt.ones((num_patches, N)) * 1e10
    
    for i in range(num_patches):
        all_dists[i][point_idxs[i]] = patch_dists[i]
        
    weights = jt.exp(-weight_gamma * all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)
    patches_denoised = []
    
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            pc_geom = None
            if model.use_geometry_input:
                geom_np = np.stack(
                    [estimate_patch_geometry(curr[j].detach().numpy(), k=model.predict_noise_estimate_k) for j in range(curr.shape[0])],
                    axis=0,
                )
                pc_geom = jt.array(geom_np)
            out, _ = model.denoise_langevin_dynamics(curr, num_steps=num_steps, noise_std=noise_std, pc_geom=pc_geom)
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        i += patch_step
    
    patches_denoised = jt.concat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_expand
    point_idxs_np = point_idxs.detach().numpy()
    best_weights_idx_np = best_weights_idx.detach().numpy()
    patches_denoised_np = patches_denoised.detach().numpy()
    patch_dists_np = patch_dists.detach().numpy()

    if aggregation in ('weighted', 'disp_weighted'):
        pcl_sum_np = np.zeros((N, d), dtype=np.float64)
        weight_sum_np = np.zeros((N, 1), dtype=np.float64)
        patch_weights_np = np.exp(-float(weight_gamma) * patch_dists_np).astype(np.float64)
        if aggregation == 'disp_weighted':
            max_disp = float(model.predict_disp_max_scale) * float(noise_std if noise_std is not None else model.noise_condition_ref)
            if max_disp <= 0:
                max_disp = 0.05
            disp_conf_gamma = float(model.predict_disp_confidence_gamma)
            disp_conf_floor = float(getattr(model, 'predict_disp_confidence_floor', 0.0))
        for patch_id in range(num_patches):
            point_ids = point_idxs_np[patch_id]
            w = patch_weights_np[patch_id][:, None]
            if aggregation == 'disp_weighted':
                patch_base = pcl_noisy_orig_np[point_ids].astype(np.float64)
                patch_pred = patches_denoised_np[patch_id].astype(np.float64)
                disp = patch_pred - patch_base
                disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
                if disp_conf_gamma > 0:
                    over = np.maximum(disp_norm / max(max_disp, 1e-8) - 1.0, 0.0)
                    conf = np.exp(-disp_conf_gamma * over * over)
                    if disp_conf_floor > 0:
                        conf = np.maximum(conf, disp_conf_floor)
                    w = w * conf
                disp = disp * np.minimum(1.0, max_disp / (disp_norm + 1e-8))
                value = patch_base + disp
            else:
                value = patches_denoised_np[patch_id].astype(np.float64)
            np.add.at(pcl_sum_np, point_ids, value * w)
            np.add.at(weight_sum_np, point_ids, w)
        valid = weight_sum_np[:, 0] > 1e-12
        pcl_out_np = np.empty((N, d), dtype=np.float32)
        pcl_out_np[valid] = (pcl_sum_np[valid] / weight_sum_np[valid]).astype(np.float32)
        if not np.all(valid):
            pcl_out_np[~valid] = pcl_noisy_orig_np[~valid]
        return jt.array(pcl_out_np)

    if aggregation != 'best':
        raise ValueError(f"unsupported patch aggregation: {aggregation}")

    pcl_out_np = np.empty((N, d), dtype=np.float32)
    filled = np.zeros((N,), dtype=bool)
    for patch_id in range(num_patches):
        point_ids = point_idxs_np[patch_id]
        mask = best_weights_idx_np[point_ids] == patch_id
        if np.any(mask):
            pcl_out_np[point_ids[mask]] = patches_denoised_np[patch_id, mask]
            filled[point_ids[mask]] = True
    if not np.all(filled):
        missing = np.where(~filled)[0]
        fallback_patch = best_weights_idx_np[missing]
        for i, patch_id in zip(missing, fallback_patch):
            pos = np.where(point_idxs_np[patch_id] == i)[0]
            if len(pos) > 0:
                pcl_out_np[i] = patches_denoised_np[patch_id, pos[0]]
            else:
                pcl_out_np[i] = pcl_noisy[0, i].detach().numpy()
    return jt.array(pcl_out_np)


def estimate_noise_std_numpy(
    pc: np.ndarray,
    k: int=24,
    max_samples: int=8192,
    lo: float=0.004,
    hi: float=0.024,
) -> float:
    """Estimate noise scale from robust local plane residuals."""
    if pc.shape[0] == 0:
        return float((lo + hi) * 0.5)
    if pc.shape[0] > max_samples:
        idx = np.random.default_rng(12345).choice(pc.shape[0], size=max_samples, replace=False)
        query = pc[idx]
    else:
        query = pc
    from scipy.spatial import cKDTree
    tree = cKDTree(pc, compact_nodes=False, balanced_tree=False)
    _, nn_idx = tree.query(query, k=min(k, pc.shape[0]))
    residuals = []
    for ids in nn_idx:
        nb = pc[ids]
        center = nb.mean(axis=0)
        centered = nb - center
        cov = centered.T @ centered / max(len(ids), 1)
        _, vec = np.linalg.eigh(cov)
        normal = vec[:, 0]
        residuals.append(np.abs((query[len(residuals)] - center) @ normal))
    residuals = np.asarray(residuals, dtype=np.float64)
    sigma = 1.4826 * np.median(residuals)
    return float(np.clip(sigma, lo, hi))
