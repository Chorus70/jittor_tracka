from typing import Optional
from jittor import nn

import jittor as jt


class PCTSelfAttention(nn.Module):
    def __init__(self, channels, position_dim=0):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        if hasattr(self.q_conv, "conv") and hasattr(self.k_conv, "conv"):
            self.q_conv.conv.weight = self.k_conv.conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.position_dim = position_dim
        if position_dim > 0:
            self.pos_conv = nn.Conv1d(position_dim, channels, 1, bias=False)

    def execute(self, x, pos=None):
        if self.position_dim > 0:
            if pos is None:
                raise ValueError("PCTSelfAttention position embedding enabled but pos is None")
            x = x + self.pos_conv(pos.permute(0, 2, 1))
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        energy = nn.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (attention.sum(dim=1, keepdims=True) + 1e-9)
        x_r = nn.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        return x + x_r


class PCTFeatureExtraction(nn.Module):
    def __init__(
        self,
        input_dim=3,
        embedding_dim=512,
        attention_channels=128,
        num_attention_layers=4,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
        use_position_embedding=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.attention_channels = attention_channels
        self.num_attention_layers = num_attention_layers
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim
        self.use_position_embedding = use_position_embedding
        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        self.conv1 = nn.Conv1d(input_dim, attention_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(attention_channels, attention_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(attention_channels)
        self.bn2 = nn.BatchNorm1d(attention_channels)
        self.relu = nn.ReLU()

        for i in range(num_attention_layers):
            setattr(
                self,
                f"sa{i + 1}",
                PCTSelfAttention(
                    attention_channels,
                    position_dim=knn_coord_dim if use_position_embedding else 0,
                ),
            )

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(attention_channels * num_attention_layers, embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(scale=0.2),
        )

    def execute(self, x):
        # x: (B, N, C)
        B, N, _ = x.shape
        x_input = x
        feat = x.permute(0, 2, 1)
        feat = self.relu(self.bn1(self.conv1(feat)))
        feat = self.relu(self.bn2(self.conv2(feat)))

        attn_feats = []
        for i in range(self.num_attention_layers):
            pos = x_input[:, :, :self.knn_coord_dim] if self.use_position_embedding else None
            feat = getattr(self, f"sa{i + 1}")(feat, pos=pos)
            attn_feats.append(feat)

        feat = jt.concat(attn_feats, dim=1)
        feat = self.conv_fuse(feat).permute(0, 2, 1)

        features = [feat]
        if self.use_global_feature:
            global_feat = jt.max(feat, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, feat.shape[-1])))
        if self.use_input_coords:
            features.append(x_input[:, :, :self.knn_coord_dim])
        return jt.concat(features, dim=-1)


class PCTLocalOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        # x: (B, N, K, C)
        B, N, K, C = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B * N, C, K)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = jt.max(x, dim=2)
        return x.reshape(B, N, -1).permute(0, 2, 1)


class PCTNeighborFeatureExtraction(nn.Module):
    def __init__(
        self,
        k=32,
        input_dim=3,
        embedding_dim=512,
        attention_channels=128,
        num_attention_layers=4,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
        use_position_embedding=False,
    ):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.attention_channels = attention_channels
        self.num_attention_layers = num_attention_layers
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim
        self.use_position_embedding = use_position_embedding
        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        self.conv1 = nn.Conv1d(input_dim, attention_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(attention_channels, attention_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(attention_channels)
        self.bn2 = nn.BatchNorm1d(attention_channels)
        self.relu = nn.ReLU()
        self.local_op = PCTLocalOp(attention_channels * 2, attention_channels)

        for i in range(num_attention_layers):
            setattr(
                self,
                f"sa{i + 1}",
                PCTSelfAttention(
                    attention_channels,
                    position_dim=knn_coord_dim if use_position_embedding else 0,
                ),
            )

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(attention_channels * (num_attention_layers + 1), embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(scale=0.2),
        )

    def _group_neighbors(self, xyz, feat):
        B, N, C = feat.shape
        k = min(self.k, max(N - 1, 1))
        idx = get_knn_idx(xyz, xyz, k, offset=1 if N > 1 else 0)
        grouped = []
        for b in range(B):
            nb = feat[b][idx[b]]
            center = feat[b].unsqueeze(1).broadcast(nb.shape)
            grouped.append(jt.concat([nb - center, center], dim=-1))
        return jt.stack(grouped, dim=0)

    def execute(self, x):
        # x: (B, N, C)
        B, N, _ = x.shape
        x_input = x
        xyz = x[:, :, :self.knn_coord_dim]
        feat = x.permute(0, 2, 1)
        feat = self.relu(self.bn1(self.conv1(feat)))
        feat = self.relu(self.bn2(self.conv2(feat)))
        feat = feat.permute(0, 2, 1)

        grouped = self._group_neighbors(xyz, feat)
        feat = self.local_op(grouped)
        attn_feats = [feat]
        for i in range(self.num_attention_layers):
            pos = xyz if self.use_position_embedding else None
            feat = getattr(self, f"sa{i + 1}")(feat, pos=pos)
            attn_feats.append(feat)

        feat = jt.concat(attn_feats, dim=1)
        feat = self.conv_fuse(feat).permute(0, 2, 1)

        features = [feat]
        if self.use_global_feature:
            global_feat = jt.max(feat, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, feat.shape[-1])))
        if self.use_input_coords:
            features.append(x_input[:, :, :self.knn_coord_dim])
        return jt.concat(features, dim=-1)


class PCTDenoiseFeatureExtraction(nn.Module):
    def __init__(
        self,
        k=32,
        input_dim=3,
        embedding_dim=512,
        attention_channels=128,
        num_attention_layers=4,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
        use_position_embedding=True,
    ):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.attention_channels = attention_channels
        self.num_attention_layers = num_attention_layers
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim
        self.use_position_embedding = use_position_embedding
        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        self.conv1 = nn.Conv1d(input_dim, attention_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(attention_channels, attention_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(attention_channels)
        self.bn2 = nn.BatchNorm1d(attention_channels)
        self.relu = nn.ReLU()

        local_in_channels = attention_channels * 2 + knn_coord_dim * 2
        self.local_op_1 = PCTLocalOp(local_in_channels, attention_channels)
        self.local_op_2 = PCTLocalOp(local_in_channels, attention_channels)

        for i in range(num_attention_layers):
            setattr(
                self,
                f"sa{i + 1}",
                PCTSelfAttention(
                    attention_channels,
                    position_dim=knn_coord_dim if use_position_embedding else 0,
                ),
            )

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(attention_channels * (num_attention_layers + 2), embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(scale=0.2),
        )

    def _group_local(self, xyz, feat):
        B, N, C = feat.shape
        k = min(self.k, max(N - 1, 1))
        idx = get_knn_idx(xyz, xyz, k, offset=1 if N > 1 else 0)
        grouped = []
        for b in range(B):
            nb_feat = feat[b][idx[b]]
            center_feat = feat[b].unsqueeze(1).broadcast(nb_feat.shape)
            nb_xyz = xyz[b][idx[b]]
            center_xyz = xyz[b].unsqueeze(1).broadcast(nb_xyz.shape)
            grouped.append(
                jt.concat(
                    [
                        nb_feat - center_feat,
                        center_feat,
                        nb_xyz - center_xyz,
                        center_xyz,
                    ],
                    dim=-1,
                )
            )
        return jt.stack(grouped, dim=0)

    def execute(self, x):
        # x: (B, N, C). The first three channels are local patch coordinates.
        B, N, _ = x.shape
        x_input = x
        xyz = x[:, :, :self.knn_coord_dim]
        feat = x.permute(0, 2, 1)
        feat = self.relu(self.bn1(self.conv1(feat)))
        feat = self.relu(self.bn2(self.conv2(feat)))
        feat = feat.permute(0, 2, 1)

        grouped_1 = self._group_local(xyz, feat)
        local_1 = self.local_op_1(grouped_1)
        grouped_2 = self._group_local(xyz, local_1.permute(0, 2, 1))
        feat = self.local_op_2(grouped_2)

        attn_feats = [local_1, feat]
        for i in range(self.num_attention_layers):
            pos = xyz if self.use_position_embedding else None
            feat = getattr(self, f"sa{i + 1}")(feat, pos=pos)
            attn_feats.append(feat)

        feat = jt.concat(attn_feats, dim=1)
        feat = self.conv_fuse(feat).permute(0, 2, 1)

        features = [feat]
        if self.use_global_feature:
            global_feat = jt.max(feat, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, feat.shape[-1])))
        if self.use_input_coords:
            features.append(x_input[:, :, :self.knn_coord_dim])
        return jt.concat(features, dim=-1)


class PCTHierarchicalFeatureExtraction(nn.Module):
    def __init__(
        self,
        k=32,
        input_dim=3,
        embedding_dim=512,
        attention_channels=128,
        num_attention_layers=4,
        num_tokens=128,
        interp_k=3,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
        use_position_embedding=True,
    ):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.attention_channels = attention_channels
        self.num_attention_layers = num_attention_layers
        self.num_tokens = num_tokens
        self.interp_k = interp_k
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim
        self.use_position_embedding = use_position_embedding
        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        self.conv1 = nn.Conv1d(input_dim, attention_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(attention_channels, attention_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(attention_channels)
        self.bn2 = nn.BatchNorm1d(attention_channels)
        self.relu = nn.ReLU()

        local_in_channels = attention_channels * 2 + knn_coord_dim * 2
        self.local_op = PCTLocalOp(local_in_channels, attention_channels)
        for i in range(num_attention_layers):
            setattr(
                self,
                f"sa{i + 1}",
                PCTSelfAttention(
                    attention_channels,
                    position_dim=knn_coord_dim if use_position_embedding else 0,
                ),
            )
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(attention_channels * (num_attention_layers + 1), embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(scale=0.2),
        )

    def _token_indices(self, N):
        T = min(max(int(self.num_tokens), 1), N)
        if T == N:
            return jt.arange(N).int32()
        # Deterministic coverage avoids a CPU FPS sync in every forward pass while
        # still giving attention a compact whole-patch token set.
        idx = jt.linspace(0, N - 1, T).int32()
        return idx

    def _group_tokens(self, xyz, feat, token_idx):
        B, N, C = feat.shape
        token_xyz = xyz[:, token_idx, :]
        token_feat = feat[:, token_idx, :]
        k = min(self.k, N)
        idx = get_knn_idx(token_xyz, xyz, k, offset=0)
        grouped = []
        for b in range(B):
            nb_feat = feat[b][idx[b]]
            center_feat = token_feat[b].unsqueeze(1).broadcast(nb_feat.shape)
            nb_xyz = xyz[b][idx[b]]
            center_xyz = token_xyz[b].unsqueeze(1).broadcast(nb_xyz.shape)
            grouped.append(
                jt.concat(
                    [nb_feat - center_feat, center_feat, nb_xyz - center_xyz, center_xyz],
                    dim=-1,
                )
            )
        return jt.stack(grouped, dim=0), token_xyz

    def _interpolate_to_points(self, xyz, token_xyz, token_feat):
        B, N, _ = xyz.shape
        T = token_xyz.shape[1]
        k = min(self.interp_k, T)
        dist = ((xyz.unsqueeze(2) - token_xyz.unsqueeze(1)) ** 2).sum(-1)
        dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
        gathered = []
        for b in range(B):
            gathered.append(token_feat[b][idx[b]])
        gathered = jt.stack(gathered, dim=0)
        weight = 1.0 / (dist_k + 1e-8)
        weight = weight / (weight.sum(dim=-1, keepdims=True) + 1e-8)
        return (gathered * weight.unsqueeze(-1)).sum(dim=2)

    def execute(self, x):
        # x: (B, N, C). The first three channels are local patch coordinates.
        B, N, _ = x.shape
        x_input = x
        xyz = x[:, :, :self.knn_coord_dim]
        feat = x.permute(0, 2, 1)
        feat = self.relu(self.bn1(self.conv1(feat)))
        feat = self.relu(self.bn2(self.conv2(feat))).permute(0, 2, 1)

        token_idx = self._token_indices(N)
        grouped, token_xyz = self._group_tokens(xyz, feat, token_idx)
        token_feat = self.local_op(grouped)

        attn_feats = [token_feat]
        for i in range(self.num_attention_layers):
            pos = token_xyz if self.use_position_embedding else None
            token_feat = getattr(self, f"sa{i + 1}")(token_feat, pos=pos)
            attn_feats.append(token_feat)

        token_feat = jt.concat(attn_feats, dim=1)
        token_feat = self.conv_fuse(token_feat).permute(0, 2, 1)
        feat = self._interpolate_to_points(xyz, token_xyz, token_feat)

        features = [feat]
        if self.use_global_feature:
            global_feat = jt.max(token_feat, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, token_feat.shape[-1])))
        if self.use_input_coords:
            features.append(x_input[:, :, :self.knn_coord_dim])
        return jt.concat(features, dim=-1)


class PCTRefineLocalFeatureExtraction(nn.Module):
    def __init__(
        self,
        k=32,
        input_dim=9,
        embedding_dim=512,
        attention_channels=96,
        num_attention_layers=2,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
        use_position_embedding=True,
    ):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.attention_channels = attention_channels
        self.num_attention_layers = num_attention_layers
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim
        self.use_position_embedding = use_position_embedding
        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        self.conv1 = nn.Conv1d(input_dim, attention_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(attention_channels, attention_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(attention_channels)
        self.bn2 = nn.BatchNorm1d(attention_channels)
        self.relu = nn.ReLU()

        local_in_channels = attention_channels * 2 + 6
        self.local_op_1 = PCTLocalOp(local_in_channels, attention_channels)
        self.local_op_2 = PCTLocalOp(local_in_channels, attention_channels)
        for i in range(num_attention_layers):
            setattr(
                self,
                f"sa{i + 1}",
                PCTSelfAttention(
                    attention_channels,
                    position_dim=knn_coord_dim if use_position_embedding else 0,
                ),
            )
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(attention_channels * (num_attention_layers + 2), embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(scale=0.2),
        )

    def _coords(self, x):
        if x.shape[-1] >= 6:
            return x[:, :, 3:6]
        return x[:, :, :self.knn_coord_dim]

    def _base_dir(self, x):
        if x.shape[-1] >= 9:
            return x[:, :, 6:9]
        return jt.zeros((x.shape[0], x.shape[1], 3))

    def _group_local(self, coord, base_dir, feat):
        B, N, C = feat.shape
        k = min(self.k, max(N - 1, 1))
        idx = get_knn_idx(coord, coord, k, offset=1 if N > 1 else 0)
        grouped = []
        for b in range(B):
            nb_feat = feat[b][idx[b]]
            center_feat = feat[b].unsqueeze(1).broadcast(nb_feat.shape)
            nb_coord = coord[b][idx[b]]
            center_coord = coord[b].unsqueeze(1).broadcast(nb_coord.shape)
            nb_dir = base_dir[b][idx[b]]
            center_dir = base_dir[b].unsqueeze(1).broadcast(nb_dir.shape)
            grouped.append(
                jt.concat(
                    [
                        nb_feat - center_feat,
                        center_feat,
                        nb_coord - center_coord,
                        nb_dir - center_dir,
                    ],
                    dim=-1,
                )
            )
        return jt.stack(grouped, dim=0)

    def execute(self, x):
        B, N, _ = x.shape
        coord = self._coords(x)
        base_dir = self._base_dir(x)
        feat = x.permute(0, 2, 1)
        feat = self.relu(self.bn1(self.conv1(feat)))
        feat = self.relu(self.bn2(self.conv2(feat))).permute(0, 2, 1)

        grouped_1 = self._group_local(coord, base_dir, feat)
        local_1 = self.local_op_1(grouped_1)
        grouped_2 = self._group_local(coord, base_dir, local_1.permute(0, 2, 1))
        feat = self.local_op_2(grouped_2)

        attn_feats = [local_1, feat]
        for i in range(self.num_attention_layers):
            pos = coord if self.use_position_embedding else None
            feat = getattr(self, f"sa{i + 1}")(feat, pos=pos)
            attn_feats.append(feat)

        feat = jt.concat(attn_feats, dim=1)
        feat = self.conv_fuse(feat).permute(0, 2, 1)

        features = [feat]
        if self.use_global_feature:
            global_feat = jt.max(feat, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, feat.shape[-1])))
        if self.use_input_coords:
            features.append(coord)
        return jt.concat(features, dim=-1)


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__()
        
        if activation == 'ReLU':
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU()
            )
            self.lin = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise Exception("Please assign valid activation to MLP!")
    
    def execute(self, x, edge_index):
        """
        x: (N, C)
        edge_index: (2, E)
        """
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        
        # gather
        x_i = x[dst]  # (E, C)
        x_j = x[src]  # (E, C)
        
        # message
        tmp = jt.concat([x_i, x_j - x_i], dim=1)  # (E, 2C)
        msg = self.mlp(tmp)  # (E, out_channels)
        
        N = x.shape[0]
        out = jt.full((N, msg.shape[1]), 0)
        cnt = jt.full((N, msg.shape[1]), 0)
        
        # scatter mean
        out = out.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), msg, reduce='add')
        cnt = cnt.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), jt.ones_like(msg), reduce='add')
        out = out / (cnt + 1)
        out_2 = self.lin(x)
        return out + out_2

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__(in_channels, out_channels, activation)
    
    def execute(self, x, edge_index):
        return super().execute(x, edge_index)

class FeatureExtraction(nn.Module):
    def __init__(
        self,
        k=32,
        input_dim=0,
        embedding_dim=512,
        distance_estimation=False,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
    ):
        super().__init__()

        self.k = k
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.distance_estimation = distance_estimation
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim
        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        self.conv1 = DynamicEdgeConv(self.input_dim, embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(
            embedding_dim // 8 + embedding_dim // 4,
            embedding_dim,
            activation=None
        )

    # ========= edge_index 构建 =========
    def get_edge_index(self, x, coord_only=False):
        # x: (B, N, C)
        B, N, _ = x.shape
        knn_x = x
        if coord_only and self.knn_coord_dim > 0 and x.shape[-1] > self.knn_coord_dim:
            knn_x = x[:, :, :self.knn_coord_dim]
        knn_idx = get_knn_idx(knn_x, knn_x, self.k + 1)  # (B, N, k+1)
        knn_idx = knn_idx[:, :, 1:]
        base = jt.arange(B) * N  # (B,)
        base = base.reshape(B, 1, 1)
        
        knn_idx = knn_idx + base  # (B, N, k)
        
        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, self.k))
        dst = dst + base
        
        src = knn_idx.reshape(-1)
        dst = dst.reshape(-1)
        
        edge_index = jt.stack([src, dst], dim=0)  # (2, E)
        
        return edge_index
    
    def normalize_patch(self, pcl):
        scale = jt.sqrt((pcl ** 2).sum(-1, keepdims=True))
        scale = scale.max(dim=-2, keepdims=True)
        return pcl / (scale + 1e-8) # type: ignore
    
    def execute(self, x):
        # x: (B, N, C)
        B, N, _ = x.shape
        x_input = x
        
        if self.distance_estimation:
            x = self.normalize_patch(x)
        
        # -------- conv1 --------
        edge_index = self.get_edge_index(
            x,
            coord_only=(self.input_dim > self.knn_coord_dim and x.shape[-1] == self.input_dim),
        )
        x_flat = x.reshape(B * N, -1)
        
        x1 = self.conv1(x_flat, edge_index)
        x1 = x1.reshape(B, N, -1)
        
        # -------- conv2 --------
        edge_index = self.get_edge_index(x1)
        x1_flat = x1.reshape(B * N, -1)
        
        x2 = self.conv2(x1_flat, edge_index)
        x2 = x2.reshape(B, N, -1)
        
        # -------- conv3 --------
        edge_index = self.get_edge_index(x2)
        
        x_combined = jt.concat([x1, x2], dim=-1)
        x_combined_flat = x_combined.reshape(B * N, -1) # type: ignore
        
        x3 = self.conv3(x_combined_flat, edge_index)
        x3 = x3.reshape(B, N, -1)

        features = [x3]
        if self.use_global_feature:
            global_feat = jt.max(x3, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, x3.shape[-1])))
        if self.use_input_coords:
            features.append(x_input[:, :, :self.knn_coord_dim])
        return jt.concat(features, dim=-1)


class EdgePCTHybridFeatureExtraction(nn.Module):
    def __init__(
        self,
        k=32,
        input_dim=0,
        embedding_dim=512,
        attention_channels=128,
        num_attention_layers=2,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
        use_position_embedding=True,
    ):
        super().__init__()

        self.k = k
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.attention_channels = attention_channels
        self.num_attention_layers = num_attention_layers
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim
        self.use_position_embedding = use_position_embedding
        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        self.conv1 = DynamicEdgeConv(self.input_dim, embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(
            embedding_dim // 8 + embedding_dim // 4,
            attention_channels,
            activation=None,
        )

        for i in range(num_attention_layers):
            setattr(
                self,
                f"sa{i + 1}",
                PCTSelfAttention(
                    attention_channels,
                    position_dim=knn_coord_dim if use_position_embedding else 0,
                ),
            )

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(attention_channels * (num_attention_layers + 1), embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(scale=0.2),
        )

    def get_edge_index(self, x, coord_only=False):
        B, N, _ = x.shape
        knn_x = x
        if coord_only and self.knn_coord_dim > 0 and x.shape[-1] > self.knn_coord_dim:
            knn_x = x[:, :, :self.knn_coord_dim]
        knn_idx = get_knn_idx(knn_x, knn_x, self.k + 1)
        knn_idx = knn_idx[:, :, 1:]
        base = jt.arange(B) * N
        base = base.reshape(B, 1, 1)
        knn_idx = knn_idx + base

        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, self.k))
        dst = dst + base

        src = knn_idx.reshape(-1)
        dst = dst.reshape(-1)
        return jt.stack([src, dst], dim=0)

    def execute(self, x):
        B, N, _ = x.shape
        x_input = x
        xyz = x[:, :, :self.knn_coord_dim]

        edge_index = self.get_edge_index(
            x,
            coord_only=(self.input_dim > self.knn_coord_dim and x.shape[-1] == self.input_dim),
        )
        x_flat = x.reshape(B * N, -1)
        x1 = self.conv1(x_flat, edge_index).reshape(B, N, -1)

        edge_index = self.get_edge_index(x1)
        x2 = self.conv2(x1.reshape(B * N, -1), edge_index).reshape(B, N, -1)

        edge_index = self.get_edge_index(x2)
        x_combined = jt.concat([x1, x2], dim=-1)
        feat = self.conv3(x_combined.reshape(B * N, -1), edge_index).reshape(B, N, -1)
        feat = feat.permute(0, 2, 1)

        attn_feats = [feat]
        for i in range(self.num_attention_layers):
            pos = xyz if self.use_position_embedding else None
            feat = getattr(self, f"sa{i + 1}")(feat, pos=pos)
            attn_feats.append(feat)

        feat = jt.concat(attn_feats, dim=1)
        feat = self.conv_fuse(feat).permute(0, 2, 1)

        features = [feat]
        if self.use_global_feature:
            global_feat = jt.max(feat, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, feat.shape[-1])))
        if self.use_input_coords:
            features.append(x_input[:, :, :self.knn_coord_dim])
        return jt.concat(features, dim=-1)


class EdgePCTResidualFeatureExtraction(nn.Module):
    def __init__(
        self,
        k=32,
        input_dim=0,
        embedding_dim=512,
        num_attention_layers=1,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
        use_position_embedding=True,
    ):
        super().__init__()

        self.k = k
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.num_attention_layers = num_attention_layers
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim
        self.use_position_embedding = use_position_embedding
        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        self.conv1 = DynamicEdgeConv(self.input_dim, embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(
            embedding_dim // 8 + embedding_dim // 4,
            embedding_dim,
            activation=None,
        )

        for i in range(num_attention_layers):
            setattr(
                self,
                f"sa{i + 1}",
                PCTSelfAttention(
                    embedding_dim,
                    position_dim=knn_coord_dim if use_position_embedding else 0,
                ),
            )
        self.residual_proj = nn.Conv1d(embedding_dim, embedding_dim, 1, bias=False)
        self.residual_proj.weight.assign(jt.zeros_like(self.residual_proj.weight))

    def get_edge_index(self, x, coord_only=False):
        B, N, _ = x.shape
        knn_x = x
        if coord_only and self.knn_coord_dim > 0 and x.shape[-1] > self.knn_coord_dim:
            knn_x = x[:, :, :self.knn_coord_dim]
        knn_idx = get_knn_idx(knn_x, knn_x, self.k + 1)
        knn_idx = knn_idx[:, :, 1:]
        base = jt.arange(B) * N
        base = base.reshape(B, 1, 1)
        knn_idx = knn_idx + base

        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, self.k))
        dst = dst + base

        src = knn_idx.reshape(-1)
        dst = dst.reshape(-1)
        return jt.stack([src, dst], dim=0)

    def execute(self, x):
        B, N, _ = x.shape
        x_input = x
        xyz = x[:, :, :self.knn_coord_dim]

        edge_index = self.get_edge_index(
            x,
            coord_only=(self.input_dim > self.knn_coord_dim and x.shape[-1] == self.input_dim),
        )
        x_flat = x.reshape(B * N, -1)
        x1 = self.conv1(x_flat, edge_index).reshape(B, N, -1)

        edge_index = self.get_edge_index(x1)
        x2 = self.conv2(x1.reshape(B * N, -1), edge_index).reshape(B, N, -1)

        edge_index = self.get_edge_index(x2)
        x_combined = jt.concat([x1, x2], dim=-1)
        base_feat = self.conv3(x_combined.reshape(B * N, -1), edge_index).reshape(B, N, -1)

        attn = base_feat.permute(0, 2, 1)
        for i in range(self.num_attention_layers):
            pos = xyz if self.use_position_embedding else None
            attn = getattr(self, f"sa{i + 1}")(attn, pos=pos)
        residual = self.residual_proj(attn).permute(0, 2, 1)
        feat = base_feat + residual

        features = [feat]
        if self.use_global_feature:
            global_feat = jt.max(feat, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, feat.shape[-1])))
        if self.use_input_coords:
            features.append(x_input[:, :, :self.knn_coord_dim])
        return jt.concat(features, dim=-1)


class Decoder(nn.Module):
    
    def __init__(self, z_dim, dim, out_dim, hidden_size):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        c_dim = z_dim
        self.lin_1 = nn.Linear(c_dim, c_dim)
        self.bn_1_out = nn.BatchNorm1d(c_dim)
        
        self.lin_2 = nn.Linear(c_dim, hidden_size)
        self.bn_2_out = nn.BatchNorm1d(hidden_size)
        
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        
        self.actvn_out = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def execute(self, c, B=None, N=None):
        """
        c: (B*N, F)
        """
        net = self.lin_1(c)
        net = self.bn_1_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        net = self.lin_2(net)
        net = self.bn_2_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        if self.out_dim == 1:
            net = net.reshape(B, N, -1)
            net = jt.max(net, dim=1, keepdims=True)
            net = self.lin_3(net)
            net = jt.sigmoid(net)
        else:
            net = self.lin_3(net)
        return net

def get_knn_idx(x, y, k, offset=0):
    """
    x: (B, N, d)
    y: (B, M, d)
    return: (B, N, k)
    """
    K = k + offset
    if x.shape[-1] == 3:
        _, idx = jt.misc.knn(x, y, K)
    else:
        dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
        _, idx = jt.topk(dist, k=K, dim=-1, largest=False)
    return idx[:, :, offset:]


# =============================================================================
# Multi-Scale Fusion Modules
# =============================================================================


class MultiScaleFusion(nn.Module):
    """Learned attention-based fusion of multi-scale features.

    Takes features from K scales (each (B, N, F)) and produces a fused
    feature (B, N, F) via learned per-point scale gating.
    """

    def __init__(self, num_scales, feature_dim, hidden_dim=64):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim

        # Per-point scale gate: takes concatenated features, outputs per-scale weight
        self.scale_gate = nn.Sequential(
            nn.Conv1d(feature_dim * num_scales, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, num_scales, 1),
            nn.Softmax(dim=1),
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def execute(self, multi_scale_feats):
        # multi_scale_feats: list of (B, N, F), length = num_scales
        B, N, F = multi_scale_feats[0].shape
        S = len(multi_scale_feats)

        # Stack scales: (B, N, S, F) -> (B, S*F, N)
        flat = jt.concat([f.permute(0, 2, 1) for f in multi_scale_feats], dim=1)

        # Learned per-point scale gate: (B, S, N)
        gate = self.scale_gate(flat)

        # Weighted sum across scales
        fused = jt.zeros((B, N, F))
        for s in range(S):
            w = gate[:, s:s + 1, :].permute(0, 2, 1)  # (B, N, 1)
            fused = fused + w * multi_scale_feats[s]

        fused_out = self.out_proj(fused.permute(0, 2, 1)).permute(0, 2, 1)
        return fused_out


class CrossScaleAttention(nn.Module):
    """Cross-attention between fine and coarse scale features.

    Fine features attend to coarse features to capture multi-scale context.
    """

    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v = nn.Conv1d(channels, channels, 1)
        self.out = nn.Conv1d(channels, channels, 1)
        self.bn = nn.BatchNorm1d(channels)

    def execute(self, fine, coarse):
        # fine: (B, N, F), coarse: (B, N, F) — both after permute(0,2,1)
        B, C_f, N_f = fine.shape
        _, C_c, N_c = coarse.shape

        q = self.q(fine)   # (B, C//4, N_f)
        k = self.k(coarse) # (B, C//4, N_c)
        v = self.v(coarse) # (B, C, N_c)

        # Attention
        attn = jt.bmm(q.permute(0, 2, 1), k) / (C_f // 4) ** 0.5  # (B, N_f, N_c)
        attn = jt.softmax(attn, dim=-1)

        # Weighted sum
        out = jt.bmm(attn, v.permute(0, 2, 1))  # (B, N_f, C)
        out = self.out(out.permute(0, 2, 1))    # (B, C, N_f)
        out = self.bn(out)

        return fine + out  # Residual connection


# =============================================================================
# Vector Attention (PointNeXt-style)
# =============================================================================


class VectorAttention(nn.Module):
    """PointNeXt-style local vector attention.

    For each point, aggregates features from k-nearest neighbors using
    learned attention weights computed from (query - key) differences.
    This is a local (non-global) attention mechanism.
    """

    def __init__(self, channels, k=16, position_dim=3):
        super().__init__()
        self.k = k
        self.position_dim = position_dim

        # QKV projections
        self.fc_q = nn.Conv1d(channels, channels, 1)
        self.fc_k = nn.Conv1d(channels, channels, 1)
        self.fc_v = nn.Conv1d(channels, channels, 1)

        # Position encoding (relative position -> feature space)
        if position_dim > 0:
            self.pos_mlp = nn.Sequential(
                nn.Conv1d(position_dim, channels, 1),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Conv1d(channels, channels, 1),
            )

        # Attention scoring MLP
        self.attn_mlp = nn.Sequential(
            nn.Conv1d(channels * 2, channels // 4, 1),
            nn.BatchNorm1d(channels // 4),
            nn.ReLU(),
            nn.Conv1d(channels // 4, channels, 1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn_out = nn.BatchNorm1d(channels)

    def execute(self, x, pos=None):
        # x: (B, C, N)
        B, C, N = x.shape

        # Determine positions for kNN
        if pos is not None:
            pos_3d = pos  # (B, 3, N)
        elif self.position_dim > 0 and C >= self.position_dim:
            pos_3d = x[:, :self.position_dim, :]
        else:
            pos_3d = x[:, :3, :]

        # kNN grouping
        k = min(self.k, N - 1) if N > 1 else 1
        pos_t = pos_3d.permute(0, 2, 1)  # (B, N, 3)
        if pos_3d.shape[1] == 3:
            _, idx = jt.misc.knn(pos_t, pos_t, k + 1)
        else:
            dist = ((pos_t.unsqueeze(2) - pos_t.unsqueeze(1)) ** 2).sum(-1)
            _, idx = jt.topk(dist, k=k + 1, dim=-1, largest=False)
        idx = idx[:, :, 1:]  # Remove self: (B, N, k)

        # Q, K, V projections
        q = self.fc_q(x)  # (B, C, N)

        # Gather neighbor features
        gathered_k = []
        gathered_v = []
        gathered_pos = []
        for b in range(B):
            gathered_k.append(q.permute(0, 2, 1)[b][idx[b]])  # (N, k, C)
            gathered_v.append(self.fc_v(x).permute(0, 2, 1)[b][idx[b]])
            gathered_pos.append(pos_t[b][idx[b]])

        gathered_k = jt.stack(gathered_k, dim=0)  # (B, N, k, C)
        gathered_v = jt.stack(gathered_v, dim=0)  # (B, N, k, C)
        gathered_pos = jt.stack(gathered_pos, dim=0)  # (B, N, k, 3)

        # Position encoding
        pos_enc = jt.zeros_like(gathered_k)
        if self.position_dim > 0:
            # Relative positions: (B, N, k, 3) -> (B, 3, N*k) -> (B, C, N*k) -> (B, C, N, k) -> (B, N, k, C)
            rel_pos = gathered_pos - pos_t.unsqueeze(2)  # (B, N, k, 3)
            pos_flat = rel_pos.permute(0, 3, 1, 2).reshape(B, 3, N * k)
            pos_enc_flat = self.pos_mlp(pos_flat)  # (B, C, N*k)
            pos_enc = pos_enc_flat.reshape(B, C, N, k).permute(0, 2, 3, 1)  # (B, N, k, C)

        # Attention scores: MLP on (q - k, k) concatenation
        q_expanded = q.permute(0, 2, 1).unsqueeze(2)  # (B, N, 1, C)
        attn_input = jt.concat([q_expanded - gathered_k, gathered_k], dim=-1)  # (B, N, k, 2C)
        attn_input_flat = attn_input.permute(0, 3, 1, 2).reshape(B, 2 * C, N * k)
        attn = self.attn_mlp(attn_input_flat).reshape(B, C, N, k).permute(0, 2, 3, 1)  # (B, N, k, C)
        attn = attn + pos_enc  # Add position encoding
        attn = self.softmax(attn.permute(0, 1, 3, 2))  # (B, N, C, k)

        # Weighted sum: (B, N, C, k) @ (B, N, k, C) -> (B, N, C)
        out = jt.bmm(
            attn.reshape(B * N, C, k),
            gathered_v.reshape(B * N, k, C),
        ).reshape(B, N, C).permute(0, 2, 1)  # (B, C, N)

        # Residual + BN
        out = self.bn_out(x + out)
        return out  # (B, C, N)


class VectorAttentionEncoder(nn.Module):
    """Encoder using local vector attention instead of global PCT self-attention."""

    def __init__(
        self,
        k=16,
        input_dim=3,
        embedding_dim=512,
        attention_channels=128,
        num_attention_layers=4,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
        use_position_embedding=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.attention_channels = attention_channels
        self.num_attention_layers = num_attention_layers
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim

        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        # Input projection
        self.conv1 = nn.Conv1d(input_dim, attention_channels, 1)
        self.bn1 = nn.BatchNorm1d(attention_channels)
        self.conv2 = nn.Conv1d(attention_channels, attention_channels, 1)
        self.bn2 = nn.BatchNorm1d(attention_channels)
        self.relu = nn.ReLU()

        # Vector attention layers
        for i in range(num_attention_layers):
            va_k = min(k, max(8, k // (i + 1) + 1)) if i > 0 else k
            setattr(
                self,
                f"va{i + 1}",
                VectorAttention(
                    channels=attention_channels,
                    k=va_k,
                    position_dim=knn_coord_dim if use_position_embedding else 0,
                ),
            )

        # Feature fusion
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(attention_channels * num_attention_layers, embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(scale=0.2),
        )

    def execute(self, x):
        B, N, _ = x.shape
        x_input = x
        xyz = x[:, :, :self.knn_coord_dim]

        feat = x.permute(0, 2, 1)
        feat = self.relu(self.bn1(self.conv1(feat)))
        feat = self.relu(self.bn2(self.conv2(feat)))

        attn_feats = []
        for i in range(self.num_attention_layers):
            pos = xyz.permute(0, 2, 1) if self.knn_coord_dim > 0 else None
            feat = getattr(self, f"va{i + 1}")(feat, pos=pos)
            attn_feats.append(feat)

        feat = jt.concat(attn_feats, dim=1)
        feat = self.conv_fuse(feat).permute(0, 2, 1)

        features = [feat]
        if self.use_global_feature:
            global_feat = jt.max(feat, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, feat.shape[-1])))
        if self.use_input_coords:
            features.append(x_input[:, :, :self.knn_coord_dim])
        return jt.concat(features, dim=-1)


# =============================================================================
# U-Net Feature Hierarchy (PointNet++ style)
# =============================================================================


def _farthest_point_sample(xyz, npoint):
    """Farthest point sampling on batch of point clouds.

    xyz: (B, N, 3)
    returns: (B, npoint) indices
    """
    B, N, _ = xyz.shape
    device = xyz.device if hasattr(xyz, 'device') else None
    indices = jt.zeros((B, npoint), dtype='int32')
    distances = jt.ones((B, N)) * 1e10
    farthest = jt.zeros((B,), dtype='int32')

    for i in range(npoint):
        indices = indices.clone()
        for b in range(B):
            indices[b, i] = farthest[b]
        centroid = xyz[jt.arange(B), farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distances = jt.minimum(distances, dist)
        farthest = jt.argmax(distances, dim=-1)[0]
        for b in range(B):
            farthest[b] = jt.argmax(distances[b], dim=-1)[0]

    return indices


def _gather_points(feat, idx):
    """Gather points from feat according to idx.

    feat: (B, C, N) or (B, N, C)
    idx: (B, M)
    """
    B = feat.shape[0]
    if feat.ndim == 3 and feat.permute(0, 2, 1).shape[-1] == feat.shape[1]:
        feat = feat.permute(0, 2, 1)  # (B, N, C)
    gathered = []
    for b in range(B):
        gathered.append(feat[b][idx[b]])
    return jt.stack(gathered, dim=0)


class SetAbstraction(nn.Module):
    """PointNet++ Set Abstraction: FPS downsampling + ball/kNN grouping + PointNet."""

    def __init__(self, npoint, radius, k, in_channels, out_channels, use_xyz=True):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.k = k
        self.use_xyz = use_xyz
        mlp_in = in_channels + (3 if use_xyz else 0)
        self.mlp = nn.Sequential(
            nn.Conv1d(mlp_in, out_channels // 2, 1),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(),
            nn.Conv1d(out_channels // 2, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def execute(self, xyz, features):
        # xyz: (B, N, 3), features: (B, N, C_in)
        B, N, _ = xyz.shape
        npoint = min(self.npoint, N)

        # FPS
        if npoint < N:
            idx = _farthest_point_sample(xyz, npoint)
            new_xyz = _gather_points(xyz, idx)  # (B, npoint, 3)
        else:
            new_xyz = xyz
            idx = jt.arange(N).unsqueeze(0).broadcast((B, N))

        # kNN grouping in new_xyz
        k = min(self.k, N)
        if k > 0:
            _, knn_idx = jt.misc.knn(new_xyz, xyz, k)
            grouped_xyz = _gather_points(xyz, knn_idx)  # (B, npoint, k, 3)
            grouped_features = _gather_points(features, knn_idx)  # (B, npoint, k, C)
        else:
            grouped_xyz = new_xyz.unsqueeze(2)
            grouped_features = features.unsqueeze(2)

        # Relative position encoding
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        # Concatenate and process
        if self.use_xyz:
            grouped = jt.concat([grouped_features, grouped_xyz], dim=-1)
        else:
            grouped = grouped_features

        # (B, npoint, k, C) -> (B, C, npoint, k)
        grouped = grouped.permute(0, 3, 1, 2)
        B2, C2, np2, k2 = grouped.shape
        grouped = grouped.reshape(B2, C2, np2 * k2)
        out = self.mlp(grouped).reshape(B2, -1, np2, k2)
        out = jt.max(out, dim=-1)  # (B, C_out, npoint)
        return new_xyz, out


class FeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation: interpolation + skip connection + MLP."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def execute(self, xyz1, xyz2, points1, points2):
        # xyz1/points1: fine level (N1)
        # xyz2/points2: coarse level (N2 < N1)
        B, N1, _ = xyz1.shape
        N2 = xyz2.shape[1]
        C2 = points2.shape[1]

        # Interpolate by kNN
        k = min(3, N2)
        dist, idx = jt.misc.knn(xyz1, xyz2, k)
        dist = jt.maximum(dist, jt.array(1e-8))
        weight = 1.0 / dist
        weight = weight / (weight.sum(dim=-1, keepdims=True) + 1e-8)

        interp = jt.zeros((B, C2, N1))
        for b in range(B):
            gathered = points2[b, :, idx[b]]  # (C2, N1, k)
            w = weight[b].unsqueeze(0)  # (1, N1, k)
            interp[b] = (gathered * w).sum(dim=-1)  # (C2, N1)

        # Skip connection
        if points1 is not None:
            new_points = jt.concat([interp, points1], dim=1)
        else:
            new_points = interp

        return self.mlp(new_points)


class UNetFeatureExtraction(nn.Module):
    """U-Net encoder for point cloud denoising with SA down + FP up."""

    def __init__(
        self,
        input_dim=3,
        embedding_dim=384,
        use_global_feature=False,
        use_input_coords=False,
        knn_coord_dim=3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.use_global_feature = use_global_feature
        self.use_input_coords = use_input_coords
        self.knn_coord_dim = knn_coord_dim

        if self.use_global_feature:
            self.embedding_dim += embedding_dim
        if self.use_input_coords:
            self.embedding_dim += knn_coord_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim // 4, 1),
            nn.BatchNorm1d(embedding_dim // 4),
            nn.ReLU(),
        )

        # Encoder (SA blocks)
        self.sa1 = SetAbstraction(2048, 0.05, 16, embedding_dim // 4, embedding_dim // 2)
        self.sa2 = SetAbstraction(512, 0.1, 16, embedding_dim // 2, embedding_dim)
        self.sa3 = SetAbstraction(128, 0.2, 8, embedding_dim, embedding_dim)

        # Decoder (FP blocks)
        self.fp3 = FeaturePropagation(embedding_dim + embedding_dim, embedding_dim // 2)
        self.fp2 = FeaturePropagation(embedding_dim // 2 + embedding_dim // 2, embedding_dim // 4)
        self.fp1 = FeaturePropagation(embedding_dim // 4 + embedding_dim // 4, embedding_dim)

        self.relu = nn.ReLU()

    def execute(self, x):
        B, N, _ = x.shape
        x_input = x
        xyz = x[:, :, :3]

        # Feature extraction
        feat_extra = x[:, :, 3:] if x.shape[-1] > 3 else None

        # Initial projection
        init_feat = self.input_proj(x.permute(0, 2, 1))  # (B, E/4, N)
        init_feat = init_feat.permute(0, 2, 1)  # (B, N, E/4)

        # Encoder
        l1_xyz, l1_feat = self.sa1(xyz, init_feat)    # (B, 2048, 3), (B, E/2, 2048)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat.permute(0, 2, 1))  # (B, 512, 3), (B, E, 512)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat.permute(0, 2, 1))  # (B, 128, 3), (B, E, 128)

        # Decoder with skip connections
        l2_up = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)  # (B, E/2, 512)
        l1_up = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_up)   # (B, E/4, 2048)
        out_feat = self.fp1(xyz, l1_xyz, init_feat.permute(0, 2, 1), l1_up)  # (B, E, N)

        out_feat = out_feat.permute(0, 2, 1)  # (B, N, E)

        features = [out_feat]
        if self.use_global_feature:
            global_feat = jt.max(out_feat, dim=1, keepdims=True)
            features.append(global_feat.broadcast((B, N, out_feat.shape[-1])))
        if self.use_input_coords:
            features.append(x_input[:, :, :self.knn_coord_dim])
        return jt.concat(features, dim=-1)
