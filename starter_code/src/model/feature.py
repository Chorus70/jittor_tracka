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
