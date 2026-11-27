# gcn_/models/autost.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, node_features, adj):
        # node_features: (batch_size, num_nodes, hidden_dim)
        # adj: (num_nodes, num_nodes)
        batch_size, num_nodes, hidden_dim = node_features.shape

        # 创建邻接矩阵掩码
        mask = adj.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_nodes, num_nodes)

        # 计算注意力分数
        # 将每个节点与其邻居节点拼接
        node_expanded = node_features.unsqueeze(2).expand(-1, -1, num_nodes,
                                                          -1)  # (batch_size, num_nodes, num_nodes, hidden_dim)
        neighbors_expanded = node_features.unsqueeze(1).expand(-1, num_nodes, -1,
                                                               -1)  # (batch_size, num_nodes, num_nodes, hidden_dim)

        # 拼接节点特征
        concat_features = torch.cat([node_expanded, neighbors_expanded],
                                    dim=-1)  # (batch_size, num_nodes, num_nodes, hidden_dim*2)

        # 计算注意力分数
        attention_scores = self.attention_linear(concat_features).squeeze(-1)  # (batch_size, num_nodes, num_nodes)

        # 应用邻接矩阵掩码
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_nodes, num_nodes)

        # 加权聚合邻居信息
        aggregated = torch.bmm(attention_weights, node_features)  # (batch_size, num_nodes, hidden_dim)

        return aggregated


class TemporalAttention(nn.Module):
    """时间注意力机制"""

    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, time_features):
        # time_features: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = time_features.shape

        # 计算时间注意力分数
        # 将每个时间步与其他时间步拼接
        time_expanded = time_features.unsqueeze(2).expand(-1, -1, seq_len,
                                                          -1)  # (batch_size, seq_len, seq_len, hidden_dim)
        others_expanded = time_features.unsqueeze(1).expand(-1, seq_len, -1,
                                                            -1)  # (batch_size, seq_len, seq_len, hidden_dim)

        # 拼接时间特征
        concat_features = torch.cat([time_expanded, others_expanded],
                                    dim=-1)  # (batch_size, seq_len, seq_len, hidden_dim*2)

        # 计算注意力分数
        attention_scores = self.attention_linear(concat_features).squeeze(-1)  # (batch_size, seq_len, seq_len)

        # 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # 加权聚合时间信息
        aggregated = torch.bmm(attention_weights, time_features)  # (batch_size, seq_len, hidden_dim)

        return aggregated


class SpatialTemporalBlock(nn.Module):
    """时空块"""

    def __init__(self, hidden_dim, num_nodes):
        super(SpatialTemporalBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # 空间和时间注意力
        self.spatial_attn = SpatialAttention(hidden_dim)
        self.temporal_attn = TemporalAttention(hidden_dim)

        # LayerNorm用于稳定训练
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # 残差连接的线性变换
        self.spatial_transform = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj):
        # x: (batch_size, seq_len, num_nodes, hidden_dim)
        batch_size, seq_len, num_nodes, hidden_dim = x.shape

        # 空间建模 - 对每个时间步进行空间注意力
        spatial_outputs = []
        for t in range(seq_len):
            node_features = x[:, t, :, :]  # (batch_size, num_nodes, hidden_dim)
            spatial_agg = self.spatial_attn(node_features, adj)  # (batch_size, num_nodes, hidden_dim)
            # 残差连接和LayerNorm
            spatial_out = self.spatial_norm(spatial_agg + self.spatial_transform(node_features))
            spatial_outputs.append(spatial_out)

        spatial_features = torch.stack(spatial_outputs, dim=1)  # (batch_size, seq_len, num_nodes, hidden_dim)

        # 时间建模 - 对每个节点进行时间注意力
        temporal_outputs = []
        for i in range(num_nodes):
            time_features = spatial_features[:, :, i, :]  # (batch_size, seq_len, hidden_dim)
            temporal_agg = self.temporal_attn(time_features)  # (batch_size, seq_len, hidden_dim)
            # 残差连接和LayerNorm
            temporal_out = self.temporal_norm(temporal_agg + self.temporal_transform(time_features))
            temporal_outputs.append(temporal_out)

        temporal_features = torch.stack(temporal_outputs, dim=2)  # (batch_size, seq_len, num_nodes, hidden_dim)

        return temporal_features


class AutoST(nn.Module):
    """优化版AutoST模型"""

    def __init__(self, input_dim, hidden_dim=64, num_nodes=10, output_dim=TARGET_DIM,
                 num_layers=2, seq_len=12):
        super(AutoST, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # 特征投影
        self.feature_projection = nn.Linear(input_dim, hidden_dim)

        # 多层时空块
        self.st_blocks = nn.ModuleList([
            SpatialTemporalBlock(hidden_dim, num_nodes)
            for _ in range(num_layers)
        ])

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, hidden_dim))

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, adj=None):
        # x: (batch_size, seq_len, input_dim)
        # adj: (num_nodes, num_nodes) 邻接矩阵，如果未提供则使用默认的全连接图

        batch_size, seq_len, input_dim = x.shape

        # 如果没有提供邻接矩阵，创建一个全连接图
        if adj is None:
            adj = torch.ones(self.num_nodes, self.num_nodes, device=x.device)

        # 特征投影
        x = self.feature_projection(x)  # (batch_size, seq_len, hidden_dim)

        # 为适应空间建模，需要将序列特征分配给节点
        # 简化处理：将特征复制到所有节点
        node_features = x.unsqueeze(2).repeat(1, 1, self.num_nodes, 1)  # (batch_size, seq_len, num_nodes, hidden_dim)

        # 通过多层时空块处理
        h = node_features
        for block in self.st_blocks:
            h = block(h, adj)  # (batch_size, seq_len, num_nodes, hidden_dim)

        # 全局池化：在序列和节点维度上进行池化
        # 首先在节点维度上池化
        h = h.mean(dim=2)  # (batch_size, seq_len, hidden_dim)
        # 然后在序列维度上池化
        h = h.mean(dim=1)  # (batch_size, hidden_dim)

        # 输出层
        output = self.output_layer(h)  # (batch_size, output_dim)
        output = output.unsqueeze(1)  # (batch_size, 1, output_dim)

        return output
