import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2

class CMT_Net(nn.Module):
    """跨链多分支图时序网络（CMT-Net）"""
    def __init__(self, input_dim, hidden_dim=64, num_nodes=10, output_dim=TARGET_DIM,
                 gcn_layers=2, short_kernel=3, long_kernel=3, long_dilation=2, att_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.att_dim = att_dim

        # -------------------------- 节点特征嵌入 --------------------------
        self.node_emb = nn.Linear(input_dim - 2, hidden_dim)  # 输入剔除节点索引
        self.relu = nn.ReLU()

        # -------------------------- 空间建模：多层GCN --------------------------
        self.gcn_layers = nn.ModuleList()
        for _ in range(gcn_layers):
            self.gcn_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # -------------------------- 短期卷积分支 --------------------------
        self.short_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=short_kernel, padding=short_kernel-1)
        self.short_relu = nn.ReLU()

        # -------------------------- 长期膨胀卷积分支 --------------------------
        self.long_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=long_kernel,
                                   padding=(long_kernel-1)*long_dilation, dilation=long_dilation)
        self.long_att_q = nn.Linear(hidden_dim, hidden_dim)
        self.long_att_k = nn.Linear(hidden_dim, hidden_dim)
        self.long_att_v = nn.Linear(hidden_dim, hidden_dim)
        self.long_relu = nn.ReLU()

        # -------------------------- 融合与输出 --------------------------
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)
        self.out_local = nn.Linear(hidden_dim, output_dim)
        self.out_cross = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj, cross_chain_idx, relay_idx):
        batch_size, seq_len, _ = x.shape

        # -------------------------- 构建节点特征矩阵 --------------------------
        spatial_feat = []
        for t in range(seq_len):
            x_t = x[:, t, :-2]  # (batch_size, input_dim-2)
            cc_idx_t = cross_chain_idx[:, t]
            r_idx_t = relay_idx[:, t]

            node_feat = torch.zeros((self.num_nodes, x_t.shape[1]), device=DEVICE)
            for i in range(batch_size):
                node_feat[cc_idx_t[i]] += x_t[i]
            for idx in torch.unique(cc_idx_t):
                count = torch.sum(cc_idx_t == idx)
                node_feat[idx] /= count

            # -------------------------- GCN层 --------------------------
            h = self.relu(self.node_emb(node_feat))
            for gcn in self.gcn_layers:
                h = self.relu(gcn(h, adj.nonzero().t()))
            spatial_feat.append(h[cc_idx_t])  # 匹配样本节点

        # (batch_size, seq_len, hidden_dim)
        spatial_feat = torch.stack(spatial_feat, dim=1)

        # -------------------------- 时序建模 --------------------------
        temporal_feat = spatial_feat.permute(0, 2, 1)  # (B, H, T)
        short_out = self.short_relu(self.short_conv(temporal_feat))[:, :, :seq_len]
        long_out = self.long_conv(temporal_feat)[:, :, :seq_len]

        # -------------------------- 长期分支轻量注意力 --------------------------
        Q = self.long_att_q(long_out.permute(0, 2, 1))  # (B, T, att_dim)
        K = self.long_att_k(long_out.permute(0, 2, 1))
        V = self.long_att_v(long_out.permute(0, 2, 1))
        att_scores = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5), dim=-1)
        long_out_att = torch.bmm(att_scores, V).permute(0, 2, 1)  # (B, att_dim, T)
        # 确保维度一致再相加
        if long_out_att.shape[1] != long_out.shape[1]:
            long_out_att = long_out_att.expand(-1, long_out.shape[1], -1)  # 扩展到与long_out相同维度
        long_out_att = long_out_att + long_out  # 残差连接

        # -------------------------- 短期+长期融合 --------------------------
        fused = torch.cat([short_out[:, :, -1], long_out_att[:, :, -1]], dim=1)
        fused = self.relu(self.fusion(fused))

        # -------------------------- 输出 --------------------------
        out_local = self.out_local(fused).unsqueeze(1)
        out_cross = self.out_cross(fused).unsqueeze(1)
        return (out_local + out_cross) / 2  # 简单融合输出
