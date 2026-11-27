# gcn_/models/gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        # 确保邻接矩阵维度与注意力权重匹配
        if adj.dim() == 2 and e.dim() == 1:
            # 如果e是一维的，我们需要reshape它
            N = Wh.size(0)
            e = e.view(N, N)
        elif adj.dim() == 2 and e.dim() == 2:
            # 确保维度匹配
            min_size = min(adj.size(0), adj.size(1), e.size(0), e.size(1))
            adj = adj[:min_size, :min_size]
            e = e[:min_size, :min_size]
        else:
            # 处理其他情况，确保维度兼容
            size = min(adj.size(0), e.size(0) if e.dim() > 0 else adj.size(0))
            adj = adj[:size, :size]
            if e.dim() == 1:
                e = e[:size * size].view(size, size)
            else:
                e = e[:size, :size]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh[:attention.size(0)])

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # 限制最大节点数以避免内存问题
        N = min(N, 50)

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh[:N].repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh[:N].repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return self.leakyrelu(torch.matmul(all_combinations_matrix, self.a).squeeze(1))

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, output_dim=TARGET_DIM, dropout=0.6, alpha=0.2, nheads=8):
        """Version of GAT that works with cross-chain data."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(hidden_dim * nheads, output_dim, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # 平均池化序列维度，将时序信息压缩为图节点特征
        x_pooled = torch.mean(x, dim=1)  # (batch_size, input_dim)

        # 处理每个样本
        outputs = []
        for i in range(batch_size):
            # 获取当前样本的特征
            sample_features = x_pooled[i]  # (input_dim)

            # 限制节点数量以避免内存问题
            max_nodes = min(adj.size(0), 50)
            adj_truncated = adj[:max_nodes, :max_nodes]

            # 扩展特征以匹配邻接矩阵大小（将单个特征复制到所有节点）
            expanded_features = sample_features.unsqueeze(0).repeat(max_nodes, 1)  # (max_nodes, input_dim)

            # 应用多头注意力
            x_att = torch.cat([att(expanded_features, adj_truncated) for att in self.attentions], dim=1)
            x_att = F.dropout(x_att, self.dropout, training=self.training)
            x_att = F.elu(self.out_att(x_att, adj_truncated))

            # 全局平均池化获取图级表示
            graph_embedding = torch.mean(x_att, dim=0)  # (output_dim)
            outputs.append(graph_embedding)

        # 组合所有输出
        output = torch.stack(outputs, dim=0)  # (batch_size, output_dim)
        output = output.unsqueeze(1)  # (batch_size, 1, output_dim)

        return output
