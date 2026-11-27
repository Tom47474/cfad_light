# gcn_/models/stsgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2


class STSGCNLayer(nn.Module):
    """简化版STSGCN层：同时处理空间和时间依赖关系"""

    def __init__(self, input_dim, output_dim, vertex_num):
        super(STSGCNLayer, self).__init__()
        self.vertex_num = vertex_num
        self.output_dim = output_dim

        # 时间卷积 (处理时序信息)
        self.temporal_conv = nn.Conv2d(
            input_dim, output_dim, kernel_size=(1, 3), padding=(0, 1)
        )

        # 空间卷积 (处理图结构信息)
        self.spatial_conv = nn.Conv2d(
            output_dim, output_dim, kernel_size=(1, 1)
        )

        # 图卷积参数
        self.theta = nn.Parameter(torch.FloatTensor(output_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)

    def normalize_adj(self, adj):
        """对称归一化邻接矩阵"""
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        degree_mat_inv_sqrt = torch.diag(degree_inv_sqrt)
        return torch.mm(torch.mm(degree_mat_inv_sqrt, adj), degree_mat_inv_sqrt)

    def forward(self, x, adj):
        """
        x: (batch_size, input_dim, vertex_num, time_steps)
        adj: (vertex_num, vertex_num) 邻接矩阵
        """
        # 时间卷积
        x = self.temporal_conv(x)  # (batch_size, output_dim, vertex_num, time_steps)

        # 空间卷积
        x = self.spatial_conv(x)  # (batch_size, output_dim, vertex_num, time_steps)

        # 图卷积
        batch_size, _, vertex_num, time_steps = x.shape
        x = x.permute(0, 3, 2, 1)  # (batch_size, time_steps, vertex_num, output_dim)
        x = x.reshape(batch_size * time_steps, vertex_num, -1)  # (batch_size*time_steps, vertex_num, output_dim)

        # 应用图卷积
        adj_normalized = self.normalize_adj(adj)
        x = torch.bmm(adj_normalized.unsqueeze(0).expand(batch_size * time_steps, -1, -1), x)
        x = torch.matmul(x, self.theta)

        # 恢复形状
        x = x.reshape(batch_size, time_steps, vertex_num, -1)  # (batch_size, time_steps, vertex_num, output_dim)
        x = x.permute(0, 3, 2, 1)  # (batch_size, output_dim, vertex_num, time_steps)

        return x


class STSGCN(nn.Module):
    """简化版Spatio-Temporal Synchronous Graph Convolutional Network"""

    def __init__(self, input_dim, hidden_dims, vertex_num, output_dim, time_steps, dropout=0.1):
        super(STSGCN, self).__init__()
        self.vertex_num = vertex_num
        self.time_steps = time_steps
        self.hidden_dims = hidden_dims

        # 输入投影层，将特征维度转换为适合图结构的表示
        self.input_projection = nn.Linear(input_dim, vertex_num)

        # 多层STSGCN
        self.stsgcn_layers = nn.ModuleList()
        dims = [1] + hidden_dims  # 从1个通道开始

        for i in range(len(dims) - 1):
            self.stsgcn_layers.append(
                STSGCNLayer(dims[i], dims[i + 1], vertex_num)
            )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x, adj):
        """
        x: (batch_size, time_steps, feature_dim)
        adj: (vertex_num, vertex_num) 邻接矩阵
        """
        batch_size = x.size(0)

        # 投影输入特征到节点维度
        x = self.input_projection(x)  # (batch_size, time_steps, vertex_num)

        # 调整维度: (batch_size, 1, vertex_num, time_steps)
        x = x.permute(0, 2, 1).unsqueeze(1)  # (batch_size, 1, vertex_num, time_steps)

        # 应用STSGCN层
        for layer in self.stsgcn_layers:
            x = layer(x, adj)
            x = F.relu(x)
            x = self.dropout(x)

        # 全局平均池化: (batch_size, hidden_dims[-1], 1, 1)
        x = self.global_pool(x)
        x = x.view(batch_size, -1)  # (batch_size, hidden_dims[-1])

        # 输出层
        out = self.output_layer(x)  # (batch_size, output_dim)
        out = out.unsqueeze(1)  # (batch_size, 1, output_dim)

        return out
