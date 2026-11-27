# gcn_/models/kan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2


class KANLinear(nn.Module):
    """KAN线性层：实现Kolmogorov-Arnold Network的基本计算单元"""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        """
        初始化KAN线性层

        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            grid_size: 网格大小
            spline_order: 样条阶数
        """
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 创建网格点
        grid_points = grid_size + 2 * spline_order + 1
        self.register_buffer("grid", torch.linspace(0, 1, grid_points))

        # 初始化样条系数
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order))

        # 初始化基础权重（线性部分）
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))

        # 初始化缩放参数
        self.scale = nn.Parameter(torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        """重置参数"""
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.xavier_uniform_(self.spline_weight)

    def b_splines(self, x):
        """
        计算B样条基函数

        Args:
            x: 输入张量，形状为 (batch_size, in_features)

        Returns:
            B样条基函数值，形状为 (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # 扩展输入维度以匹配网格
        x = x.unsqueeze(-1)  # (batch_size, in_features, 1)
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size + 2*spline_order + 1)

        # 计算0阶B样条
        b_splines = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()

        # 递归计算高阶B样条
        for k in range(1, self.spline_order + 1):
            # 左项计算
            denominator1 = grid[:, :, k:-1] - grid[:, :, :-k - 1]
            # 避免除零错误
            denominator1 = torch.where(denominator1 == 0, torch.ones_like(denominator1), denominator1)
            left_term = torch.where(
                denominator1 != 0,
                (x - grid[:, :, :-k - 1]) / denominator1 * b_splines[:, :, :-1],
                torch.zeros_like(b_splines[:, :, :-1])
            )

            # 右项计算
            denominator2 = grid[:, :, k + 1:] - grid[:, :, 1:-k]
            # 避免除零错误
            denominator2 = torch.where(denominator2 == 0, torch.ones_like(denominator2), denominator2)
            right_term = torch.where(
                denominator2 != 0,
                (grid[:, :, k + 1:] - x) / denominator2 * b_splines[:, :, 1:],
                torch.zeros_like(b_splines[:, :, 1:])
            )

            # 合并左右项
            b_splines = left_term + right_term

        return b_splines.contiguous()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, in_features)

        Returns:
            输出张量，形状为 (batch_size, out_features)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # 基础线性变换
        base_output = F.linear(x, self.base_weight)

        # 样条变换
        b_splines = self.b_splines(x)  # (batch_size, in_features, grid_size + spline_order)
        spline_output = torch.einsum('bik,oik->bo', b_splines, self.spline_weight)

        # 组合输出
        output = base_output + self.scale * spline_output

        return output


class KANModel(nn.Module):
    """Kolmogorov-Arnold Network模型用于跨链性能预测"""

    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=TARGET_DIM,
                 grid_size=5, spline_order=3, dropout=0.1):
        """
        初始化KAN模型

        Args:
            input_dim: 输入特征维度 (seq_len * feature_dim)
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            grid_size: 网格大小
            spline_order: 样条阶数
            dropout: Dropout比率
        """
        super(KANModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # 构建网络层
        layers = []
        prev_dim = input_dim

        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(KANLinear(prev_dim, hidden_dim, grid_size, spline_order))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层
        layers.append(KANLinear(prev_dim, output_dim, grid_size, spline_order))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, feature_dim)

        Returns:
            输出张量，形状为 (batch_size, 1, output_dim)
        """
        # 将输入展平为 (batch_size, seq_len * feature_dim)
        batch_size, seq_len, feature_dim = x.shape
        x = x.view(batch_size, -1)

        # 通过KAN网络
        output = self.network(x)

        # 调整输出形状为 (batch_size, 1, output_dim)
        output = output.unsqueeze(1)

        return output
