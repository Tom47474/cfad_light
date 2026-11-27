# gcn_/models/bpnn.py
import torch
import torch.nn as nn


class BPNN(nn.Module):
    """BPNN (Back Propagation Neural Network) 模型"""

    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=2, dropout=0.2):
        """
        初始化BPNN模型

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（默认为2，对应延迟和吞吐量）
            dropout: Dropout比率
        """
        super(BPNN, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            输出张量，形状为 (batch_size, 1, output_dim)
        """
        # 将输入展平为 (batch_size, seq_len * input_dim)
        batch_size, seq_len, input_dim = x.shape
        x = x.view(batch_size, -1)

        # 通过神经网络
        output = self.network(x)

        # 调整输出形状为 (batch_size, 1, output_dim)
        output = output.unsqueeze(1)

        return output
