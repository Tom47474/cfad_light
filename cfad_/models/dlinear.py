# cfad_/models/dlinear.py
import torch
import torch.nn as nn
from cfad_.layers.Embed import DataEmbedding_inverted


class DLinear(nn.Module):
    """
    DLinear模型：分解线性模型，将时间序列分解为趋势和季节性成分分别处理
    """

    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.hidden_dim = configs.d_model  # 为了兼容性

        # 两个线性层分别处理趋势和季节性成分
        self.individual = False  # 是否对每个特征使用独立的线性层
        self.channels = 1  # 特征数量

        # 趋势分解的线性层
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [Batch, Sequence Length, Channel]
        if self.use_norm:
            # 归一化
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # 分解为趋势和季节性成分
        seasonal_init, trend_init = self.decompsition(x_enc)

        # 转换维度以适应线性层: [B, C, T]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # 通过线性层
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)

        # 合并结果
        output = seasonal_output + trend_output
        output = output.permute(0, 2, 1)  # [B, P, C]

        # 反归一化
        if self.use_norm:
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return output

    def decompsition(self, x):
        """
        使用移动平均分解时间序列
        """
        # 计算趋势成分（移动平均）
        moving_mean = self.moving_avg(x, kernel_size=25)
        # 季节性成分 = 原始序列 - 趋势成分
        seasonal = x - moving_mean
        return seasonal, moving_mean

    def moving_avg(self, x, kernel_size):
        """
        计算移动平均
        """
        # 填充
        padding = (kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding, 1)
        end = x[:, -1:, :].repeat(1, padding, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # 应用平均池化
        x_padded = x_padded.permute(0, 2, 1)  # [B, C, T]
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)(x_padded)
        avg = avg.permute(0, 2, 1)  # [B, T, C]
        return avg
