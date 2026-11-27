# cfad_/models/dlinear_large.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinear(nn.Module):
    """
    大型DLinear模型：增加参数量以符合实际应用需求
    """

    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.channels = 1  # 特征数量
        self.decomp_kernel = 25
        self.hidden_dim = configs.d_model

        # 增加模型复杂度的参数
        self.large_hidden_dim = 512  # 增加隐藏层维度
        self.num_experts = 4  # MoE experts数量
        self.top_k = 2  # MoE top-k experts

        # 趋势和季节性分解的复杂线性层
        self.linear_trend = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_len, self.large_hidden_dim),
                nn.GELU(),
                nn.Linear(self.large_hidden_dim, self.large_hidden_dim),
                nn.GELU(),
                nn.Linear(self.large_hidden_dim, self.pred_len)
            ) for _ in range(2)  # 两个并行路径
        ])

        self.linear_seasonal = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_len, self.large_hidden_dim),
                nn.GELU(),
                nn.Linear(self.large_hidden_dim, self.large_hidden_dim),
                nn.GELU(),
                nn.Linear(self.large_hidden_dim, self.pred_len)
            ) for _ in range(2)  # 两个并行路径
        ])

        # 专家混合(MoE)层
        self.experts_trend = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.pred_len, self.large_hidden_dim),
                nn.GELU(),
                nn.Linear(self.large_hidden_dim, self.large_hidden_dim),
                nn.GELU(),
                nn.Linear(self.large_hidden_dim, self.pred_len)
            ) for _ in range(self.num_experts)
        ])

        self.experts_seasonal = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.pred_len, self.large_hidden_dim),
                nn.GELU(),
                nn.Linear(self.large_hidden_dim, self.large_hidden_dim),
                nn.GELU(),
                nn.Linear(self.large_hidden_dim, self.pred_len)
            ) for _ in range(self.num_experts)
        ])

        # 门控网络
        self.gate_trend = nn.Sequential(
            nn.Linear(self.pred_len, self.large_hidden_dim),
            nn.GELU(),
            nn.Linear(self.large_hidden_dim, self.num_experts)
        )

        self.gate_seasonal = nn.Sequential(
            nn.Linear(self.pred_len, self.large_hidden_dim),
            nn.GELU(),
            nn.Linear(self.large_hidden_dim, self.num_experts)
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.pred_len * 2, self.large_hidden_dim),
            nn.GELU(),
            nn.Linear(self.large_hidden_dim, self.large_hidden_dim),
            nn.GELU(),
            nn.Linear(self.large_hidden_dim, self.pred_len)
        )

        # 时间特征嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(2, self.large_hidden_dim),  # 时间标记特征
            nn.GELU(),
            nn.Linear(self.large_hidden_dim, self.large_hidden_dim),
            nn.GELU(),
            nn.Linear(self.large_hidden_dim, self.pred_len)
        )

        # Layer Normalization
        self.norm_trend = nn.LayerNorm(self.pred_len)
        self.norm_seasonal = nn.LayerNorm(self.pred_len)

        # Dropout
        self.dropout = nn.Dropout(0.1)

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

        # 通过多个并行路径处理趋势成分
        trend_outputs = []
        for layer in self.linear_trend:
            trend_outputs.append(layer(trend_init))
        trend_combined = torch.stack(trend_outputs, dim=-1).mean(dim=-1)

        # 通过多个并行路径处理季节性成分
        seasonal_outputs = []
        for layer in self.linear_seasonal:
            seasonal_outputs.append(layer(seasonal_init))
        seasonal_combined = torch.stack(seasonal_outputs, dim=-1).mean(dim=-1)

        # 应用MoE到趋势成分
        trend_moe = self.moe_forward(trend_combined, self.experts_trend, self.gate_trend)

        # 应用MoE到季节性成分
        seasonal_moe = self.moe_forward(seasonal_combined, self.experts_seasonal, self.gate_seasonal)

        # 正则化
        trend_moe = self.norm_trend(trend_moe)
        seasonal_moe = self.norm_seasonal(seasonal_moe)

        # Dropout
        trend_moe = self.dropout(trend_moe)
        seasonal_moe = self.dropout(seasonal_moe)

        # 融合趋势和季节性成分
        combined = torch.cat([trend_moe, seasonal_moe], dim=-1)
        output = self.fusion_layer(combined)

        # 添加时间特征信息
        if x_mark_dec is not None:
            time_features = self.time_embedding(x_mark_dec.mean(dim=1))  # [B, pred_len]
            output = output + time_features.unsqueeze(1)  # [B, 1, pred_len]

        output = output.permute(0, 2, 1)  # [B, P, C]

        # 反归一化
        if self.use_norm:
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return output

    def moe_forward(self, x, experts, gate):
        """
        专家混合前向传播
        """
        # 计算门控权重
        gate_logits = gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # 应用专家
        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = experts[i](x)
            # 根据top-k索引选择权重
            expert_weight = (top_k_indices == i).float() * top_k_weights
            expert_weight = expert_weight.sum(dim=-1, keepdim=True)
            expert_outputs.append(expert_output * expert_weight)

        return torch.stack(expert_outputs, dim=-1).sum(dim=-1)

    def decompsition(self, x):
        """
        使用移动平均分解时间序列
        """
        # 计算趋势成分（移动平均）
        moving_mean = self.moving_avg(x, self.decomp_kernel)
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
