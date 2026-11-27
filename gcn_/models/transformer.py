# gcn_/models/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    """基于Transformer的跨链性能预测模型"""

    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=1, dim_feedforward=256, dropout=0.1, output_dim=TARGET_DIM):
        """
        初始化Transformer模型

        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 多头注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            output_dim: 输出维度
        """
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # 输入形状: (seq_len, batch_size, d_model)
        )

        # 输出投影层
        self.output_projection = nn.Linear(d_model, output_dim)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()

    def generate_square_subsequent_mask(self, sz):
        """生成后续位置掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(DEVICE)

    def forward(self, src):
        """
        前向传播

        Args:
            src: 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            输出张量，形状为 (batch_size, 1, output_dim)
        """
        # 调整维度: (seq_len, batch_size, input_dim)
        src = src.permute(1, 0, 2)

        # 输入投影到d_model维度
        src = self.input_projection(src) * math.sqrt(self.d_model)

        # 添加位置编码
        src = self.pos_encoder(src)

        # 创建目标序列（用于预测最后一个时间步）
        tgt = src[-1:, :, :]  # 取最后一个时间步作为目标输入

        # 生成掩码
        src_mask = self.generate_square_subsequent_mask(src.size(0))
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))

        # Transformer前向传播
        # 注意：这里我们使用最后一个时间步作为目标输入来预测下一个时间步
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # 输出投影
        output = self.output_projection(output)

        # 调整输出形状为 (batch_size, 1, output_dim)
        output = output.permute(1, 0, 2)

        return output
