import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer


class STAR_CNN(nn.Module):
    def __init__(self, d_series, d_core, kernel_size=3):
        super(STAR_CNN, self).__init__()
        """
        时序CNN
        Args:
            d_series: 输入序列的隐藏维度
            d_core: 核心表示的维度
            kernel_size: CNN卷积核大小
        """
        # 替代原gen1-gen4的MLP
        self.conv_aggregate = nn.Sequential(
            nn.Conv1d(in_channels=d_series, out_channels=d_series,
                      kernel_size=kernel_size, padding='same'),
            nn.GELU(),
            nn.Conv1d(in_channels=d_series, out_channels=d_core,
                      kernel_size=kernel_size, padding='same')
        )


        self.conv_fuse = nn.Sequential(
            nn.Conv1d(in_channels=d_series + d_core, out_channels=d_series,
                      kernel_size=1),  # 1x1卷积等价于线性层
            nn.GELU(),
            nn.Conv1d(in_channels=d_series, out_channels=d_series,
                      kernel_size=1)
        )

        # 如果输入和输出维度不匹配，添加一个1x1卷积调整维度
        self.residual_proj = nn.Conv1d(in_channels=d_series, out_channels=d_series,
                                       kernel_size=1) if d_series != d_series else None

    def forward(self, input, *args, **kwargs):
        """
        input: [B, C, d_series]
        output: [B, C, d_series]
        修改点：使用*args和**kwargs来兼容所有可能的参数传递
        """
        batch_size, channels, d_series = input.shape
        residual = input

        # --- 1. CNN聚合 ---
        # [B, C, d_series] → [B, d_series, C] (CNN需要在时间维度卷积)
        x_permuted = input.permute(0, 2, 1)  # [B, d_series, C]

        # 通过CNN聚合时间信息
        combined_mean = self.conv_aggregate(x_permuted)  # [B, d_core, C]
        combined_mean = combined_mean.permute(0, 2, 1)  # [B, C, d_core]

        # --- 2. 拼接与融合 ---
        combined_cat = torch.cat([input, combined_mean], dim=-1)  # [B, C, d_series + d_core]

        # 通过1x1 CNN融合
        output = self.conv_fuse(combined_cat.permute(0, 2, 1))  # [B, d_series, C]
        output = output.permute(0, 2, 1)  # [B, C, d_series]

        # # # 处理残差连接
        # if self.residual_proj is not None:
        #     residual = self.residual_proj(residual.permute(0, 2, 1)).permute(0, 2, 1)

        # 添加残差连接
        output = output + residual

        return output, None  # 保持与原STAR输出格式一致



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR_CNN(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
        )

        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]