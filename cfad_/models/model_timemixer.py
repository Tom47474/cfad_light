# models/model_timemixer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from cfad_.layers.Embed import DataEmbedding_inverted


class TimeMixerLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, mix_kernel_size=3):
        super(TimeMixerLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.mix_kernel_size = mix_kernel_size

        # 时间混合卷积
        self.time_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=mix_kernel_size, padding='same', groups=d_model)

        # 特征混合MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, D]
        residual = x

        # 时间混合
        x_permuted = x.permute(0, 2, 1)  # [B, D, T]
        x_mixed = self.time_conv(x_permuted)  # [B, D, T]
        x_mixed = x_mixed.permute(0, 2, 1)  # [B, T, D]
        x = self.norm1(x_mixed + residual)

        # 特征混合
        residual = x
        x = self.feature_mlp(x)
        x = self.norm2(x + residual)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        # TimeMixer layers
        self.mixer_layers = nn.ModuleList([
            TimeMixerLayer(configs.d_model, configs.d_ff, configs.dropout)
            for _ in range(configs.e_layers)
        ])

        # Decoder projection
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Hidden dimension for compatibility
        self.hidden_dim = configs.d_model

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means, stdev = None, None

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, D]

        # TimeMixer layers
        for layer in self.mixer_layers:
            enc_out = layer(enc_out)

        # Projection
        dec_out = self.projection(enc_out[:, -self.pred_len:, :])  # [B, L, D]

        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1))

        return dec_out, enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], enc_out
