# models/model_itransformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from cfad_.layers.Embed import DataEmbedding_inverted


class InvertibleLinear(nn.Module):
    def __init__(self, d_model):
        super(InvertibleLinear, self).__init__()
        self.d_model = d_model
        # 可逆变换矩阵
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, N, D]
        return self.linear(x)


class iTransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(iTransformerBlock, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 可逆变换层
        self.invertible_linear = InvertibleLinear(d_model)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, N, D]
        residual = x
        x = self.invertible_linear(x)
        x = self.norm1(x + residual)

        residual = x
        x = self.ffn(x)
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

        # Encoder layers
        self.encoder = nn.ModuleList([
            iTransformerBlock(configs.d_model, configs.d_ff, configs.dropout)
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
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, N, D]

        # Encoder
        for layer in self.encoder:
            enc_out = layer(enc_out)

        # Projection
        dec_out = self.projection(enc_out)  # [B, N, pred_len]

        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1))

        return dec_out, enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], enc_out
