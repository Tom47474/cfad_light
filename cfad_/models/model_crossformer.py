# models/model_crossformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from cfad_.layers.Embed import DataEmbedding_inverted


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None):
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

        self.W_Q = nn.Linear(d_model, d_keys * n_heads)
        self.W_K = nn.Linear(d_model, d_keys * n_heads)
        self.W_V = nn.Linear(d_model, d_values * n_heads)
        self.W_O = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values):
        B, L, D = queries.shape
        _, S, _ = keys.shape

        Q = self.W_Q(queries).view(B, L, self.n_heads, self.d_keys).transpose(1, 2)
        K = self.W_K(keys).view(B, S, self.n_heads, self.d_keys).transpose(1, 2)
        V = self.W_V(values).view(B, S, self.n_heads, self.d_values).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_keys)
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.d_values)
        output = self.W_O(context)

        return output


class CrossformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(CrossformerBlock, self).__init__()

        self.attention = CrossAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        attn_out = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


class Crossformer(nn.Module):
    def __init__(self, configs):
        super(Crossformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        # 使用配置中已有的参数，如果没有则使用默认值
        n_heads = getattr(configs, 'n_heads', 8)  # 默认8个注意力头
        d_ff = getattr(configs, 'd_ff', 2048)  # 默认前馈网络维度
        e_layers = getattr(configs, 'e_layers', 2)  # 默认2层

        # Crossformer blocks
        self.layers = nn.ModuleList([
            CrossformerBlock(configs.d_model, n_heads, d_ff, configs.dropout)
            for _ in range(e_layers)
        ])

        # Decoder projection
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Hidden dimension for compatibility
        self.hidden_dim = configs.d_model

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means, stdev = None, None

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, D]

        # Crossformer blocks
        for layer in self.layers:
            enc_out = layer(enc_out)

        # Projection
        dec_out = self.projection(enc_out[:, -self.pred_len:, :])  # [B, L, D]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1))

        return dec_out, enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], enc_out
