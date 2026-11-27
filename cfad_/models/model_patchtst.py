# models/model_patchtst.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from cfad_.layers.Embed import DataEmbedding_inverted


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PatchTST_backbone(nn.Module):
    def __init__(self, configs, patch_len=12, stride=8):
        super(PatchTST_backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = int((configs.seq_len - patch_len) / stride + 1)

        # Patching
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.patch_num += 1

        # Embedding
        self.value_embedding = nn.Linear(patch_len, configs.d_model)
        self.position_embedding = PositionalEncoding(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=getattr(configs, 'n_heads', 8),
            dim_feedforward=getattr(configs, 'd_ff', 2048),
            dropout=configs.dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=getattr(configs, 'e_layers', 3)
        )

        # Head
        self.head_nf = self.patch_num * configs.d_model
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.head_nf, configs.pred_len)
        )

    def forward(self, x_enc, x_mark_enc):
        # x_enc: [Batch, Input length, Channel]
        x_enc = x_enc.permute(0, 2, 1)  # [Batch, Channel, Input length]

        # Padding
        x_enc = self.padding_patch_layer(x_enc)

        # Patching
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_enc = x_enc.permute(0, 2, 1, 3)  # [Batch, Patch num, Channel, Patch len]

        # Flatten and embedding
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[2], x_enc.shape[1], x_enc.shape[3])
        x_enc = self.value_embedding(x_enc)  # [Batch * Channel, Patch num, D_model]
        x_enc = self.position_embedding(x_enc)
        x_enc = self.dropout(x_enc)

        # Encoder
        x_enc = self.encoder(x_enc)  # [Batch * Channel, Patch num, D_model]

        # Head
        x_enc = self.head(x_enc)  # [Batch * Channel, Pred len]
        x_enc = x_enc.reshape(-1, self.pred_len)
        x_enc = x_enc.reshape(-1, self.pred_len, 1)  # [Batch, Pred len, Channel]

        return x_enc


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Model
        self.model = PatchTST_backbone(configs)

        # Hidden dimension for compatibility
        self.hidden_dim = configs.d_model

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        dec_out = self.model(x_enc, x_mark_enc)
        dec_out = dec_out[:, -self.pred_len:, :]

        # De-Normalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
