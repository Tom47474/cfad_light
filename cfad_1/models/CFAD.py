# cfad_1/models/CFAD.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer

# ============================
# STAR_CNN: 时序卷积模块
# ============================
class STAR_CNN(nn.Module):
    def __init__(self, d_series, d_core, kernel_size=3):
        super(STAR_CNN, self).__init__()
        self.conv_aggregate = nn.Sequential(
            nn.Conv1d(in_channels=d_series, out_channels=d_series,
                      kernel_size=kernel_size, padding='same'),
            nn.GELU(),
            nn.Conv1d(in_channels=d_series, out_channels=d_core,
                      kernel_size=kernel_size, padding='same')
        )

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(in_channels=d_series + d_core, out_channels=d_series,
                      kernel_size=1),
            nn.GELU(),
            nn.Conv1d(in_channels=d_series, out_channels=d_series, kernel_size=1)
        )

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape
        residual = input

        x_permuted = input.permute(0, 2, 1)
        combined_mean = self.conv_aggregate(x_permuted)
        combined_mean = combined_mean.permute(0, 2, 1)

        combined_cat = torch.cat([input, combined_mean], dim=-1)
        output = self.conv_fuse(combined_cat.permute(0, 2, 1)).permute(0, 2, 1)

        output = output + residual
        return output, None  # 与原STAR_CNN保持输出格式一致

# ============================
# CFAD 基础模型
# ============================
class CFAD_Base(nn.Module):
    def __init__(self, configs):
        super(CFAD_Base, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder([
            EncoderLayer(
                STAR_CNN(configs.d_model, configs.d_core),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for l in range(configs.e_layers)
        ])

        # Projection for decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

# ============================
# Teacher 模型（完整模型）
# ============================
class CFAD_Teacher(CFAD_Base):
    def __init__(self, configs):
        super(CFAD_Teacher, self).__init__(configs)
        # Teacher 保留完整参数
        self.name = "Teacher"

# ============================
# Student 模型（轻量化）
# ============================
class CFAD_Student(CFAD_Base):
    def __init__(self, configs):
        # Student 模型层数减半，特征维度可缩小
        configs_student = configs
        configs_student.e_layers = max(1, configs.e_layers // 2)
        configs_student.d_model = max(16, configs.d_model // 2)
        configs_student.d_core = max(8, configs.d_core // 2)
        configs_student.d_ff = max(32, configs.d_ff // 2)
        super(CFAD_Student, self).__init__(configs_student)
        self.name = "Student"

# ============================
# 知识蒸馏训练函数
# ============================
def distillation_loss(student_out, teacher_out, y_true, alpha=0.5):
    """
    student_out: [B, L, D]
    teacher_out: [B, L, D]
    y_true: [B, L, D]
    """
    task_loss = F.mse_loss(student_out, y_true)
    kd_loss = F.mse_loss(student_out, teacher_out.detach())
    total_loss = alpha * task_loss + (1 - alpha) * kd_loss
    return total_loss
