import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from cfad_.layers.Embed import DataEmbedding_inverted
from cfad_.layers.Transformer_EncDec import Encoder, EncoderLayer

# ============================
# STAR_CNN: 时序卷积模块（保持兼容）
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
        # input: [B, C, T]
        batch_size, channels, d_series = input.shape
        residual = input

        x_permuted = input.permute(0, 2, 1)
        combined_mean = self.conv_aggregate(x_permuted)
        combined_mean = combined_mean.permute(0, 2, 1)

        combined_cat = torch.cat([input, combined_mean], dim=-1)
        output = self.conv_fuse(combined_cat.permute(0, 2, 1)).permute(0, 2, 1)

        output = output + residual
        return output, None  # 保持和原实现一致的返回格式


# ============================
# CFAD 基础模型（改动：返回 enc_out 以支持中间层蒸馏）
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
            ) for _ in range(configs.e_layers)
        ])

        # Projection for decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # 暴露隐藏维度，方便创建 hidden projection
        self.hidden_dim = configs.d_model

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means, stdev = None, None

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, D]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [B, T, D]

        # 修改后的预测逻辑
        dec_out = self.projection(enc_out[:, -self.pred_len:, :])  # [B, L, D]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1))

        return dec_out, enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # 保持外部接口可兼容：返回 (pred, enc_out)
        return dec_out[:, -self.pred_len:, :], enc_out


# ============================
# Teacher / Student
# ============================
class CFAD_Teacher(CFAD_Base):
    def __init__(self, configs):
        super(CFAD_Teacher, self).__init__(configs)
        self.name = "Teacher"


class CFAD_Student(CFAD_Base):
    def __init__(self, configs):
        # 拷贝 configs，使用保守缩放策略（避免 d_model 太小导致 GPU 利用率下降）
        configs_student = copy.deepcopy(configs)
        scale = 0.7  # 推荐 0.6~0.8 之间尝试
        configs_student.d_model = max(64, int(configs.d_model * scale))
        configs_student.d_core = max(8, int(configs.d_core * scale))
        configs_student.d_ff = max(64, int(configs.d_ff * scale))
        configs_student.e_layers = max(2, configs.e_layers // 2)
        super(CFAD_Student, self).__init__(configs_student)
        self.name = "Student"


# ============================
# Distillation loss（温度 + 隐层蒸馏）
# 返回 (total_loss, task_loss, kd_loss, hidden_loss)
# ============================
class HiddenProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


def distillation_loss(student_out, teacher_out, y_true,
                      student_hidden: Optional[torch.Tensor] = None,
                      teacher_hidden: Optional[torch.Tensor] = None,
                      hidden_proj: Optional[nn.Module] = None,
                      alpha: float = 0.5,
                      beta: float = 0.3,
                      T: float = 2.0):
    """
    student_out: [B, L, D]
    teacher_out: [B, L, D]
    y_true: [B, L, D]
    student_hidden: [B, T, D_s]
    teacher_hidden: [B, T, D_t]
    hidden_proj: nn.Module to project teacher_hidden -> student_hidden dim (optional)
    """
    # 任务损失
    task_loss = F.mse_loss(student_out, y_true)

    # 蒸馏预测损失（温度缩放）
    if T != 1.0:
        kd_loss = F.mse_loss(student_out / T, teacher_out.detach() / T)
    else:
        kd_loss = F.mse_loss(student_out, teacher_out.detach())

    total = alpha * task_loss + (1 - alpha) * kd_loss
    hidden_loss = None

    # 中间层蒸馏（可选）
    if (student_hidden is not None) and (teacher_hidden is not None):
        # teacher_hidden 维度投影到 student_hidden
        if hidden_proj is not None:
            teacher_proj = hidden_proj(teacher_hidden.detach())
            hidden_loss = F.mse_loss(student_hidden, teacher_proj)
        else:
            if student_hidden.shape == teacher_hidden.shape:
                hidden_loss = F.mse_loss(student_hidden, teacher_hidden.detach())
            else:
                hidden_loss = None  # 如果维度不匹配且没有投影层则跳过

        if hidden_loss is not None:
            total = total + beta * hidden_loss

    return total, task_loss, kd_loss, hidden_loss
