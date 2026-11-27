import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, q, k, v):
        batch_size, seq_len, _ = q.size()

        # Linear projections
        q = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # ProbSparse attention (simplified version)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return output


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # ProbSparse self-attention
        src2 = self.attention(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.cross_attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # Self-attention
        tgt2 = self.self_attention(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2 = self.cross_attention(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward network
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Informer(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.dropout = dropout

        # Informer parameters
        self.nhead = 8  # number of attention heads
        self.dim_feedforward = 2048  # feedforward network dimension

        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(hidden_size, self.nhead, self.dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            InformerDecoderLayer(hidden_size, self.nhead, self.dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, src, tgt):
        src = src.to(self.device)  # [batch_size, src_seq_len, input_size]
        tgt = tgt.to(self.device)  # [batch_size, tgt_seq_len, input_size]

        # Embed inputs and add positional encoding
        src = self.embedding(src)  # [batch_size, src_seq_len, hidden_size]
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt)  # [batch_size, tgt_seq_len, hidden_size]
        tgt = self.pos_encoder(tgt)

        # Encoder processing
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory)

        # Decoder processing
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory)

        # Get last timestep output and apply FC layer
        output_last = output[:, -1, :]  # [batch_size, hidden_size]
        out = self.fc(output_last)  # [batch_size, 1]
        out = out.unsqueeze(1)  # [batch_size, 1, 1]

        return out