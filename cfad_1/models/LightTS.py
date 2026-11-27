import torch
import torch.nn as nn
import math


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


class LightTS(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.dropout = dropout

        # LightTS parameters
        self.nhead = 8  # number of attention heads
        self.dim_feedforward = 2048  # feedforward network dimension

        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        # LightTS encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # LightTS-specific components
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            padding_mode='replicate'
        )  # Temporal convolution for local pattern extraction
        self.norm = nn.LayerNorm(hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        src = batch.to(self.device)  # [batch_size, seq_len, input_size]
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, input_size]

        # Embed input and add positional encoding
        src = self.embedding(src)  # [seq_len, batch_size, hidden_size]
        src = self.pos_encoder(src)  # add positional encoding

        # Temporal convolution for local pattern extraction
        src = src.permute(1, 2, 0)  # [batch_size, hidden_size, seq_len]
        src = self.temporal_conv(src)  # [batch_size, hidden_size, seq_len]
        src = src.permute(2, 0, 1)  # [seq_len, batch_size, hidden_size]
        src = self.norm(src)  # Layer normalization

        # Transformer encoder processing
        for layer in self.encoder_layers:
            src = layer(src)  # [seq_len, batch_size, hidden_size]

        # Get last timestep output and apply FC layer
        output_last = src[-1]  # [batch_size, hidden_size]
        out = self.fc(output_last)  # [batch_size, 1]
        out = out.unsqueeze(1)  # [batch_size, 1, 1]

        return out