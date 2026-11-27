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


class Transformer(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.dropout = dropout

        # Transformer parameters with default values
        self.nhead = 8  # number of attention heads
        self.dim_feedforward = 2048  # feedforward network dimension

        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        src = batch.to(self.device)  # [batch_size, seq_len, input_size]
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, input_size]

        # Embed input and add positional encoding
        src = self.embedding(src)  # [seq_len, batch_size, hidden_size]
        src = self.pos_encoder(src)  # add positional encoding

        # Transformer processing
        output = self.transformer_encoder(src)  # [seq_len, batch_size, hidden_size]

        # Get last timestep output and apply FC layer
        output_last = output[-1]  # [batch_size, hidden_size]
        out = self.fc(output_last)  # [batch_size, 1]
        out = out.unsqueeze(1)  # [batch_size, 1, 1]

        return out