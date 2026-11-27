import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.dropout = dropout
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=1)

    def forward(self, batch):
        src = batch.to(self.device)  # [batch_size, seq_len, feature_dim]
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim]

        # Initialize hidden and cell states
        h_0 = torch.zeros(self.layers * self.num_directions, src.shape[1], self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.layers * self.num_directions, src.shape[1], self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(src, (h_0, c_0))

        # out = [seq_len, batch_size, hidden_size * num_directions]
        # Use the last time step's output
        out_last = out[-1].unsqueeze(0).permute(1, 0, 2)  # [batch_size, out_len = 1, hidden_size * num_directions]

        # Apply fully connected layer
        out = self.fc(out_last)  # out = [batch_size, 1, 1]

        return out