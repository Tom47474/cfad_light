import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, bidirectional=False):
        super(GRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 伪双向标志，仅用于维度匹配，GRU本身不支持双向
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # 全连接层的输入特征数为 hidden_size * num_directions
        self.fc = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=1)

    def forward(self, x):
        x = x.to(self.device)  # [batch_size, seq_len, feature_dim]
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)

        # out = [batch_size, seq_len, hidden_size * num_directions] (if bidirectional=True, else hidden_size)
        # Use the last time step's output
        out_last = out[:, -1, :]  # [batch_size, hidden_size * num_directions]

        # Apply fully connected layer
        out = self.fc(out_last)  # out = [batch_size, 1]

        # Reshape to match LSTM output format [batch_size, 1, 1]
        out = out.unsqueeze(2)

        return out


# # 示例使用
# input_size = 24
# hidden_size = 20
# num_layers = 2
# bidirectional = False  # GRU不支持真正的双向，但为了与LSTM代码结构一致，这里保留此参数
#
# model = GRU(device='cpu', input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
#             bidirectional=bidirectional)
# x = torch.randn(5, 3, input_size)
# output = model(x)
#
# print(x.shape)  # [5, 3, 24]
# print(output.shape)  # [5, 1, 1]