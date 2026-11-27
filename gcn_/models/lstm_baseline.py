import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2

class LSTM_Baseline(nn.Module):
    """仅时序建模，复用 paper1.pdf 时序预测逻辑，无空间建模"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=3, output_dim=TARGET_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_dim)  # 预测跨链性能

    def forward(self, x):
        """x: (batch_size, seq_len, input_dim) → (batch_size, pred_len, output_dim)"""
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        # 取最后一个时序步的输出预测
        out = self.fc(lstm_out[:, -1, :]).unsqueeze(1)  # (batch_size, 1, output_dim)
        return out
