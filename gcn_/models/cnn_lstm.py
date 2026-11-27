# gcn_/models/cnn_lstm.py
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2


class CNN_LSTM(nn.Module):
    """CNN-LSTM模型：使用CNN提取空间特征，然后通过LSTM处理时序信息"""

    def __init__(self, input_dim, cnn_channels=[32, 64], kernel_sizes=[3, 3],
                 lstm_hidden_dim=64, lstm_num_layers=2, output_dim=TARGET_DIM, dropout=0.2):
        """
        初始化CNN-LSTM模型

        Args:
            input_dim: 输入特征维度
            cnn_channels: CNN通道数列表
            kernel_sizes: CNN卷积核大小列表
            lstm_hidden_dim: LSTM隐藏层维度
            lstm_num_layers: LSTM层数
            output_dim: 输出维度
            dropout: Dropout比率
        """
        super(CNN_LSTM, self).__init__()

        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.kernel_sizes = kernel_sizes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.output_dim = output_dim

        # 构建CNN层
        cnn_layers = []
        prev_channels = input_dim

        for i, (channels, kernel_size) in enumerate(zip(cnn_channels, kernel_sizes)):
            cnn_layers.append(nn.Conv1d(prev_channels, channels, kernel_size, padding=kernel_size // 2))
            cnn_layers.append(nn.ReLU())
            if i < len(cnn_channels) - 1:  # 不在最后一层添加dropout
                cnn_layers.append(nn.Dropout(dropout))
            prev_channels = channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )

        # 输出层
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            输出张量，形状为 (batch_size, 1, output_dim)
        """
        # 调整维度以适应CNN: (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)

        # CNN特征提取
        cnn_out = self.cnn(x)  # (batch_size, cnn_channels[-1], seq_len)

        # 调整维度以适应LSTM: (batch_size, seq_len, cnn_channels[-1])
        cnn_out = cnn_out.permute(0, 2, 1)

        # LSTM时序建模
        lstm_out, _ = self.lstm(cnn_out)  # (batch_size, seq_len, lstm_hidden_dim)

        # 取最后一个时间步的输出
        final_out = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)

        # 全连接层输出
        output = self.fc(final_out)  # (batch_size, output_dim)

        # 调整输出形状为 (batch_size, 1, output_dim)
        output = output.unsqueeze(1)

        return output
