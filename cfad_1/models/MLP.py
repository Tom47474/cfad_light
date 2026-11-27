import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, device, input_size, hidden_sizes, output_size, num_layers):
        super(MLP, self).__init__()
        self.device = device
        self.num_layers = num_layers
        # 输入层到第一个隐藏层
        self.layers = nn.ModuleList()
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_size, hidden_sizes))
        self.layers.append(nn.ReLU())
        # 中间的隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_sizes, hidden_sizes))
            self.layers.append(nn.ReLU())
        # 最后一个隐藏层到输出层
        self.layers.append(nn.Linear(hidden_sizes, output_size))

    def forward(self, x):
        x = x.to(self.device)
        # 将时间序列数据展平成一个向量（假设x的形状是[batch_size, seq_len, feature_dim]）
        # x = x.view(x.size(0), -1)  # 形状变为[batch_size, seq_len * feature_dim]
        # print("x.shape--->before--->", x.shape)
        for layer in self.layers:
            x = layer(x)
        # x = x.unsqueeze(1)
        # print("x.shape--->after--->", x.shape)
        return x


# # 示例使用
# input_dim = 24 * 3  # 因为我们忽略了 seq_len，所以将 feature_dim 乘以 seq_len
# hidden_dim = 64
# output_dim = 1
# num_layers = 2  # 包括输入层和输出层在内的总层数（这里的 2 表示有一个隐藏层）
# device = 'cpu'
# model = MLP(device, input_dim, hidden_dim, output_dim, num_layers)
# x = torch.randn(5, 3, 24)  # 与 GRU 示例相同的输入形状
# output = model(x)
#
# print(x.shape)  # [5, 3, 24]
# print(output.shape)  # [5, 1, 1]