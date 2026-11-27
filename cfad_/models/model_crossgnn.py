# models/model_crossgnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from cfad_.layers.Embed import DataEmbedding_inverted


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input: [B, N, D]
        # adj: [N, N]
        support = torch.matmul(input, self.weight)  # [B, N, D] * [D, D'] = [B, N, D']
        output = torch.matmul(adj, support)  # [N, N] * [B, N, D'] = [B, N, D']
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class CrossGNNLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(CrossGNNLayer, self).__init__()
        self.d_model = d_model
        self.gc1 = GraphConvolution(d_model, d_model)
        self.gc2 = GraphConvolution(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, adj):
        # x: [B, N, D] where N is number of nodes
        # adj: [N, N] adjacency matrix
        residual = x
        x = self.gc1(x, adj)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = self.norm1(x + residual)

        # FFN
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)

        return x


class CrossGNN(nn.Module):
    def __init__(self, configs):
        super(CrossGNN, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        # Graph construction parameters
        self.num_nodes = getattr(configs, 'num_nodes', 1)  # Default to 1 if not specified
        self.e_layers = getattr(configs, 'e_layers', 2)

        # Create adjacency matrix (learnable)
        self.adj = nn.Parameter(torch.randn(self.num_nodes, self.num_nodes))

        # CrossGNN layers
        self.crossgnn_layers = nn.ModuleList([
            CrossGNNLayer(configs.d_model, configs.dropout)
            for _ in range(self.e_layers)
        ])

        # Decoder projection
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Hidden dimension for compatibility
        self.hidden_dim = configs.d_model

    def construct_adj(self, x):
        # x: [B, N, D]
        # Construct adjacency matrix based on feature similarity or use learnable matrix
        batch_size, num_nodes, d_model = x.shape

        # If num_nodes doesn't match our expected num_nodes, we need to adjust
        if num_nodes != self.num_nodes:
            # Use a simple identity matrix for now
            adj = torch.eye(num_nodes).to(x.device)
        else:
            # Use learnable adjacency matrix
            adj = F.softmax(self.adj, dim=1)

        return adj

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means, stdev = None, None

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, N, D]

        # Construct adjacency matrix
        adj = self.construct_adj(enc_out)

        # CrossGNN layers
        for layer in self.crossgnn_layers:
            enc_out = layer(enc_out, adj)

        # Projection
        dec_out = self.projection(enc_out)  # [B, N, pred_len]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1))

        return dec_out, enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Return only the last pred_len time steps for prediction
        return dec_out[:, -self.pred_len:, :], enc_out
