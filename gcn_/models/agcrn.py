# gcn_/models/agcrn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2


class AVWGCN(nn.Module):
    """ Adaptive Virtual Weight Graph Convolutional Network """

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x, node_embeddings):
        # x shaped [B, N, C], node_embeddings shaped [N, D]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])

        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    """ AGCRN Cell """

    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_size = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_size, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_size, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_size
        # state: B, num_nodes, hidden_size
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_size, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_size)


class AVWEmbedding(nn.Module):
    """ Adaptive Virtual Weight Embedding """

    def __init__(self, embed_size, node_num):
        super(AVWEmbedding, self).__init__()
        self.node_embeddings = nn.Parameter(torch.randn(node_num, embed_size), requires_grad=True)
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self):
        return self.node_embeddings


class AGCRN(nn.Module):
    """ Simplified Attention-based Graph Convolutional Recurrent Network for Cross-chain Prediction """

    def __init__(self, input_dim, rnn_units, num_nodes, output_dim, seq_len, cheb_k=3, embed_dim=8, num_layers=1):
        super(AGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # For cross-chain data, we treat each time step as a "node"
        # We'll reshape the input to fit the model's expectation
        self.input_projection = nn.Linear(input_dim, rnn_units)

        # Embedding layer - for a simplified version, we use a fixed embedding
        self.embedding = AVWEmbedding(embed_dim, num_nodes)

        # RNN layers
        self.rnn_cells = nn.ModuleList()
        self.rnn_cells.append(AGCRNCell(num_nodes, rnn_units, rnn_units, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.rnn_cells.append(AGCRNCell(num_nodes, rnn_units, rnn_units, cheb_k, embed_dim))

        # Output layer
        self.fc = nn.Linear(rnn_units, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        batch_size = x.size(0)

        # Project input features
        x = self.input_projection(x)  # (batch_size, seq_len, rnn_units)

        # Reshape to (batch_size, num_nodes, feature_dim)
        # For simplicity, we treat seq_len as num_nodes
        if self.num_nodes > x.size(1):
            # Pad with zeros if we have fewer "nodes" than expected
            padding = torch.zeros(batch_size, self.num_nodes - x.size(1), self.rnn_units).to(x.device)
            x = torch.cat([x, padding], dim=1)
        elif self.num_nodes < x.size(1):
            # Truncate if we have more "nodes" than expected
            x = x[:, :self.num_nodes, :]

        # Transpose to (batch_size, num_nodes, rnn_units)
        x = x[:, :self.num_nodes, :]  # Ensure correct size

        # Get node embeddings
        node_embeddings = self.embedding()

        # Initialize hidden states
        hidden_states = []
        for i in range(self.num_layers):
            hidden_states.append(self.rnn_cells[i].init_hidden_state(batch_size))

        # RNN forward - process each time step
        outputs = []
        for t in range(self.num_nodes):
            if t < x.size(1):
                x_t = x[:, t, :].unsqueeze(1)  # (batch_size, 1, rnn_units)
                # Expand to match node dimension
                x_t = x_t.expand(-1, self.num_nodes, -1)  # (batch_size, num_nodes, rnn_units)
            else:
                x_t = torch.zeros(batch_size, self.num_nodes, self.rnn_units).to(x.device)  # padding

            # Process through RNN layers
            h = x_t
            for i in range(self.num_layers):
                h = self.rnn_cells[i](h, hidden_states[i], node_embeddings)
                hidden_states[i] = h
            outputs.append(h[:, 0, :])  # Take the first node's output

        # Take the last output
        last_output = outputs[-1]  # (batch_size, rnn_units)

        # Final prediction
        out = self.fc(last_output)  # (batch_size, output_dim)

        # Reshape to (batch_size, 1, output_dim) to match expected output format
        out = out.unsqueeze(1)

        return out
