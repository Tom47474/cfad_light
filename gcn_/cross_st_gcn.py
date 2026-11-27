import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import networkx as nx
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from tqdm import tqdm

# -------------------------- 配置参数 --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 96
PRED_LEN = 10
BATCH_SIZE = 64
EPOCHS = 1  # 可先调小调试
LR = 1e-3
FEATURE_DIM = 6  # 输入特征维度（去掉索引列）
TARGET_DIM = 2

# -------------------------- 数据集 --------------------------
class CrossChainDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, scaler_X, scaler_y, cross_chain_map, relay_map):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        # 特征列
        feature_cols = [
            "tx_arrival_rate", "block_size_current", "orderer_node_count", "latency_local",
            "latency_relay", "tx_cross_count"
        ]
        target_cols = ["latency_cross", "throughput_cross"]

        # 索引列
        data["cross_chain_idx"] = data["cross_chain_pair"].map(cross_chain_map)
        data["relay_idx"] = data["relay_id"].map(relay_map)

        self.cross_chain_idx = data["cross_chain_idx"].values
        self.relay_idx = data["relay_idx"].values

        # 标准化
        self.X = scaler_X.transform(data[feature_cols].values)
        self.y = scaler_y.transform(data[target_cols].values)

        # 样本索引
        self.sample_indices = []
        for i in range(len(self.X) - seq_len - pred_len + 1):
            self.sample_indices.append((i, i + seq_len, i + seq_len + pred_len))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_x, end_x, end_y = self.sample_indices[idx]
        seq_x = torch.FloatTensor(self.X[start_x:end_x]).to(DEVICE)
        seq_y = torch.FloatTensor(self.y[end_x:end_y]).to(DEVICE)
        cross_idx = torch.LongTensor(self.cross_chain_idx[start_x:end_x]).to(DEVICE)
        relay_idx = torch.LongTensor(self.relay_idx[start_x:end_x]).to(DEVICE)
        return seq_x, seq_y, cross_idx, relay_idx

def load_and_preprocess_data(data_path="HFBDP-Cross.csv", train_ratio=0.7):
    data = pd.read_csv(data_path).sort_values("timestamp").reset_index(drop=True)
    cross_chain_map = {v: i for i, v in enumerate(data["cross_chain_pair"].unique())}
    relay_map = {v: i for i, v in enumerate(data["relay_id"].unique())}
    num_cross_chain = len(cross_chain_map)
    num_relay = len(relay_map)
    num_total_nodes = num_cross_chain + num_relay

    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()

    # 标准化
    feature_cols = ["tx_arrival_rate", "block_size_current", "orderer_node_count",
                    "latency_local", "latency_relay", "tx_cross_count"]
    target_cols = ["latency_cross", "throughput_cross"]

    scaler_X = StandardScaler().fit(train_data[feature_cols].values)
    scaler_y = StandardScaler().fit(train_data[target_cols].values)

    train_dataset = CrossChainDataset(train_data, SEQ_LEN, PRED_LEN, scaler_X, scaler_y, cross_chain_map, relay_map)
    test_dataset = CrossChainDataset(test_data, SEQ_LEN, PRED_LEN, scaler_X, scaler_y, cross_chain_map, relay_map)

    # 构建图
    G = nx.Graph()
    G.add_nodes_from(range(num_total_nodes))
    for cross_idx in range(num_cross_chain):
        for relay_idx in range(num_relay):
            G.add_edge(cross_idx, num_cross_chain + relay_idx)
    edge_index = from_networkx(G).edge_index.to(DEVICE)

    # 保存到 dataset
    train_dataset.edge_index = edge_index
    test_dataset.edge_index = edge_index

    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False),
        scaler_y, num_total_nodes
    )

# -------------------------- 模型 --------------------------
class CFAD_Paper1(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=TARGET_DIM, num_cfadm_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_emb = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, padding="same")
        self.cfadm_layers = nn.ModuleList()
        for _ in range(num_cfadm_layers):
            self.cfadm_layers.append(nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 3, padding="same"), nn.ReLU()))
            self.cfadm_layers.append(nn.Conv1d(2*hidden_dim, hidden_dim, 1, padding="same"))
            self.cfadm_layers.append(nn.Sigmoid())
            self.cfadm_layers.append(nn.Conv1d(2*hidden_dim, hidden_dim, 1, padding="same"))
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, 1, padding="same")

    def forward(self, x):
        x = x.permute(0,2,1)
        s = self.seq_emb(x)
        for i in range(0, len(self.cfadm_layers), 4):
            s_agg = self.cfadm_layers[i](s)
            s_concat = torch.cat([s_agg, s], dim=1)
            g = self.cfadm_layers[i+1](s_concat)
            g = self.cfadm_layers[i+2](g)
            d = self.cfadm_layers[i+3](s_concat)
            s = g*d
        out = self.output_proj(s)
        out = out[:, :, -PRED_LEN:]  # 多步输出
        return out.permute(0,2,1)

class LSTM_Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, output_dim=TARGET_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out,_ = self.lstm(x)
        out = self.fc(lstm_out[:, -PRED_LEN:, :])
        return out

class CMT_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_nodes=10, output_dim=TARGET_DIM, gcn_layers=2):
        super().__init__()
        self.node_emb_cc = nn.Embedding(num_nodes, hidden_dim)
        self.node_emb_relay = nn.Embedding(num_nodes, hidden_dim)
        self.gcn_layers = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(gcn_layers)])
        self.relu = nn.ReLU()
        self.short_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.long_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, cross_chain_idx, relay_idx):
        batch_size, seq_len, feat_dim = x.shape
        out_seq = []

        for t in range(seq_len):
            x_t = x[:,t,:]
            node_feat = self.node_emb_cc(cross_chain_idx[:,t]) + self.node_emb_relay(relay_idx[:,t])
            h = self.relu(node_feat)
            for gcn in self.gcn_layers:
                h = self.relu(gcn(h, edge_index))
            out_seq.append(h[cross_chain_idx[:,t]])

        h_seq = torch.stack(out_seq, dim=1).permute(0,2,1)
        short_out = self.relu(self.short_conv(h_seq))
        long_out = self.relu(self.long_conv(h_seq))
        # 保证 seq_len 一致
        min_len = min(short_out.size(2), long_out.size(2))
        short_out = short_out[:, :, :min_len]
        long_out = long_out[:, :, :min_len]

        # 取最后一个时间步
        fused = torch.cat([short_out[:, :, -1], long_out[:, :, -1]], dim=1)
        fused = self.relu(self.fusion(fused))

        out = self.out(fused).unsqueeze(1).repeat(1,PRED_LEN,1)
        return out

# -------------------------- 训练与评估 --------------------------
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for seq_x, seq_y, cc_idx, r_idx in tqdm(train_loader):
        optimizer.zero_grad()
        if isinstance(model, CMT_Net):
            edge_index = train_loader.dataset.edge_index
            pred = model(seq_x, edge_index, cc_idx, r_idx)
        else:
            pred = model(seq_x)
        loss = criterion(pred, seq_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seq_x.shape[0]
    return total_loss / len(train_loader.dataset)

def evaluate_model(model, test_loader, scaler_y, criterion):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for seq_x, seq_y, cc_idx, r_idx in tqdm(test_loader):
            if isinstance(model, CMT_Net):
                edge_index = test_loader.dataset.edge_index
                pred = model(seq_x, edge_index, cc_idx, r_idx)
            else:
                pred = model(seq_x)
            loss = criterion(pred, seq_y)
            total_loss += loss.item() * seq_x.shape[0]

            y_true.extend(scaler_y.inverse_transform(seq_y.cpu().numpy().reshape(-1,2)))
            y_pred.extend(scaler_y.inverse_transform(pred.cpu().numpy().reshape(-1,2)))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae_latency = mean_absolute_error(y_true[:,0], y_pred[:,0])
    rmse_latency = np.sqrt(mean_squared_error(y_true[:,0], y_pred[:,0]))
    mape_latency = np.mean(np.abs((y_true[:,0]-y_pred[:,0])/(y_true[:,0]+1e-6)))*100
    mae_throughput = mean_absolute_error(y_true[:,1], y_pred[:,1])
    rmse_throughput = np.sqrt(mean_squared_error(y_true[:,1], y_pred[:,1]))
    mape_throughput = np.mean(np.abs((y_true[:,1]-y_pred[:,1])/(y_true[:,1]+1e-6)))*100
    return {
        "avg_loss": total_loss / len(test_loader.dataset),
        "mae_latency": mae_latency,
        "rmse_latency": rmse_latency,
        "mape_latency": mape_latency,
        "mae_throughput": mae_throughput,
        "rmse_throughput": rmse_throughput,
        "mape_throughput": mape_throughput
    }

# -------------------------- 主训练流程 --------------------------
train_loader, test_loader, scaler_y, num_nodes = load_and_preprocess_data()

models = {}
cfad_model = CFAD_Paper1(FEATURE_DIM).to(DEVICE)
lstm_model = LSTM_Baseline(FEATURE_DIM).to(DEVICE)
cmt_model = CMT_Net(FEATURE_DIM, num_nodes=num_nodes).to(DEVICE)

models["CFAD-Paper1"] = {"model": cfad_model, "optimizer": optim.Adam(cfad_model.parameters(), lr=LR)}
models["LSTM"] = {"model": lstm_model, "optimizer": optim.Adam(lstm_model.parameters(), lr=LR)}
models["CMT-Net"] = {"model": cmt_model, "optimizer": optim.Adam(cmt_model.parameters(), lr=LR)}

criterion = nn.MSELoss()

for name, m in models.items():
    print(f"=== Training {name} ===")
    for epoch in range(EPOCHS):
        loss = train_model(m["model"], train_loader, criterion, m["optimizer"])
        if (epoch+1) % 10 == 0 or epoch==0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {loss:.4f}")
    metrics = evaluate_model(m["model"], test_loader, scaler_y, criterion)
    print(f"Test Metrics for {name}: {metrics}")
