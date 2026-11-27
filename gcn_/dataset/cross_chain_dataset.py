import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossChainDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, scaler_X, scaler_y, cross_chain_map, relay_map):
        """
        跨链数据集类：处理时序特征与图结构特征
        参数：
            data: 预处理后的DataFrame
            seq_len: 输入时序长度
            pred_len: 预测长度
            scaler_X: 特征标准化器（修改为 MinMaxScaler）
            scaler_y: 目标标准化器
            cross_chain_map: 跨链对→索引映射（用于图结构）
            relay_map: 中继节点→索引映射（用于图结构）
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        # 1. 特征工程（单链特征+跨链特征）
        feature_cols = [
            # 单链特征
            "tx_arrival_rate", "block_size_current", "orderer_node_count", "latency_local",
            # 跨链特征
            "latency_relay", "tx_cross_count",
            "cross_chain_idx",  # 跨链对索引（图节点1）
            "relay_idx"  # 中继节点索引（图节点2）
        ]
        target_cols = ["latency_cross", "throughput_cross"]  # 预测目标

        # 2. 编码离散特征（跨链对、中继节点→索引，用于构建图）
        data["cross_chain_idx"] = data["cross_chain_pair"].map(cross_chain_map)
        data["relay_idx"] = data["relay_id"].map(relay_map)

        # 3. 标准化（使用 MinMaxScaler 替代 StandardScaler）
        self.X = scaler_X.transform(data[feature_cols].values)
        self.y = scaler_y.transform(data[target_cols].values)

        # 4. 构建时序样本
        self.sample_indices = []
        for i in range(len(self.X) - seq_len - pred_len + 1):
            self.sample_indices.append((i, i + seq_len, i + seq_len + pred_len - 1))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_x, end_x, end_y = self.sample_indices[idx]
        # 时序特征：(seq_len, feature_dim)
        seq_x = torch.FloatTensor(self.X[start_x:end_x]).to(DEVICE)
        # 目标值：(pred_len, target_dim)
        seq_y = torch.FloatTensor(self.y[end_y:end_y + 1]).to(DEVICE)  # 预测1步，故取end_y
        # 图节点索引：每个时序步的跨链对、中继节点索引
        cross_chain_idx = torch.LongTensor(self.X[start_x:end_x, -2]).to(DEVICE)  # 倒数第2列为跨链对索引
        relay_idx = torch.LongTensor(self.X[start_x:end_x, -1]).to(DEVICE)  # 倒数第1列为中继节点索引
        return seq_x, seq_y, cross_chain_idx, relay_idx
