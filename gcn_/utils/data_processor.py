import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from gcn_.dataset.cross_chain_dataset import CrossChainDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(data_path="HFBDP-Cross.csv", train_ratio=0.8, seq_len=96, pred_len=10):
    """加载数据并预处理，使用 MinMaxScaler 进行标准化"""
    # 1. 加载数据
    data = pd.read_csv(data_path)
    data = data.sort_values("timestamp").reset_index(drop=True)  # 按时间排序

    # 2. 编码离散字段（跨链对、中继节点→唯一索引）
    cross_chain_pairs = data["cross_chain_pair"].unique()
    relays = data["relay_id"].unique()
    cross_chain_map = {pair: i for i, pair in enumerate(cross_chain_pairs)}
    relay_map = {relay: i for i, relay in enumerate(relays)}
    num_cross_chain = len(cross_chain_pairs)
    num_relay = len(relays)
    num_total_nodes = num_cross_chain + num_relay  # 图总节点数

    # 3. 划分训练集/测试集
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    # 4. 标准化（使用 MinMaxScaler 替代 StandardScaler）
    feature_cols = [
        "tx_arrival_rate", "block_size_current", "orderer_node_count", "latency_local",
        "latency_relay", "tx_cross_count", "cross_chain_idx", "relay_idx"
    ]
    target_cols = ["latency_cross", "throughput_cross"]

    # 先编码训练集离散字段
    train_data["cross_chain_idx"] = train_data["cross_chain_pair"].map(cross_chain_map)
    train_data["relay_idx"] = train_data["relay_id"].map(relay_map)
    test_data["cross_chain_idx"] = test_data["cross_chain_pair"].map(cross_chain_map)
    test_data["relay_idx"] = test_data["relay_id"].map(relay_map)

    # 拟合 MinMaxScaler（模仿 data_processor.py 的方式）
    scaler_X = MinMaxScaler()
    scaler_X.fit(train_data[feature_cols].values)
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_data[target_cols].values)

    # 5. 构建数据集
    train_dataset = CrossChainDataset(
        train_data, seq_len, pred_len, scaler_X, scaler_y, cross_chain_map, relay_map
    )
    test_dataset = CrossChainDataset(
        test_data, seq_len, pred_len, scaler_X, scaler_y, cross_chain_map, relay_map
    )

    # 6. 构建跨链图
    G = nx.Graph()
    G.add_nodes_from(range(num_total_nodes))
    # 为每个跨链对添加与所有中继节点的边
    for cross_idx in range(num_cross_chain):
        for relay_idx in range(num_relay):
            G.add_edge(cross_idx, num_cross_chain + relay_idx)  # 中继节点索引偏移
    # 转换为PyTorch Geometric格式的邻接矩阵
    adj = torch.zeros((num_total_nodes, num_total_nodes), device=DEVICE)
    for u, v in G.edges():
        adj[u, v] = 1.0
        adj[v, u] = 1.0  # 无向图

    return (
        DataLoader(train_dataset, batch_size=64, shuffle=True),
        DataLoader(test_dataset, batch_size=64, shuffle=False),
        scaler_y, adj, num_total_nodes
    )
