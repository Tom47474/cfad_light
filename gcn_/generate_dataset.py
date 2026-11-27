# gcn_/generate_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_hfbdp_cross_data(total_days: int = 60, output_path: str = "HFBDP-Cross.csv") -> None:
    """
    生成模拟的区块链跨链数据集（HFBDP-Cross），更贴近真实场景的数据分布

    参数:
        total_days: 数据集总时长（天），默认60天
        output_path: 输出CSV文件路径
    """
    # -------------------------- 1. 基础参数配置 --------------------------
    time_interval = 5  # 分钟
    total_samples = total_days * 24 * 60 // time_interval  # 总样本数

    # 单链字段范围（更贴近真实区块链网络）
    tx_arrival_rate_range = (5, 500)  # TPS (更宽范围，包含低负载和高负载情况)
    block_size_range = (5, 2000)  # KB (更大范围，包含小交易和大区块)
    orderer_node_options = [1, 3, 5, 7, 9, 15]  # 排序节点数（增加1个节点的选项，模拟单节点网络）
    latency_local_range = (0.05, 5.0)  # 秒（更宽范围，包含高性能和低性能网络）
    throughput_local_range = (10, 1000)  # TPS（更宽范围）

    # 跨链字段范围（扩展更多真实场景）
    cross_chain_pairs = [
        "Ethereum-BinanceSmartChain", "Ethereum-Polygon", "Ethereum-Solana",
        "BinanceSmartChain-Polygon", "BinanceSmartChain-Solana",
        "Polygon-Solana", "Ethereum-Avalanche", "BinanceSmartChain-Avalanche",
        "Polygon-Avalanche", "Solana-Avalanche", "Ethereum-Cosmos",
        "BinanceSmartChain-Cosmos", "Polygon-Cosmos", "Solana-Cosmos",
        "Avalanche-Cosmos"
    ]

    relay_ids = [
        "Relay_1", "Relay_2", "Relay_3", "Relay_4", "Relay_5",
        "Relay_6", "Relay_7", "Relay_8", "Relay_9", "Relay_10",
        "Chainlink", "Wormhole", "Multichain", "cBridge", "Across"
    ]

    latency_relay_range = (0.1, 3.0)  # 秒（更宽范围）
    throughput_relay_range = (5, 500)  # TPS（更宽范围）

    # -------------------------- 2. 时间序列 --------------------------
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i * time_interval) for i in range(total_samples)]
    timestamps = [dt.strftime("%Y/%m/%d %H:%M") for dt in timestamps]

    # -------------------------- 3. 单链字段（更真实的数据分布）--------------------------
    # 交易到达率：结合多种周期性模式（日、周）和随机波动
    daily_cycle = np.sin(np.linspace(0, 2 * np.pi * total_days, total_samples))  # 日周期
    weekly_cycle = np.sin(np.linspace(0, 2 * np.pi * total_days / 7, total_samples))  # 周周期

    # 基础交易率（考虑多种周期性）
    base_rate = (tx_arrival_rate_range[1] - tx_arrival_rate_range[0]) * 0.5 * \
                (1 + 0.7 * daily_cycle + 0.3 * weekly_cycle) + \
                tx_arrival_rate_range[0]

    # 添加随机波动和突发事件（模拟真实网络中的流量尖峰）
    random_noise = np.random.normal(0, 15, total_samples)

    # 模拟突发事件（如市场波动导致的交易激增）
    spike_events = np.random.choice([0, 1], total_samples, p=[0.99, 0.01])  # 1%概率的突发事件
    spikes = spike_events * np.random.uniform(100, 300, total_samples)  # 突发增加100-300 TPS

    tx_arrival_rate = base_rate + random_noise + spikes
    tx_arrival_rate = np.clip(tx_arrival_rate, *tx_arrival_rate_range).round().astype(int)

    # 块大小：与交易率相关但有噪声
    block_size_current = block_size_range[0] + (block_size_range[1] - block_size_range[0]) * \
                         np.power((tx_arrival_rate - tx_arrival_rate_range[0]) /
                                  (tx_arrival_rate_range[1] - tx_arrival_rate_range[0]), 1.2) + \
                         np.random.normal(0, 50, total_samples)
    block_size_current = np.clip(block_size_current, *block_size_range).round().astype(int)

    # 排序节点数：与网络负载相关（高负载网络通常有更多节点）
    orderer_probabilities = []
    for rate in tx_arrival_rate:
        # 高负载网络更可能有更多节点
        if rate < 50:
            orderer_probabilities.append([0.4, 0.3, 0.2, 0.07, 0.02, 0.01])  # 低负载网络
        elif rate < 150:
            orderer_probabilities.append([0.1, 0.2, 0.3, 0.25, 0.1, 0.05])  # 中等负载网络
        else:
            orderer_probabilities.append([0.05, 0.1, 0.2, 0.3, 0.25, 0.1])  # 高负载网络

    orderer_node_count = []
    for prob in orderer_probabilities:
        orderer_node_count.append(np.random.choice(orderer_node_options, p=prob))

    # 本地延迟：与交易率和块大小相关
    # 高交易率和大块通常导致更高延迟
    latency_factor = 0.3 * (tx_arrival_rate - tx_arrival_rate_range[0]) / \
                     (tx_arrival_rate_range[1] - tx_arrival_rate_range[0]) + \
                     0.7 * (block_size_current - block_size_range[0]) / \
                     (block_size_range[1] - block_size_range[0])

    latency_local = latency_local_range[0] + (latency_local_range[1] - latency_local_range[0]) * \
                    latency_factor + np.random.normal(0, 0.1, total_samples)
    latency_local = np.clip(latency_local, *latency_local_range)

    # 本地吞吐量：与排序节点数正相关，与延迟负相关
    throughput_base = 50 + 950 * (np.array(orderer_node_count) - 1) / (15 - 1)
    throughput_factor = 1 - 0.5 * (latency_local - latency_local_range[0]) / \
                        (latency_local_range[1] - latency_local_range[0])
    throughput_local = throughput_base * throughput_factor * (0.8 + 0.4 * np.random.random(total_samples))
    throughput_local = np.clip(throughput_local, *throughput_local_range)

    # -------------------------- 4. 跨链字段（更真实的跨链行为）--------------------------
    cross_chain_pair = [random.choice(cross_chain_pairs) for _ in range(total_samples)]
    relay_id = [random.choice(relay_ids) for _ in range(total_samples)]

    # 跨链交易数：与源链和目标链的交易率相关，但有延迟效应
    # 使用不同时间点的交易率来模拟跨链交易的延迟
    lag_indices = np.maximum(np.arange(total_samples) - np.random.randint(5, 20, total_samples), 0)
    tx_cross_lag = tx_arrival_rate[lag_indices]

    tx_cross_count = 0.05 * (tx_arrival_rate + tx_cross_lag) + \
                     np.random.normal(0, 10, total_samples)
    tx_cross_count = np.clip(tx_cross_count, 0, 100).round().astype(int)

    # 中继延迟：基础值 + 跨链交易数影响 + 随机噪声 + 网络状态影响
    # 高交易量时中继延迟增加
    relay_congestion = 0.5 * (tx_cross_count / 100)  # 拥塞因子
    latency_relay = latency_relay_range[0] + \
                    (latency_relay_range[1] - latency_relay_range[0]) * \
                    (0.5 + 0.5 * relay_congestion) + \
                    np.random.normal(0, 0.1, total_samples)
    latency_relay = np.clip(latency_relay, *latency_relay_range)

    # 跨链端到端延迟：源链延迟 + 目标链延迟 + 中继延迟 + 网络波动
    latency_target_local = latency_local * (0.7 + 0.6 * np.random.random(total_samples))

    # 网络波动因子（模拟网络不稳定）
    network_fluctuation = 1 + 0.2 * np.sin(np.linspace(0, 2 * np.pi * total_days * 2, total_samples)) + \
                          0.1 * np.random.random(total_samples)

    latency_cross = (latency_local + latency_target_local + latency_relay) * network_fluctuation

    # 中继吞吐量：与中继类型相关，并受跨链交易数影响
    # 不同中继有不同的基础吞吐量
    relay_base_throughput = {}
    for relay in relay_ids:
        if "Relay" in relay:
            relay_base_throughput[relay] = np.random.uniform(50, 200)
        else:  # 知名中继通常有更高吞吐量
            relay_base_throughput[relay] = np.random.uniform(200, 500)

    throughput_relay = []
    for i in range(total_samples):
        base = relay_base_throughput[relay_id[i]]
        # 高跨链交易数会降低中继吞吐量
        congestion_factor = 1 - 0.7 * (tx_cross_count[i] / 100)
        relay_tp = base * congestion_factor + np.random.normal(0, 10)
        throughput_relay.append(relay_tp)

    throughput_relay = np.clip(throughput_relay, *throughput_relay_range)

    # 目标链本地吞吐量
    throughput_target_local = throughput_local * (0.7 + 0.6 * np.random.random(total_samples))

    # 跨链吞吐量：取瓶颈值（源链、目标链、中继三者中的最小值）
    throughput_cross = np.minimum(np.minimum(throughput_local, throughput_target_local), throughput_relay)

    # -------------------------- 5. 保存数据集 --------------------------
    data = pd.DataFrame({
        "timestamp": timestamps,
        "tx_arrival_rate": tx_arrival_rate,
        "block_size_current": block_size_current,
        "orderer_node_count": orderer_node_count,
        "latency_local": np.round(latency_local, 4),
        "throughput_local": np.round(throughput_local, 2),
        "cross_chain_pair": cross_chain_pair,
        "relay_id": relay_id,
        "latency_relay": np.round(latency_relay, 4),
        "tx_cross_count": tx_cross_count,
        "latency_cross": np.round(latency_cross, 4),
        "throughput_cross": np.round(throughput_cross, 2)
    })

    data.to_csv(output_path, index=False)
    print(f"HFBDP-Cross数据集生成完成！规模：{total_samples} 条样本，输出路径：{output_path}")

    # 打印数据集统计信息
    print("\n数据集统计信息：")
    print(f"时间范围: {timestamps[0]} 到 {timestamps[-1]}")
    print(f"跨链对数量: {len(set(cross_chain_pair))}")
    print(f"中继节点数量: {len(set(relay_id))}")
    print("\n关键字段统计:")
    print(data[["tx_arrival_rate", "block_size_current", "latency_local",
                "throughput_local", "latency_cross", "throughput_cross"]].describe())


if __name__ == "__main__":
    generate_hfbdp_cross_data(total_days=360, output_path="HFBDP-Cross.csv")
