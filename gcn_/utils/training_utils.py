# gcn_/utils/training_utils.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, criterion, optimizer, epoch, epochs):
    """训练模型，复用 paper1.pdf MSE损失函数（第1-124节）"""
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)"):
        seq_x, seq_y, cc_idx, r_idx = batch
        optimizer.zero_grad()

        # 前向传播（不同模型输入不同）
        # 检查模型类型并相应处理
        model_name = type(model).__name__
        if model_name in ['CMT_Net']:
            # 空间/时空模型需要邻接矩阵
            adj = train_loader.dataset.adj if hasattr(train_loader.dataset, "adj") else None
            pred = model(seq_x, adj, cc_idx, r_idx)
        elif model_name in ['STSGCN', 'GAT', 'AutoST']:
            # STSGCN、GAT和AutoST模型需要邻接矩阵，但不需要节点索引
            adj = train_loader.dataset.adj if hasattr(train_loader.dataset, "adj") else None
            pred = model(seq_x, adj)
        else:
            # 时序/CFAD模型仅需时序特征
            pred = model(seq_x)

        # 计算损失（paper1.pdf 第1-124节：MSE损失）
        loss = criterion(pred, seq_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seq_x.shape[0]
    return total_loss / len(train_loader.dataset)


def evaluate_model(model, test_loader, scaler_y, criterion):
    """评估模型，计算 MAE/RMSE/MAPE（与 paper1.pdf 第1-124、1-134节一致）"""
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating (Test)"):
            seq_x, seq_y, cc_idx, r_idx = batch

            # 前向传播
            model_name = type(model).__name__
            if model_name in ['CMT_Net']:
                adj = test_loader.dataset.adj if hasattr(test_loader.dataset, "adj") else None
                pred = model(seq_x, adj, cc_idx, r_idx)
            elif model_name in ['STSGCN', 'GAT', 'AutoST']:
                # STSGCN、GAT和AutoST模型需要邻接矩阵，但不需要节点索引
                adj = test_loader.dataset.adj if hasattr(test_loader.dataset, "adj") else None
                pred = model(seq_x, adj)
            else:
                pred = model(seq_x)

            # 记录损失与预测结果
            loss = criterion(pred, seq_y)
            total_loss += loss.item() * seq_x.shape[0]

            # 反标准化（还原为真实值，用于计算MAE/MAPE）
            # 正确处理维度转换
            y_true_batch = seq_y.cpu().numpy()  # (batch_size, 1, 2) 或 (batch_size, 2, 1)
            pred_batch = pred.cpu().numpy()     # (batch_size, 1, 2) 或 (batch_size, 2, 1)

            # 确保正确的形状 (batch_size, 2)
            if y_true_batch.ndim == 3:
                if y_true_batch.shape[1] == 1:
                    y_true_batch = y_true_batch.squeeze(1)  # (batch_size, 2)
                elif y_true_batch.shape[2] == 1:
                    y_true_batch = y_true_batch.squeeze(2)  # (batch_size, 2)
                else:
                    y_true_batch = y_true_batch.reshape(y_true_batch.shape[0], -1)

            if pred_batch.ndim == 3:
                if pred_batch.shape[1] == 1:
                    pred_batch = pred_batch.squeeze(1)  # (batch_size, 2)
                elif pred_batch.shape[2] == 1:
                    pred_batch = pred_batch.squeeze(2)  # (batch_size, 2)
                else:
                    pred_batch = pred_batch.reshape(pred_batch.shape[0], -1)

            y_true.extend(scaler_y.inverse_transform(y_true_batch))
            y_pred.extend(scaler_y.inverse_transform(pred_batch))

    # 转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算评估指标（与 paper1.pdf 第1-134节一致）
    mae_latency = mean_absolute_error(y_true[:, 0], y_pred[:, 0])  # 跨链延迟MAE
    rmse_latency = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))  # 跨链延迟RMSE
    mape_latency = np.mean(np.abs((y_true[:, 0] - y_pred[:, 0]) / (y_true[:, 0] + 1e-6))) * 100  # 跨链延迟MAPE（%）

    mae_throughput = mean_absolute_error(y_true[:, 1], y_pred[:, 1])  # 跨链吞吐量MAE
    rmse_throughput = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))  # 跨链吞吐量RMSE
    mape_throughput = np.mean(np.abs((y_true[:, 1] - y_pred[:, 1]) / (y_true[:, 1] + 1e-6))) * 100  # 跨链吞吐量MAPE（%）

    avg_loss = total_loss / len(test_loader.dataset)
    return {
        "avg_loss": avg_loss,
        "latency": {"MAE": mae_latency, "RMSE": rmse_latency, "MAPE": mape_latency},
        "throughput": {"MAE": mae_throughput, "RMSE": rmse_throughput, "MAPE": mape_throughput}
    }
