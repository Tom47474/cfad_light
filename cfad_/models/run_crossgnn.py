# models/run_crossgnn.py
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune

from cfad_.dataset.data_processor import load_and_split_data
from cfad_.utils.Config import configs
from cfad_.utils.metrics import MAE, MSE, RMSE, MAPE
from cfad_.utils.tools import plot_model_comparison, plot_speed_accuracy_tradeoff, count_parameters, \
    measure_inference_time, EarlyStopping

from model_crossgnn import CrossGNN
from thop import profile


# ============================
# 固定随机种子
# ============================
def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================
# 构造时序数据集
# ============================
def create_time_series_dataset(data, seq_len, pred_len):
    x_data, y_data, x_mark = [], [], []

    if 'date' in data.columns:
        dates = pd.to_datetime(data['date'])
        hour = dates.dt.hour / 23.0 - 0.5
        day_of_week = dates.dt.dayofweek / 6.0 - 0.5
        x_mark_data = np.column_stack([hour, day_of_week])
    else:
        x_mark_data = np.zeros((len(data), 2))

    feature_columns = [col for col in data.columns if col != 'date']
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found in data")

    for i in range(seq_len, len(data) - pred_len + 1):
        x_seq = data[feature_columns].iloc[i - seq_len:i].values
        y_seq = data[feature_columns].iloc[i:i + pred_len].values
        x_mark_seq = x_mark_data[i - seq_len:i]
        x_data.append(x_seq)
        y_data.append(y_seq)
        x_mark.append(x_mark_seq)

    return np.array(x_data), np.array(x_mark), np.array(y_data)


# ============================
# 剪枝工具
# ============================
def prune_model(model, pruning_ratio=0.3):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)


def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass


def get_sparsity(model):
    total_params = 0
    zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            total_params += module.weight.numel()
            zero_params += torch.sum(module.weight == 0).item()
            if module.bias is not None:
                total_params += module.bias.numel()
                zero_params += torch.sum(module.bias == 0).item()
    return zero_params / total_params if total_params > 0 else 0


# ============================
# 绘图工具
# ============================
def plot_loss_curve(train_losses, val_losses, save_path='loss_curve.png'):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=1)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=1)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to {save_path}")


# ============================
# 训练函数
# ============================
def train_crossgnn(num_epochs, lr=1e-3, batch_size=32, pruning_ratio=0.0):
    set_seed(2021)
    device = torch.device('cuda' if torch.cuda.is_available() and configs.use_gpu else 'cpu')

    train_data, test_data = load_and_split_data()
    seq_len, pred_len = configs.seq_len, configs.pred_len

    train_x, train_x_mark, train_y = create_time_series_dataset(train_data, seq_len, pred_len)
    test_x, test_x_mark, test_y = create_time_series_dataset(test_data, seq_len, pred_len)

    train_x, train_x_mark, train_y = torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_x_mark,
                                                                                              dtype=torch.float32), torch.tensor(
        train_y, dtype=torch.float32)
    test_x, test_x_mark, test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_x_mark,
                                                                                          dtype=torch.float32), torch.tensor(
        test_y, dtype=torch.float32)

    val_size = int(0.1 * len(train_x))
    train_size = len(train_x) - val_size

    train_dataset = torch.utils.data.TensorDataset(train_x[:train_size], train_x_mark[:train_size],
                                                   train_y[:train_size])
    val_dataset = torch.utils.data.TensorDataset(train_x[train_size:], train_x_mark[train_size:], train_y[train_size:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CrossGNN(configs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True, delta=0)

    train_losses, val_losses = [], []
    best_state, best_val = None, float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]
            x_dec, x_mark_dec = x_enc[:, -pred_len:, :], x_mark_enc[:, -pred_len:, :]

            pred, _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            loss = F.mse_loss(pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]
                x_dec, x_mark_dec = x_enc[:, -pred_len:, :], x_mark_enc[:, -pred_len:, :]
                pred, _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                val_loss += F.mse_loss(pred, y_true).item()

        avg_train, avg_val = total_loss / max(1, len(train_loader)), val_loss / max(1, len(val_loader))
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = model.state_dict()

        early_stopping(avg_val, model, 'checkpoints')
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 保存最佳
    os.makedirs('checkpoints', exist_ok=True)
    if best_state is not None:
        torch.save(best_state, 'checkpoints/crossgnn_best.pth')
        model.load_state_dict(best_state)

    # plot_loss_curve(train_losses, val_losses, 'crossgnn_training_loss_curve.png')

    if pruning_ratio > 0:
        prune_model(model, pruning_ratio)
        print(f"Sparsity after pruning: {get_sparsity(model):.2%}")
        remove_pruning(model)
        torch.save(model.state_dict(), f'checkpoints/crossgnn_pruned_{pruning_ratio}.pth')

    torch.save(model.state_dict(), 'checkpoints/crossgnn_model.pth')
    print("Saved CrossGNN model to checkpoints/crossgnn_model.pth")
    return model


# ===== 计算 FLOPs =====
def compute_flops(model, sample_input):
    """
    计算模型的FLOPs
    """
    try:
        x_enc, x_mark_enc, x_dec, x_mark_dec = sample_input

        # 确保所有输入都在正确的设备上（这里使用第一个输入的设备）
        device = x_enc.device if hasattr(x_enc, 'device') else torch.device('cpu')
        x_enc = x_enc.to(device)
        x_mark_enc = x_mark_enc.to(device)
        x_dec = x_dec.to(device)
        x_mark_dec = x_mark_dec.to(device)

        # 创建模型包装器
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                # 确保输入维度正确
                out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return out[0] if isinstance(out, (tuple, list)) else out

        model.eval()
        wrapper = ModelWrapper(model).to(device)

        # 使用thop计算FLOPs
        flops, params = profile(
            wrapper,
            inputs=(x_enc, x_mark_enc, x_dec, x_mark_dec),
            verbose=False
        )

        print(f"Calculated FLOPs: {flops}, Params: {params}")

        # 检查结果是否有效
        if flops is None or flops <= 0:
            print("Warning: FLOPs calculation returned invalid value")
            return 0.0

        return flops / 1e9  # 转换为GFLOPs
    except Exception as e:
        print(f"Warning: FLOPs calculation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0  # 返回默认值避免程序崩溃


# ============================
# 测试函数
# ============================
def test_crossgnn(model, test_loader, device, configs):
    model.eval()

    model_params = count_parameters(model)

    import io
    model_buffer = io.BytesIO()
    torch.save(model.state_dict(), model_buffer)
    model_size = len(model_buffer.getvalue()) / (1024 * 1024)

    # 测量FLOPs
    sample_input = next(iter(test_loader))
    x_enc_sample, x_mark_enc_sample = sample_input[0][:1].to(device), sample_input[1][:1].to(device)
    x_dec_sample, x_mark_dec_sample = x_enc_sample[:, -configs.pred_len:, :], x_mark_enc_sample[:, -configs.pred_len:,
                                                                              :]

    model_flops = compute_flops(model, (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample))

    # 测量推理时间
    model_inference_time = measure_inference_time(model,
                                                  (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample),
                                                  device)

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]
            x_dec, x_mark_dec = x_enc[:, -configs.pred_len:, :], x_mark_enc[:, -configs.pred_len:, :]

            pred, _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            preds.append(pred.cpu())
            targets.append(y_true.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    mae = MAE(preds, targets)
    mse = MSE(preds, targets)
    rmse = RMSE(preds, targets)
    mape = MAPE(preds, targets)

    print("\n=== CrossGNN Model Test Results ===")
    print(f"Model:")
    print(f"  Parameters: {model_params:,}")
    print(f"  Size: {model_size:.2f} MB")
    print(f"  FLOPs: {model_flops:.2f} GFLOPs")
    print(f"  Inference Time: {model_inference_time:.2f} ms")
    print(f"  Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

    results = {
        'model': {
            'params': model_params,
            'size': model_size,
            'flops': model_flops,
            'inference_time': model_inference_time,
            'metrics': (mae, mse, rmse, mape)
        }
    }
    return results


# ============================
# 主流程
# ============================
if __name__ == "__main__":
    model = train_crossgnn(num_epochs=100, lr=1e-3, batch_size=32, pruning_ratio=0)

    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() and configs.use_gpu else 'cpu')
    model = CrossGNN(configs).to(device)

    if os.path.exists('checkpoints/crossgnn_model.pth'):
        model.load_state_dict(torch.load('checkpoints/crossgnn_model.pth', map_location=device))

    train_data, test_data = load_and_split_data()
    test_x, test_x_mark, test_y = create_time_series_dataset(test_data, configs.seq_len, configs.pred_len)
    test_x, test_x_mark, test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_x_mark,
                                                                                          dtype=torch.float32), torch.tensor(
        test_y, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_x_mark, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)

    test_results = test_crossgnn(model, test_loader, device, configs)

    print("\n=== CrossGNN Test Results ===")
    model_results = test_results['model']
    print(f"Parameters: {model_results['params']:,}")
    print(f"Size: {model_results['size']:.2f} MB")
    print(f"FLOPs: {model_results['flops']:.2f} GFLOPs")
    print(f"Inference Time: {model_results['inference_time']:.2f} ms")
    mae, mse, rmse, mape = model_results['metrics']
    print(f"Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
