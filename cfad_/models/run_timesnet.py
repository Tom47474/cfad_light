# cfad_/models/run_timesnet.py
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

from cfad_.dataset.data_processor import load_and_split_data
from cfad_.utils.Config import configs
from cfad_.utils.metrics import MAE, MSE, RMSE, MAPE
from cfad_.utils.tools import plot_model_comparison, plot_speed_accuracy_tradeoff, count_parameters, \
    measure_inference_time, EarlyStopping

from model_timesnet import Model  # 导入TimesNet模型
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
        # 根据freq参数生成相应的时间特征
        # 'h'表示小时频率，需要生成month, day, weekday, hour特征
        month = dates.dt.month / 12.0 - 0.5
        day = dates.dt.day / 31.0 - 0.5
        weekday = dates.dt.dayofweek / 6.0 - 0.5
        hour = dates.dt.hour / 23.0 - 0.5
        # 按照TimeFeatureEmbedding期望的顺序排列: month, day, weekday, hour
        x_mark_data = np.column_stack([month, day, weekday, hour])
    else:
        # 如果没有日期列，创建默认的时间标记 (batch_size, seq_len, 4)
        x_mark_data = np.zeros((len(data), 4))

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
# 绘图工具（loss curve）
# ============================
def plot_loss_curve(train_losses, val_losses, save_path='timesnet_loss_curve.png'):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=1)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=1)
    plt.title('TimesNet Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to {save_path}")


# ============================
# TimesNet 训练
# ============================
def train_timesnet(num_epochs, lr=1e-3, batch_size=32):
    set_seed(2021)
    device = torch.device('cuda' if torch.cuda.is_available() and configs.use_gpu else 'cpu')

    # 数据加载
    train_data, test_data = load_and_split_data()
    seq_len = configs.seq_len
    pred_len = configs.pred_len

    train_x, train_x_mark, train_y = create_time_series_dataset(train_data, seq_len, pred_len)
    test_x, test_x_mark, test_y = create_time_series_dataset(test_data, seq_len, pred_len)

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_x_mark = torch.tensor(train_x_mark, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_x_mark = torch.tensor(test_x_mark, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    val_size = int(0.1 * len(train_x))
    train_size = len(train_x) - val_size

    train_dataset = torch.utils.data.TensorDataset(train_x[:train_size], train_x_mark[:train_size],
                                                   train_y[:train_size])
    val_dataset = torch.utils.data.TensorDataset(train_x[train_size:], train_x_mark[train_size:], train_y[train_size:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型初始化
    model = Model(configs).to(device)

    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True, delta=0)

    train_losses = []
    val_losses = []
    best_state = None
    best_val = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]
            # 修改：x_dec和x_mark_dec应该与y_true形状一致
            x_dec = torch.zeros_like(y_true).to(device)
            x_mark_dec = torch.zeros_like(x_mark_enc[:, -pred_len:, :]).to(device)

            pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

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
                # 修改：x_dec和x_mark_dec应该与y_true形状一致
                x_dec = torch.zeros_like(y_true).to(device)
                x_mark_dec = torch.zeros_like(x_mark_enc[:, -pred_len:, :]).to(device)

                pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                val_loss += F.mse_loss(pred, y_true).item()

        avg_train = total_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
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

    # 保存最佳模型
    os.makedirs('checkpoints', exist_ok=True)
    if best_state is not None:
        torch.save(best_state, 'checkpoints/timesnet_best.pth')
        model.load_state_dict(best_state)

    # 保存训练曲线
    loss_df = pd.DataFrame(
        {'epoch': range(1, len(train_losses) + 1), 'train_loss': train_losses, 'val_loss': val_losses})
    loss_df.to_csv('timesnet_training_losses.csv', index=False)
    # plot_loss_curve(train_losses, val_losses, 'timesnet_training_loss_curve.png')

    torch.save(model.state_dict(), 'checkpoints/timesnet_model.pth')
    print("Saved TimesNet model to checkpoints/timesnet_model.pth")
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
                return out

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
def test_timesnet(model, test_loader, device, configs):
    model.eval()

    model_params = count_parameters(model)

    import io
    model_buffer = io.BytesIO()
    torch.save(model.state_dict(), model_buffer)
    model_size = len(model_buffer.getvalue()) / (1024 * 1024)

    # 测量FLOPs
    sample_input = next(iter(test_loader))
    x_enc_sample, x_mark_enc_sample = sample_input[0][:1].to(device), sample_input[1][:1].to(device)
    x_dec_sample = torch.zeros(x_enc_sample.shape[0], configs.pred_len, x_enc_sample.shape[2]).to(device)
    x_mark_dec_sample = torch.zeros(x_enc_sample.shape[0], configs.pred_len, x_mark_enc_sample.shape[2]).to(device)

    model_flops = compute_flops(model, (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample))

    # 测量推理时间
    model_inference_time = measure_inference_time(model,
                                                  (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample),
                                                  device)

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]
            x_dec = torch.zeros_like(y_true).to(device)
            x_mark_dec = torch.zeros_like(x_mark_enc[:, -configs.pred_len:, :]).to(device)

            pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            preds.append(pred.cpu())
            targets.append(y_true.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    mae = MAE(preds, targets)
    mse = MSE(preds, targets)
    rmse = RMSE(preds, targets)
    mape = MAPE(preds, targets)

    print("\n=== TimesNet Model Test Results ===")
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
# 主流程示例（脚本运行）
# ============================
if __name__ == "__main__":
    # 训练TimesNet模型
    timesnet_model = train_timesnet(num_epochs=100, lr=1e-3, batch_size=32)

    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() and configs.use_gpu else 'cpu')
    timesnet_model = Model(configs).to(device)

    # 加载最佳模型权重
    if os.path.exists('checkpoints/timesnet_best.pth'):
        timesnet_model.load_state_dict(torch.load('checkpoints/timesnet_best.pth', map_location=device))
        print("Loaded best TimesNet model weights")

    # 加载测试数据
    test_data, _ = load_and_split_data()
    test_x, test_x_mark, test_y = create_time_series_dataset(test_data, configs.seq_len, configs.pred_len)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_x_mark = torch.tensor(test_x_mark, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_x_mark, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)

    # 测试模型
    test_results = test_timesnet(timesnet_model, test_loader, device, configs)

    print("\nTimesNet training and testing completed!")
