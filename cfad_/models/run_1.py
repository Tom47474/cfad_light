import os
import random
import time
import copy
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

from model_1 import CFAD_Teacher, CFAD_Student, distillation_loss, HiddenProj
from dlinear import DLinear

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
# Student 训练
# ============================
def train_student(teacher_path, num_epochs, lr=1e-3, batch_size=32, alpha=0.5, beta=0.3, T=2.0, pruning_ratio=0.0):
    set_seed(2021)
    device = torch.device('cuda' if torch.cuda.is_available() and configs.use_gpu else 'cpu')

    train_data, test_data = load_and_split_data()
    seq_len, pred_len = configs.seq_len, configs.pred_len

    train_x, train_x_mark, train_y = create_time_series_dataset(train_data, seq_len, pred_len)
    test_x, test_x_mark, test_y = create_time_series_dataset(test_data, seq_len, pred_len)

    train_x, train_x_mark, train_y = torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_x_mark, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)
    test_x, test_x_mark, test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_x_mark, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)

    val_size = int(0.1 * len(train_x))
    train_size = len(train_x) - val_size

    train_dataset = torch.utils.data.TensorDataset(train_x[:train_size], train_x_mark[:train_size], train_y[:train_size])
    val_dataset = torch.utils.data.TensorDataset(train_x[train_size:], train_x_mark[train_size:], train_y[train_size:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    teacher, student = CFAD_Teacher(configs).to(device), CFAD_Student(configs).to(device)
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        print(f"Loaded teacher: {teacher_path}")
    teacher.eval()

    hidden_proj = None
    if hasattr(teacher, 'hidden_dim') and hasattr(student, 'hidden_dim'):
        if teacher.hidden_dim != student.hidden_dim:
            hidden_proj = HiddenProj(teacher.hidden_dim, student.hidden_dim).to(device)

    params = list(student.parameters())
    if hidden_proj is not None:
        params += list(hidden_proj.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True, delta=0)

    train_losses, val_losses = [], []
    best_state, best_val = None, float('inf')

    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        for batch in train_loader:
            x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]
            x_dec, x_mark_dec = x_enc[:, -pred_len:, :], x_mark_enc[:, -pred_len:, :]

            with torch.no_grad():
                teacher_pred, teacher_hidden = teacher(x_enc, x_mark_enc, x_dec, x_mark_dec)

            student_pred, student_hidden = student(x_enc, x_mark_enc, x_dec, x_mark_dec)

            loss, _, _, _ = distillation_loss(
                student_pred, teacher_pred, y_true,
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                hidden_proj=hidden_proj,
                alpha=alpha, beta=beta, T=T
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]
                x_dec, x_mark_dec = x_enc[:, -pred_len:, :], x_mark_enc[:, -pred_len:, :]
                student_pred, _ = student(x_enc, x_mark_enc, x_dec, x_mark_dec)
                val_loss += F.mse_loss(student_pred, y_true).item()

        avg_train, avg_val = total_loss / max(1, len(train_loader)), val_loss / max(1, len(val_loader))
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}")

        if avg_val < best_val:
            best_val, best_state = avg_val, {'student': student.state_dict(), 'hidden_proj': hidden_proj.state_dict() if hidden_proj is not None else None}

        early_stopping(avg_val, student, 'checkpoints')
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 保存最佳
    os.makedirs('checkpoints', exist_ok=True)
    if best_state is not None:
        torch.save(best_state['student'], 'checkpoints/student_best.pth')
        if best_state['hidden_proj'] is not None:
            torch.save(best_state['hidden_proj'], 'checkpoints/hidden_proj.pth')
        student.load_state_dict(best_state['student'])

    plot_loss_curve(train_losses, val_losses, 'training_loss_curve.png')

    if pruning_ratio > 0:
        prune_model(student, pruning_ratio)
        print(f"Sparsity after pruning: {get_sparsity(student):.2%}")
        remove_pruning(student)
        torch.save(student.state_dict(), f'checkpoints/student_pruned_{pruning_ratio}.pth')

    torch.save(student.state_dict(), 'checkpoints/student_model.pth')
    print("Saved student model to checkpoints/student_model.pth")
    return student


from thop import profile

# ===== 自定义 FLOPs hook =====
def count_conv1d(m, x, y):
    # x[0] shape: [B, C_in, L]
    x = x[0]
    batch_size = x.shape[0]
    cin = m.in_channels
    cout = m.out_channels
    k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
    out_len = y.shape[-1]
    flops = batch_size * 2 * cin * cout * k * out_len
    m.total_ops += torch.DoubleTensor([flops])

def count_linear(m, x, y):
    x = x[0]
    batch_size = x.shape[0]
    in_f = m.in_features
    out_f = m.out_features
    flops = batch_size * 2 * in_f * out_f
    m.total_ops += torch.DoubleTensor([flops])

def count_starcnn(m, x, y):
    # 简化估算：把内部 conv_aggregate 和 conv_fuse 的 FLOPs 全算
    flops = 0
    for sub in m.conv_aggregate:
        if isinstance(sub, nn.Conv1d):
            flops += 2 * sub.in_channels * sub.out_channels * sub.kernel_size[0] * y.shape[-2]
    for sub in m.conv_fuse:
        if isinstance(sub, nn.Conv1d):
            flops += 2 * sub.in_channels * sub.out_channels * sub.kernel_size[0] * y.shape[-2]
    m.total_ops += torch.DoubleTensor([flops])

def count_encoderlayer(m, x, y):
    # 只估算 attention + FFN 里的 conv
    flops = 0
    if hasattr(m, 'attention') and hasattr(m.attention, 'conv_aggregate'):
        for sub in m.attention.conv_aggregate:
            if isinstance(sub, nn.Conv1d):
                flops += 2 * sub.in_channels * sub.out_channels * sub.kernel_size[0] * y.shape[-2]
        for sub in m.attention.conv_fuse:
            if isinstance(sub, nn.Conv1d):
                flops += 2 * sub.in_channels * sub.out_channels * sub.kernel_size[0] * y.shape[-2]
    if hasattr(m, 'conv1'):
        flops += 2 * m.conv1.in_channels * m.conv1.out_channels * m.conv1.kernel_size[0] * y.shape[-2]
    if hasattr(m, 'conv2'):
        flops += 2 * m.conv2.in_channels * m.conv2.out_channels * m.conv2.kernel_size[0] * y.shape[-2]
    m.total_ops += torch.DoubleTensor([flops])

# ===== 计算 FLOPs =====
# 替换你现有的compute_flops函数为以下版本
# 修改函数定义，移除device参数
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
# 模型对比
# ============================
def compare_models(teacher_model, student_model, test_loader, device, configs):
    teacher_model.eval()
    student_model.eval()

    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)

    import io
    teacher_buffer, student_buffer = io.BytesIO(), io.BytesIO()
    torch.save(teacher_model.state_dict(), teacher_buffer)
    torch.save(student_model.state_dict(), student_buffer)
    teacher_size = len(teacher_buffer.getvalue()) / (1024 * 1024)
    student_size = len(student_buffer.getvalue()) / (1024 * 1024)

    sample_input = next(iter(test_loader))
    x_enc_sample, x_mark_enc_sample = sample_input[0][:1].to(device), sample_input[1][:1].to(device)
    x_dec_sample, x_mark_dec_sample = x_enc_sample[:, -configs.pred_len:, :], x_mark_enc_sample[:, -configs.pred_len:, :]

    teacher_flops = compute_flops(teacher_model, (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample))
    student_flops = compute_flops(student_model, (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample))

    teacher_inference_time = measure_inference_time(teacher_model, (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample), device)
    student_inference_time = measure_inference_time(student_model, (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample), device)


    teacher_preds, student_preds, targets = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]
            x_dec, x_mark_dec = x_enc[:, -configs.pred_len:, :], x_mark_enc[:, -configs.pred_len:, :]
            teacher_pred, _ = teacher_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            student_pred, _ = student_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            teacher_preds.append(teacher_pred.cpu())
            student_preds.append(student_pred.cpu())
            targets.append(y_true.cpu())

    teacher_preds, student_preds, targets = torch.cat(teacher_preds), torch.cat(student_preds), torch.cat(targets)

    teacher_mae, teacher_mse, teacher_rmse, teacher_mape = MAE(teacher_preds, targets), MSE(teacher_preds, targets), RMSE(teacher_preds, targets), MAPE(teacher_preds, targets)
    student_mae, student_mse, student_rmse, student_mape = MAE(student_preds, targets), MSE(student_preds, targets), RMSE(student_preds, targets), MAPE(student_preds, targets)

    print("\n=== Model Comparison Results ===")
    print(f"Teacher Model:")
    print(f"  Parameters: {teacher_params:,}")
    print(f"  Size: {teacher_size:.2f} MB")
    print(f"  FLOPs: {teacher_flops:.2f} GFLOPs")
    print(f"  Inference Time: {teacher_inference_time:.2f} ms")
    print(f"  Metrics - MAE: {teacher_mae:.4f}, MSE: {teacher_mse:.4f}, RMSE: {teacher_rmse:.4f}, MAPE: {teacher_mape:.4f}%")

    print(f"\nStudent Model:")
    print(f"  Parameters: {student_params:,} ({(student_params / teacher_params) * 100:.1f}% of teacher)")
    print(f"  Size: {student_size:.2f} MB ({(student_size / teacher_size) * 100:.1f}% of teacher)")
    print(f"  FLOPs: {student_flops:.2f} GFLOPs ({(student_flops / teacher_flops) * 100:.1f}% of teacher)")
    print(f"  Inference Time: {student_inference_time:.2f} ms ({(student_inference_time / teacher_inference_time) * 100:.1f}% of teacher)")
    print(f"  Metrics - MAE: {student_mae:.4f}, MSE: {student_mse:.4f}, RMSE: {student_rmse:.4f}, MAPE: {student_mape:.4f}%")

    results = {
        'teacher': {'params': teacher_params, 'size': teacher_size, 'flops': teacher_flops, 'inference_time': teacher_inference_time, 'metrics': (teacher_mae, teacher_mse, teacher_rmse, teacher_mape)},
        'student': {'params': student_params, 'size': student_size, 'flops': student_flops, 'inference_time': student_inference_time, 'metrics': (student_mae, student_mse, student_rmse, student_mape)},
        'ratios': {
            'compression_params': teacher_params / student_params,
            'compression_size': teacher_size / student_size,
            'compression_flops': teacher_flops / student_flops if student_flops > 0 else 0,
            'speedup': teacher_inference_time / student_inference_time
        }
    }
    return results


# ============================
# 主流程
# ============================
if __name__ == "__main__":
    teacher_model_path = os.path.join('../checkpoints', 'teacher_model_2.pth')
    student_model = train_student(teacher_model_path, num_epochs=200, lr=1e-3, batch_size=32, alpha=0.5, beta=0.3, T=2.0, pruning_ratio=0)

    device = torch.device('cuda' if torch.cuda.is_available() and configs.use_gpu else 'cpu')
    teacher, student = CFAD_Teacher(configs).to(device), CFAD_Student(configs).to(device)

    if os.path.exists(teacher_model_path):
        teacher.load_state_dict(torch.load(teacher_model_path, map_location=device))

    test_data, _ = load_and_split_data()
    test_x, test_x_mark, test_y = create_time_series_dataset(test_data, configs.seq_len, configs.pred_len)
    test_x, test_x_mark, test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_x_mark, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_x_mark, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)

    comparison_results = compare_models(teacher, student, test_loader, device, configs)
    plot_model_comparison(comparison_results)
    plot_speed_accuracy_tradeoff(comparison_results)
