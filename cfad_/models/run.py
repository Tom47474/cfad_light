# run.py
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cfad_.dataset.data_processor import load_and_split_data  # 你已有的数据处理方法
from cfad_.utils.Config import configs
from model import CFAD_Teacher, CFAD_Student, distillation_loss  # 上面写的模型
from cfad_.utils.metrics import MAE, MSE, RMSE, MAPE  # 导入所有指标函数
from cfad_.utils.tools import plot_model_comparison, plot_speed_accuracy_tradeoff, count_parameters, \
    measure_inference_time, EarlyStopping
import torch.nn.utils.prune as prune


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
# 创建时间序列数据集
# ============================
def create_time_series_dataset(data, seq_len, pred_len):
    """
    将时间序列数据转换为模型输入格式
    """
    x_data = []
    y_data = []
    x_mark = []

    # 生成时间特征
    if 'date' in data.columns:
        dates = pd.to_datetime(data['date'])
        # 使用 .dt 访问器获取时间属性
        hour = dates.dt.hour / 23.0 - 0.5
        day_of_week = dates.dt.dayofweek / 6.0 - 0.5
        x_mark_data = np.column_stack([hour, day_of_week])
    else:
        # 如果没有日期列，创建简单的时间标记
        x_mark_data = np.zeros((len(data), 2))

    # 假设除了日期列外，其他列都是特征
    feature_columns = [col for col in data.columns if col != 'date']

    # 确保至少有一个特征列
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found in data")

    for i in range(seq_len, len(data) - pred_len + 1):
        # 输入序列
        x_seq = data[feature_columns].iloc[i - seq_len:i].values
        x_data.append(x_seq)

        # 目标序列
        y_seq = data[feature_columns].iloc[i:i + pred_len].values
        y_data.append(y_seq)

        # 时间标记
        x_mark_seq = x_mark_data[i - seq_len:i]
        x_mark.append(x_mark_seq)

    return np.array(x_data), np.array(x_mark), np.array(y_data)


def prune_model(model, pruning_ratio=0.3):
    """
    对模型进行剪枝

    Args:
        model: 要剪枝的模型
        pruning_ratio: 剪枝比例 (0-1之间)

    Returns:
        None (原地修改模型)
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 对线性层和卷积层进行剪枝
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            # 如果你也想剪枝bias，可以取消下面这行的注释
            # prune.l1_unstructured(module, name='bias', amount=pruning_ratio)


def remove_pruning(model):
    """
    移除剪枝的重新参数化，使模型真正稀疏化

    Args:
        model: 已剪枝的模型

    Returns:
        None (原地修改模型)
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            try:
                prune.remove(module, 'weight')
                # 如果剪枝了bias，也需要移除
                # prune.remove(module, 'bias')
            except ValueError:
                # 如果模块没有被剪枝，则跳过
                pass


def get_sparsity(model):
    """
    计算模型的稀疏度

    Args:
        model: 模型

    Returns:
        float: 稀疏度 (0-1之间)
    """
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


def plot_loss_curve(train_losses, val_losses, save_path='loss_curve.png'):
    """
    绘制训练和验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=1)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=1)

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Loss curve saved to {save_path}")


# ============================
# 训练 Student 模型
# ============================
def train_student(teacher_path, num_epochs, lr=1e-3, batch_size=32, alpha=0.5, pruning_ratio=0.0):
    # -------------------------
    # 1. 设置环境
    # -------------------------
    set_seed(2021)
    device = torch.device('cuda' if torch.cuda.is_available() and configs.use_gpu else 'cpu')

    # -------------------------
    # 2. 数据准备
    # -------------------------
    train_data, test_data = load_and_split_data()

    # 创建时间序列格式的数据
    seq_len = configs.seq_len
    pred_len = configs.pred_len

    train_x, train_x_mark, train_y = create_time_series_dataset(train_data, seq_len, pred_len)
    test_x, test_x_mark, test_y = create_time_series_dataset(test_data, seq_len, pred_len)

    # 转换为tensor
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_x_mark = torch.tensor(train_x_mark, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_x_mark = torch.tensor(test_x_mark, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    # 创建训练和验证数据集 (使用部分训练数据作为验证集)
    val_size = int(0.1 * len(train_x))  # 使用10%的训练数据作为验证集
    train_size = len(train_x) - val_size

    # 分割训练和验证数据
    train_dataset = torch.utils.data.TensorDataset(
        train_x[:train_size], train_x_mark[:train_size], train_y[:train_size])
    val_dataset = torch.utils.data.TensorDataset(
        train_x[train_size:], train_x_mark[train_size:], train_y[train_size:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 3. 初始化模型
    # -------------------------
    teacher = CFAD_Teacher(configs).to(device)
    student = CFAD_Student(configs).to(device)

    # 加载已训练 Teacher
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        print(f"成功加载教师模型: {teacher_path}")
    else:
        print(f"警告: 未找到教师模型文件 {teacher_path}")
        # 如果没有教师模型，可以考虑先训练一个或者使用随机初始化
        # 这里我们继续执行，但实际应用中应该确保有预训练的教师模型

    teacher.eval()  # Teacher 不训练

    # -------------------------
    # 4. 优化器和早停机制
    # -------------------------
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True, delta=0)

    # -------------------------
    # 5. 开始训练
    # -------------------------
    # 记录损失值
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        student.train()
        train_loss = 0.0
        for batch in train_loader:
            x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]

            # 创建 decoder 输入 (使用 encoder 最后几个时间步作为 decoder 输入)
            x_dec = x_enc[:, -pred_len:, :]
            x_mark_dec = x_mark_enc[:, -pred_len:, :]

            # Teacher 输出
            with torch.no_grad():
                teacher_out = teacher(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # Student 输出
            student_out = student(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # 蒸馏损失
            loss = distillation_loss(student_out, teacher_out, y_true, alpha=alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]

                # 创建 decoder 输入
                x_dec = x_enc[:, -pred_len:, :]
                x_mark_dec = x_mark_enc[:, -pred_len:, :]

                # Teacher 输出
                teacher_out = teacher(x_enc, x_mark_enc, x_dec, x_mark_dec)

                # Student 输出
                student_out = student(x_enc, x_mark_enc, x_dec, x_mark_dec)

                # 验证损失 (只计算任务损失，不计算知识蒸馏损失)
                loss = F.mse_loss(student_out, y_true)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 记录损失
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step()
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # 检查早停条件
        early_stopping(avg_val_loss, student, 'checkpoints')

        if early_stopping.early_stop:
            print("Early stopping triggered")
            # 截断损失记录到早停的epoch
            train_losses = train_losses[:epoch]
            val_losses = val_losses[:epoch]
            break

    # 保存损失值到CSV文件
    loss_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_df.to_csv('training_losses.csv', index=False)
    print("Training losses saved to training_losses.csv")

    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, 'training_loss_curve.png')

    # 加载最佳模型权重
    if os.path.exists('checkpoints/checkpoint.pth'):
        student.load_state_dict(torch.load('checkpoints/checkpoint.pth'))
        print("Loaded best model weights from checkpoint")

    # -------------------------
    # 6. 保存 Student 模型
    # -------------------------
    student_save_path = os.path.join('checkpoints', 'student_model.pth')
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(student.state_dict(), student_save_path)
    print(f"Student model saved to {student_save_path}")

    # -------------------------
    # 7. 测试 Student 模型（可选）
    # -------------------------
    student.eval()
    with torch.no_grad():
        x_dec_test = test_x[:, -pred_len:, :]
        x_mark_dec_test = test_x_mark[:, -pred_len:, :]
        pred = student(test_x.to(device), test_x_mark.to(device), x_dec_test.to(device), x_mark_dec_test.to(device))
        print("Test output shape:", pred.shape)

    # 在训练完成后进行剪枝（如果需要）
    if pruning_ratio > 0:
        print(f"Pruning student model with ratio {pruning_ratio}")
        prune_model(student, pruning_ratio)
        sparsity = get_sparsity(student)
        print(f"Model sparsity after pruning: {sparsity:.2%}")

        # 保存剪枝后的模型
        student_save_path = os.path.join('checkpoints', f'student_model_pruned_{pruning_ratio}.pth')
        os.makedirs('checkpoints', exist_ok=True)
        # 移除剪枝的重新参数化
        remove_pruning(student)
        torch.save(student.state_dict(), student_save_path)
        print(f"Pruned student model saved to {student_save_path}")

    return student


# models/run.py

def compare_models(teacher_model, student_model, test_loader, device, configs):
    """
    比较教师模型和学生模型的性能
    """
    teacher_model.eval()
    student_model.eval()

    # 计算参数量
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)

    # 通过序列化计算模型大小（以MB为单位）
    import io
    teacher_buffer = io.BytesIO()
    torch.save(teacher_model.state_dict(), teacher_buffer)
    teacher_size = len(teacher_buffer.getvalue()) / (1024 * 1024)  # 转换为MB

    student_buffer = io.BytesIO()
    torch.save(student_model.state_dict(), student_buffer)
    student_size = len(student_buffer.getvalue()) / (1024 * 1024)  # 转换为MB

    # 测量推理时间
    sample_input = next(iter(test_loader))
    x_enc_sample = sample_input[0][:1].to(device)  # 取一个样本
    x_mark_enc_sample = sample_input[1][:1].to(device)
    x_dec_sample = x_enc_sample[:, -configs.pred_len:, :].to(device)
    x_mark_dec_sample = x_mark_enc_sample[:, -configs.pred_len:, :].to(device)

    # 构造完整的输入元组
    teacher_input = (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample)
    student_input = (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample)

    teacher_inference_time = measure_inference_time(
        teacher_model,
        teacher_input,
        device
    )

    student_inference_time = measure_inference_time(
        student_model,
        student_input,
        device
    )

    # 计算精度指标
    teacher_preds, student_preds, targets = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_enc, x_mark_enc, y_true = [b.to(device) for b in batch]
            x_dec = x_enc[:, -configs.pred_len:, :]
            x_mark_dec = x_mark_enc[:, -configs.pred_len:, :]

            # 正确调用模型的前向传播
            teacher_pred = teacher_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            student_pred = student_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            teacher_preds.append(teacher_pred.cpu())
            student_preds.append(student_pred.cpu())
            targets.append(y_true.cpu())

    teacher_preds = torch.cat(teacher_preds)
    student_preds = torch.cat(student_preds)
    targets = torch.cat(targets)

    # 使用metrics.py中的函数计算指标
    teacher_mae = MAE(teacher_preds, targets)
    teacher_mse = MSE(teacher_preds, targets)
    teacher_rmse = RMSE(teacher_preds, targets)
    teacher_mape = MAPE(teacher_preds, targets)

    student_mae = MAE(student_preds, targets)
    student_mse = MSE(student_preds, targets)
    student_rmse = RMSE(student_preds, targets)
    student_mape = MAPE(student_preds, targets)

    # 打印比较结果
    print("\n=== Model Comparison Results ===")
    print(f"Teacher Model:")
    print(f"  Parameters: {teacher_params:,}")
    print(f"  Size: {teacher_size:.2f} MB")
    print(f"  Inference Time: {teacher_inference_time:.2f} ms")
    print(
        f"  Metrics - MAE: {teacher_mae:.4f}, MSE: {teacher_mse:.4f}, RMSE: {teacher_rmse:.4f}, MAPE: {teacher_mape:.4f}%")

    print(f"\nStudent Model:")
    print(f"  Parameters: {student_params:,} ({(student_params / teacher_params) * 100:.1f}% of teacher)")
    print(f"  Size: {student_size:.2f} MB ({(student_size / teacher_size) * 100:.1f}% of teacher)")
    print(
        f"  Inference Time: {student_inference_time:.2f} ms ({(student_inference_time / teacher_inference_time) * 100:.1f}% of teacher)")
    print(
        f"  Metrics - MAE: {student_mae:.4f}, MSE: {student_mse:.4f}, RMSE: {student_rmse:.4f}, MAPE: {student_mape:.4f}%")

    # 计算压缩比等额外指标
    compression_ratio_params = teacher_params / student_params
    compression_ratio_size = teacher_size / student_size
    speedup_ratio = teacher_inference_time / student_inference_time

    print("\n=== Edge Device Suitability Analysis ===")
    print(f"Parameter Compression Ratio: {compression_ratio_params:.2f}x")
    print(f"Model Size Compression Ratio: {compression_ratio_size:.2f}x")
    print(f"Inference Speedup Ratio: {speedup_ratio:.2f}x")

    # 边缘设备适用性评估
    if student_size < 10 and student_inference_time < 100:
        print("✅ Student model is suitable for edge deployment")
        print("   - Model size < 10MB")
        print("   - Inference time < 100ms")
    else:
        print("⚠️  Student model may not be suitable for edge deployment")
        if student_size >= 10:
            print("   - Model size is large (≥10MB)")
        if student_inference_time >= 100:
            print("   - Inference time is slow (≥100ms)")

    # 更新返回结果
    results = {
        'teacher': {
            'params': teacher_params,
            'size': teacher_size,
            'inference_time': teacher_inference_time,
            'metrics': (teacher_mae, teacher_mse, teacher_rmse, teacher_mape)
        },
        'student': {
            'params': student_params,
            'size': student_size,
            'inference_time': student_inference_time,
            'metrics': (student_mae, student_mse, student_rmse, student_mape)
        },
        'ratios': {
            'compression_params': compression_ratio_params,
            'compression_size': compression_ratio_size,
            'speedup': speedup_ratio
        }
    }
    return results


# ============================
# 主函数
# ============================
if __name__ == "__main__":
    teacher_model_path = '../checkpoints/teacher_model.pth'
    student_model = train_student(teacher_model_path, num_epochs=500, lr=1e-3, batch_size=32, alpha=0.5,
                                  pruning_ratio=0)

    # 加载训练好的模型进行比较
    device = torch.device('cuda' if torch.cuda.is_available() and configs.use_gpu else 'cpu')

    # 直接使用训练好的模型，不从文件加载
    teacher = CFAD_Teacher(configs).to(device)
    student = CFAD_Student(configs).to(device)

    # 注意：这里我们不加载预训练权重，而是使用刚刚训练好的student模型
    # teacher模型会在compare_models函数中随机初始化并用于对比

    # 创建测试数据加载器
    test_data, _ = load_and_split_data()
    test_x, test_x_mark, test_y = create_time_series_dataset(test_data, configs.seq_len, configs.pred_len)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_x_mark = torch.tensor(test_x_mark, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_x_mark, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)

    # 比较模型
    comparison_results = compare_models(teacher, student, test_loader, device, configs)

    # 可视化结果
    plot_model_comparison(comparison_results)
    plot_speed_accuracy_tradeoff(comparison_results)
