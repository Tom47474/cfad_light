import math

import numpy as np
import torch


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "constant":
        lr_adjust = {}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


# utils/tools.py

import time


def count_parameters(model):
    """
    计算模型参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# utils/tools.py

def measure_inference_time(model, input_data, device, num_runs=100):
    """
    测量模型推理时间
    """
    model.eval()
    model.to(device)

    # 解包输入数据并移动到设备
    x_enc, x_mark_enc, x_dec, x_mark_dec = [d.to(device) for d in input_data]

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    # 实际测量
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
    return avg_inference_time



def get_model_size(model_path):
    """
    获取模型文件大小
    """
    import os
    return os.path.getsize(model_path) / (1024 * 1024)  # 转换为MB


# utils/tools.py

import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_comparison(comparison_results):
    """
    可视化模型比较结果
    """
    teacher_data = comparison_results['teacher']
    student_data = comparison_results['student']

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 参数量对比
    params = [teacher_data['params'], student_data['params']]
    labels = ['Teacher', 'Student']
    axes[0, 0].bar(labels, params, color=['blue', 'orange'])
    axes[0, 0].set_title('Model Parameters')
    axes[0, 0].set_ylabel('Number of Parameters')
    for i, v in enumerate(params):
        axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom')

    # 模型大小对比
    sizes = [teacher_data['size'], student_data['size']]
    axes[0, 1].bar(labels, sizes, color=['blue', 'orange'])
    axes[0, 1].set_title('Model Size')
    axes[0, 1].set_ylabel('Size (MB)')
    for i, v in enumerate(sizes):
        axes[0, 1].text(i, v, f'{v:.2f}MB', ha='center', va='bottom')

    # 推理时间对比
    inference_times = [teacher_data['inference_time'], student_data['inference_time']]
    axes[1, 0].bar(labels, inference_times, color=['blue', 'orange'])
    axes[1, 0].set_title('Inference Time')
    axes[1, 0].set_ylabel('Time (ms)')
    for i, v in enumerate(inference_times):
        axes[1, 0].text(i, v, f'{v:.2f}ms', ha='center', va='bottom')

    # 精度指标对比
    metrics_names = ['MAE', 'MSE', 'RMSE']
    teacher_metrics = teacher_data['metrics'][:3]
    student_metrics = student_data['metrics'][:3]

    x = range(len(metrics_names))
    width = 0.35

    axes[1, 1].bar([i - width / 2 for i in x], teacher_metrics, width, label='Teacher', color='blue')
    axes[1, 1].bar([i + width / 2 for i in x], student_metrics, width, label='Student', color='orange')
    axes[1, 1].set_title('Accuracy Metrics')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_speed_accuracy_tradeoff(comparison_results):
    """
    绘制速度-精度权衡图
    """
    teacher_data = comparison_results['teacher']
    student_data = comparison_results['student']

    inference_times = [teacher_data['inference_time'], student_data['inference_time']]
    mae_values = [teacher_data['metrics'][0], student_data['metrics'][0]]
    labels = ['Teacher', 'Student']

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(inference_times, mae_values, s=100, c=['blue', 'orange'])
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('MAE')
    plt.title('Speed-Accuracy Tradeoff')

    # 添加标签
    for i, label in enumerate(labels):
        plt.annotate(label, (inference_times[i], mae_values[i]),
                     xytext=(5, 5), textcoords='offset points')

    plt.grid(True, alpha=0.3)
    plt.savefig('speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
