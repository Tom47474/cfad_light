# cfad_1/run_model_comparison.py
import os
import torch
import time
import io
import numpy as np
import pandas as pd
import torch.nn as nn
from exp.exp_custom import Exp_Custom
from utils.Config import configs
from dataset.data_processor import load_and_split_data

# 设置随机种子
def set_seed(seed=2021):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """
    计算模型参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """
    获取模型文件大小
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = len(buffer.getvalue()) / (1024 * 1024)  # 转换为MB
    return size_mb

def measure_inference_time(model, input_data, device, num_runs=100):
    """
    测量模型推理时间
    """
    model.eval()
    model.to(device)

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(*input_data)

    # 实际测量
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(*input_data)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
    return avg_inference_time

def compare_models(model_configs):
    """
    比较不同模型的参数量、大小和推理时间
    """
    # 使用CPU设备
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # 加载数据
    train_data, test_data = load_and_split_data()

    results = []

    for model_name, model_type in model_configs.items():
        print(f"\n{'=' * 50}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 50}")

        try:
            # 更新配置
            configs.model = model_type

            # 创建实验对象
            exp = Exp_Custom(configs)

            # 获取测试数据加载器
            test_dataset, test_loader = exp._get_data(test_data, mode='test')

            # 构建模型
            model = exp._build_model().to(device)

            # 计算参数量
            params = count_parameters(model)
            print(f"Parameters: {params:,}")

            # 计算模型大小
            size_mb = get_model_size(model)
            print(f"Model Size: {size_mb:.2f} MB")

            # 测量推理时间
            sample_input = next(iter(test_loader))

            if model_type in ['SOFTS', 'SOFTS_Modified', 'CFAD_Teacher', 'CFAD_Student']:
                x_enc_sample = sample_input[0][:1].to(device)
                x_mark_enc_sample = sample_input[2][:1].to(device)
                x_dec_sample = x_enc_sample[:, -configs.pred_len:, :].to(device)
                x_mark_dec_sample = x_mark_enc_sample[:, -configs.pred_len:, :].to(device)
                input_data = (x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample)
            elif model_type in ['TCN', 'LSTM', 'GRU', 'MLP', 'Transformer', 'Informer', 'Autoformer', 'LightTS']:
                x_enc_sample = sample_input[0][:1].to(device)
                x_mark_enc_sample = sample_input[2][:1].to(device)
                input_data = (x_enc_sample, x_mark_enc_sample)
            else:
                x_enc_sample = sample_input[0][:1].to(device)
                input_data = (x_enc_sample,)

            inference_time = measure_inference_time(model, input_data, device)
            print(f"Inference Time: {inference_time:.2f} ms")

            # 保存结果
            results.append({
                'Model': model_name,
                'Parameters': params,
                'Size (MB)': size_mb,
                'Inference Time (ms)': inference_time
            })

        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            results.append({
                'Model': model_name,
                'Parameters': 0,
                'Size (MB)': 0,
                'Inference Time (ms)': 0,
                'Error': str(e)
            })

    return results

def main():
    set_seed(2021)

    # 定义要测试的模型
    model_configs = {
        'SOFTS': 'SOFTS',
        'SOFTS_Modified': 'SOFTS_Modified',
        # 'CFAD_Teacher': 'CFAD_Teacher',
        # 'CFAD_Student': 'CFAD_Student',
        'TCN': 'TCN',
        'LSTM': 'LSTM',
        'GRU': 'GRU',
        'MLP': 'MLP',
        'Transformer': 'Transformer',
        'Informer': 'Informer',
        'Autoformer': 'Autoformer',
        'LightTS': 'LightTS'
    }

    # 进行模型比较
    results = compare_models(model_configs)

    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to model_comparison_results.csv")

    # 打印结果
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
