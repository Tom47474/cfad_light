# gcn_/main.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from config import *
from gcn_.models.agcrn import AGCRN
from gcn_.models.stsgcn import STSGCN
from gcn_.models.kan import KANModel
from gcn_.models.transformer import TransformerModel
from gcn_.models.gat import GAT
from gcn_.models.autost import AutoST
from gcn_.models.cgal import CGAL_CrossChain  # 添加CGAL模型
from gcn_.utils.data_processor import load_and_preprocess_data
from gcn_.utils.training_utils import train_model, evaluate_model
from gcn_.utils.early_stopping import EarlyStopping
from gcn_.models.lstm_baseline import LSTM_Baseline
from gcn_.models.cmt_net import CMT_Net
from gcn_.models.bpnn import BPNN
from gcn_.models.cnn_lstm import CNN_LSTM


def run_cross_chain_experiment(data_path="HFBDP-Cross.csv"):
    # 1. 加载数据
    train_loader, test_loader, scaler_y, adj, num_total_nodes = load_and_preprocess_data(
        data_path, seq_len=SEQ_LEN, pred_len=PRED_LEN)

    # 2. 初始化模型、损失函数、优化器（均遵循 paper1.pdf 逻辑）
    criterion = nn.MSELoss()
    models = {}

    # CGAL模型
    models["CGAL"] = {
        "model": CGAL_CrossChain(input_dim=FEATURE_DIM, num_nodes=num_total_nodes).to(DEVICE),
        "optimizer": optim.Adam(
            CGAL_CrossChain(input_dim=FEATURE_DIM, num_nodes=num_total_nodes).to(DEVICE).parameters(), lr=LR)
    }

    # AutoST模型
    models["AutoST"] = {
        "model": AutoST(input_dim=FEATURE_DIM, hidden_dim=64, num_nodes=num_total_nodes,
                        output_dim=TARGET_DIM, seq_len=SEQ_LEN).to(DEVICE),
        "optimizer": optim.Adam(
            AutoST(input_dim=FEATURE_DIM, hidden_dim=64, num_nodes=num_total_nodes,
                   output_dim=TARGET_DIM, seq_len=SEQ_LEN).to(DEVICE).parameters(), lr=LR)
    }

    # STSGCN模型
    models["STSGCN"] = {
        "model": STSGCN(input_dim=FEATURE_DIM, hidden_dims=[32, 64], vertex_num=num_total_nodes,
                        output_dim=TARGET_DIM, time_steps=SEQ_LEN).to(DEVICE),
        "optimizer": optim.Adam(
            STSGCN(input_dim=FEATURE_DIM, hidden_dims=[32, 64], vertex_num=num_total_nodes,
                   output_dim=TARGET_DIM, time_steps=SEQ_LEN).to(DEVICE).parameters(), lr=LR)
    }

    models["GAT"] = {
        "model": GAT(input_dim=FEATURE_DIM, hidden_dim=8, nheads=8).to(DEVICE),
        "optimizer": optim.Adam(GAT(input_dim=FEATURE_DIM, hidden_dim=8, nheads=8).to(DEVICE).parameters(), lr=LR)
    }

    # AGCRN模型
    models["AGCRN"] = {
        "model": AGCRN(input_dim=FEATURE_DIM, rnn_units=32, num_nodes=SEQ_LEN, output_dim=TARGET_DIM, seq_len=SEQ_LEN, embed_dim=8).to(DEVICE),
        "optimizer": optim.Adam(
            AGCRN(input_dim=FEATURE_DIM, rnn_units=32, num_nodes=SEQ_LEN, output_dim=TARGET_DIM, seq_len=SEQ_LEN, embed_dim=8).to(
                DEVICE).parameters(), lr=LR)
    }

    # Transformer模型
    models["Transformer"] = {
        "model": TransformerModel(input_dim=FEATURE_DIM).to(DEVICE),
        "optimizer": optim.Adam(TransformerModel(input_dim=FEATURE_DIM).to(DEVICE).parameters(), lr=LR)
    }

    models["KAN"] = {
        "model": KANModel(input_dim=FEATURE_DIM * SEQ_LEN).to(DEVICE),
        "optimizer": optim.Adam(KANModel(input_dim=FEATURE_DIM * SEQ_LEN).to(DEVICE).parameters(), lr=LR)
    }

    # CNN-LSTM模型
    models["CNN-LSTM"] = {
        "model": CNN_LSTM(input_dim=FEATURE_DIM).to(DEVICE),
        "optimizer": optim.Adam(CNN_LSTM(input_dim=FEATURE_DIM).to(DEVICE).parameters(), lr=LR)
    }

    # BPNN模型
    models["BPNN"] = {
        "model": BPNN(input_dim=FEATURE_DIM * SEQ_LEN).to(DEVICE),
        "optimizer": optim.Adam(BPNN(input_dim=FEATURE_DIM * SEQ_LEN).to(DEVICE).parameters(), lr=LR)
    }

    # LSTM基线
    models["LSTM-Baseline"] = {
        "model": LSTM_Baseline(input_dim=FEATURE_DIM).to(DEVICE),
        "optimizer": optim.Adam(LSTM_Baseline(input_dim=FEATURE_DIM).to(DEVICE).parameters(), lr=LR)
    }

    # CMT-Net（Ours）
    models["CMT-Net (Ours)"] = {
        "model": CMT_Net(input_dim=FEATURE_DIM, num_nodes=num_total_nodes).to(DEVICE),
        "optimizer": optim.Adam(CMT_Net(input_dim=FEATURE_DIM, num_nodes=num_total_nodes).to(DEVICE).parameters(),
                                lr=LR)
    }


    # 3. 为数据集添加邻接矩阵（空间/时空模型需要）
    train_loader.dataset.adj = adj
    test_loader.dataset.adj = adj

    # 4. 训练与评估所有模型
    results = {}
    for model_name, info in models.items():
        print(f"\n=== 开始训练：{model_name} ===")
        model = info["model"]
        optimizer = info["optimizer"]

        # 对于需要邻接矩阵的模型，将邻接矩阵设置为模型属性
        if model_name in ["GAT", "AutoST", "CGAL"]:
            model.adj = adj

        # 初始化早停机制
        early_stopping = EarlyStopping(patience=10, verbose=True, path=f'checkpoints/{model_name}_best.pth')

        # 训练
        best_loss = float("inf")
        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):
            train_loss = train_model(model, train_loader, criterion, optimizer, epoch, EPOCHS)
            # 验证
            eval_res = evaluate_model(model, test_loader, scaler_y, criterion)

            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(eval_res["avg_loss"])

            # 打印训练信息
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.6f}, Val Loss: {eval_res['avg_loss']:.6f}")

            # 早停机制检查
            early_stopping(eval_res["avg_loss"], model, f'checkpoints/{model_name}_best.pth')

            # 如果触发早停，则退出训练循环
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            # 保存最佳模型
            if eval_res["avg_loss"] < best_loss:
                best_loss = eval_res["avg_loss"]
                torch.save(model.state_dict(), f"{model_name}_best.pth")

        # 加载最优模型评估
        if os.path.exists(f'checkpoints/{model_name}_best.pth'):
            model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pth'))
            print(f"Loaded best model from checkpoints/{model_name}_best.pth")
        else:
            print("Warning: No best model checkpoint found. Using current model weights.")

        final_res = evaluate_model(model, test_loader, scaler_y, criterion)
        results[model_name] = final_res

        # 打印结果
        print(f"\n=== {model_name} 最终评估结果 ===")
        print(
            f"跨链延迟 - MAE: {final_res['latency']['MAE']:.4f}, RMSE: {final_res['latency']['RMSE']:.4f}, MAPE: {final_res['latency']['MAPE']:.4f}%")
        print(
            f"跨链吞吐量 - MAE: {final_res['throughput']['MAE']:.4f}, RMSE: {final_res['throughput']['RMSE']:.4f}, MAPE: {final_res['throughput']['MAPE']:.4f}%")

    # 5. 整理结果为表格
    print("\n=== 所有模型对比结果汇总 ===")
    result_df = pd.DataFrame()
    for model_name, res in results.items():
        row = {
            "Model": model_name,
            "Latency-MAE": res["latency"]["MAE"],
            "Latency-RMSE": res["latency"]["RMSE"],
            "Latency-MAPE(%)": res["latency"]["MAPE"],
            "Throughput-MAE": res["throughput"]["MAE"],
            "Throughput-RMSE": res["throughput"]["RMSE"],
            "Throughput-MAPE(%)": res["throughput"]["MAPE"]
        }
        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

    # 按跨链延迟MAPE排序
    result_df = result_df.sort_values("Latency-MAPE(%)").reset_index(drop=True)
    print(result_df.round(4))
    result_df.to_csv("Cross_Chain_Experiment_Results.csv", index=False)
    print("\n实验结果已保存至 Cross_Chain_Experiment_Results.csv")


# -------------------------- 执行实验 --------------------------
if __name__ == "__main__":
    # 需先确保 HFBDP-Cross.csv 已生成
    run_cross_chain_experiment(data_path="HFBDP-Cross.csv")
