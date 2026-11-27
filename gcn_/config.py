import torch

# -------------------------- 配置参数（严格复用 paper1.pdf 实验逻辑）--------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 12  # 输入时序长度  5分钟间隔 60分钟
PRED_LEN = 3  # 预测长度3=未来15分钟   6=未来30分钟   12=未来60分钟
BATCH_SIZE = 64  # 批次大小，参考 paper1.pdf 训练参数
EPOCHS = 5  # 训练轮次，与 paper1.pdf 模型训练轮次一致
LR = 1e-3  # 学习率（Adam优化器），与 paper1.pdf 第1-124节一致
FEATURE_DIM = 8  # 输入特征维度（单链4维+跨链4维，见下文特征工程）
TARGET_DIM = 2  # 预测目标维度（latency_cross、throughput_cross）
