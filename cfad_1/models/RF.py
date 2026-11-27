import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归器
from torch.utils.data import DataLoader

from dataset.data_processor import load_and_split_data
from exp.exp_custom import Dataset_Custom
from utils.Config import configs

warnings.filterwarnings('ignore')

# 加载和分割数据
train_data, test_data = load_and_split_data()

# 创建数据集和数据加载器
train_dataset = Dataset_Custom(
    train_data,
    configs.seq_len,
    configs.pred_len,
    configs.freq,
    mode='train',
    stride=1
)
test_dataset = Dataset_Custom(
    test_data,
    configs.seq_len,
    configs.pred_len,
    configs.freq,
    mode='test',
    stride=1
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=configs.batch_size,
    shuffle=True,
    num_workers=configs.num_workers
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=configs.batch_size,
    shuffle=False,
    num_workers=configs.num_workers
)

# 从数据加载器中提取数据
def extract_data_from_dataloader(dataloader):
    features = []
    labels = []
    for batch in dataloader:
        # 假设 batch 是一个包含 (features, labels, ...) 的元组
        batch_features, batch_labels, _ = batch
        features.append(batch_features.numpy())  # 转换为 NumPy 数组
        labels.append(batch_labels.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

train_features, train_labels = extract_data_from_dataloader(train_dataloader)
test_features, test_labels = extract_data_from_dataloader(test_dataloader)

# 重塑特征以匹配模型的输入要求（如果必要）
# 这里假设每个样本的特征是平铺的，即没有额外的维度需要重塑
# 如果特征是多维的（例如图像），则需要相应地重塑

# 训练随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)  # 使用 100 棵树和固定的随机种子
train_features = train_features.reshape(train_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)
# 注意：随机森林回归直接处理多维标签，但我们需要确保它们匹配
# 如果标签是多维的，请确保它们在 reshape 操作后仍然保持正确的形状
# 由于我们未知标签的原始形状，这里假设标签是一维的
# 如果标签是多维的，需要根据实际情况调整
train_labels = train_labels.reshape(-1)  # 确保标签是一维的
test_labels = test_labels.reshape(-1)
rf.fit(train_features, train_labels)  # 训练随机森林回归模型

# 进行预测
train_predictions = rf.predict(train_features)  # 随机森林回归器的预测
test_predictions = rf.predict(test_features)

# 计算误差指标
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values').mean()  # 计算 MSE，取平均值
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values').mean()  # 计算 MAE，取平均值
    rmse = np.sqrt(mse)  # 计算 RMSE

    # 计算 MAPE，仅考虑非零的真实值
    y_true_non_zero = y_true[y_true != 0]
    y_pred_non_zero = y_pred[y_true_non_zero.argsort()]  # 使用与 y_true_non_zero 相同的索引排序 y_pred
    mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100

    return mse, mae, rmse, mape

train_mse, train_mae, train_rmse, train_mape = calculate_metrics(train_labels, train_predictions)
test_mse, test_mae, test_rmse, test_mape = calculate_metrics(test_labels, test_predictions)

# 打印结果
print(f'Training MSE: {train_mse}')
print(f'Training MAE: {train_mae}')
print(f'Training RMSE: {train_rmse}')
print(f'Training MAPE: {train_mape}%')
print(f'Test MSE: {test_mse}')
print(f'Test MAE: {test_mae}')
print(f'Test RMSE: {test_rmse}')
print(f'Test MAPE: {test_mape}%')