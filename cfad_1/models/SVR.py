import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from torch.utils.data import DataLoader

from getdata import exp, train_dataset, train_loader, test_dataset, test_loader
from utils.Config import configs

warnings.filterwarnings('ignore')

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

# 提取训练和测试数据
train_features, train_labels = extract_data_from_dataloader(train_loader)
test_features, test_labels = extract_data_from_dataloader(test_loader)

# 重塑特征以匹配 SVR 的输入要求
train_features = train_features.reshape(train_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)
train_labels = train_labels.reshape(train_labels.shape[0], -1)
test_labels = test_labels.reshape(test_labels.shape[0], -1)

# 训练 SVR 模型
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(train_features, train_labels.flatten())  # 重塑以匹配 SVR 的输出要求

# 进行预测
train_predictions = svr.predict(train_features).reshape(train_labels.shape)  # 重塑以匹配原始标签形状
test_predictions = svr.predict(test_features).reshape(test_labels.shape)

# 计算误差指标
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values').mean()  # 计算 MSE，取平均值
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values').mean()  # 计算 MAE，取平均值
    rmse = np.sqrt(mse)  # 计算 RMSE

    # 计算 MAPE，仅考虑非零的真实值
    y_true_non_zero = y_true[y_true != 0]
    y_pred_non_zero = y_pred[y_true != 0]
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

test_labels = test_labels.flatten()
test_predictions = test_predictions.flatten()

# 创建一个 DataFrame 来保存测试集的真实值和预测值
results_df = pd.DataFrame({
    'True_Value': test_labels,  # 使用 ravel() 确保是一维数组
    'Predicted_Value': test_predictions
})

# 保存 DataFrame 到 CSV 文件
results_df.to_csv('SVR_pred_true.csv', index=False)
