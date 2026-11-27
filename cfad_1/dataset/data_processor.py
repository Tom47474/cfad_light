import os
import pickle

import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
import numpy as np

from utils.Config import configs


def load_and_split_data(data_path=None):

    if data_path is None:
        data_path = os.path.join(configs.root_path, configs.data_path)

    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # 加载数据
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])

    # 分离日期列和其他特征列
    date_column = 'date'
    feature_columns = data.columns.difference([date_column])

    # 计算训练集和测试集的索引范围
    total_length = len(data)
    train_end_index = int(total_length * 0.8)

    # 分割数据集
    train_data = data.iloc[:train_end_index]
    test_data = data.iloc[train_end_index:]



    print(train_data.head())
    print(test_data.head())

    # 确保每个集合都包含足够的行
    if len(train_data) != train_end_index or len(test_data) != (total_length - train_end_index):
        raise ValueError("Dataset split did not produce the expected number of rows.")

    # 标准化数据（排除日期列）
    # scaler = StandardScaler()
    # scaler.fit(train_data[feature_columns])
    #
    # train_data[feature_columns] = scaler.transform(train_data[feature_columns])
    # test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    # lmbda = 0.001
    # train_data[feature_columns] = boxcox(train_data[feature_columns], lmbda=lmbda)
    # test_data[feature_columns] = boxcox(test_data[feature_columns], lmbda=lmbda)
########################################################
    scaler = MinMaxScaler()
    scaler.fit(train_data[feature_columns])

    train_data[feature_columns] = scaler.transform(train_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    print(train_data.head())
    print(test_data.head())

    configs.scaler = scaler

    return train_data, test_data