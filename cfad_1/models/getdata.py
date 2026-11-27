# 假设另一个文件为 get_data_example.py

from exp.exp_custom import Exp_Custom
import pandas as pd
from utils.Config import configs  # 假设配置文件在 utils/Config.py 中

configs.use_gpu = False
# 假设已经有一个配置对象 args
args = configs

# 初始化 Exp_Custom 实例
exp = Exp_Custom(args)

# 假设有一个数据集 data
data = pd.read_csv('../dataset/rate_dataset/ReserveSendRates_5minutes.csv')

# 获取训练数据
train_dataset, train_loader = exp._get_data(data, mode='train')

# 获取测试数据
test_dataset, test_loader = exp._get_data(data, mode='test')

# 获取预测数据
pred_dataset, pred_loader = exp._get_data(data, mode='pred', stride=1)

print(train_dataset)
print(test_dataset)
print(pred_dataset)

# 示例：遍历训练数据加载器
# for batch_x, batch_y, batch_x_mark in train_loader:
#     print(batch_x.shape, batch_y.shape, batch_x_mark.shape)
