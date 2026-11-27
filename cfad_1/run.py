import argparse
import os
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

from exp.exp_custom import Exp_Custom
from utils.Config import configs
from dataset.data_processor import load_and_split_data

# fix seed for reproducibility
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.set_num_threads(6)

configs.use_gpu = True if torch.cuda.is_available() and configs.use_gpu else False

def main(model_type='SOFTS'):
    train_data, test_data = load_and_split_data()

    print(train_data.head())
    print(test_data.head())

    # 更新配置以反映所选的模型类型
    configs.model = model_type

    Exp = Exp_Custom(configs)
    setting = f'{configs.data}_{configs.model}_{configs.seq_len}_{configs.pred_len}'
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    Exp.train(setting=setting, train_data=train_data, vali_data=test_data, test_data=test_data)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    Exp.test(setting=setting, test_data=test_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test models')
    # 修改训练的模型，修改default即可
    parser.add_argument('--model', type=str, default='SOFTS_Modified', choices=['CFAD_Teacher', 'CFAD_Student', 'SOFTS', 'SOFTS_Modified', 'TCN', 'LSTM', 'GRU', 'MLP', 'Transformer', 'Informer', 'Autoformer', 'LightTS'], help='Model type to train')
    args = parser.parse_args()
    main(model_type=args.model)

