import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from cfad_.exp.exp_basic import Exp_Basic
from cfad_.utils.timefeatures import time_features
from cfad_.utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
from cfad_.utils.Config import configs

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, data, seq_len, pred_len, freq='h', mode='pred', stride=1):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.freq = freq
        self.mode = mode
        self.stride = stride
        self.__read_data__()

    def __read_data__(self):
        """
        self.dataset.columns: ['date', ...(other features)]
        """
        if 'date' in self.data.columns:
            cols_data = self.data.columns[1:]
            self.data_x = self.data[cols_data].values

            df_stamp = self.data[['date']]
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            self.data_stamp = data_stamp.transpose(1, 0)
        else:
            self.data_x = self.data.values
            self.data_stamp = np.zeros((self.data_x.shape[0], 1))

    def __getitem__(self, index):
        if self.mode != 'pred':
            s_begin = index * self.stride
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_x[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]

            return seq_x, seq_y, seq_x_mark
        else:
            s_begin = index * self.stride
            s_end = s_begin + self.seq_len
            seq_x = self.data_x[s_begin:s_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            return seq_x, seq_x_mark

    def __len__(self):
        if self.mode != 'pred':
            return (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.stride
        else:
            return (len(self.data_x) - self.seq_len + 1) // self.stride


class Exp_Custom(Exp_Basic):
    def __init__(self, args):
        self.model_class = self.get_model_class(args.model)
        super(Exp_Custom, self).__init__(args)

    def get_model_class(self, model_type):
        if model_type == 'SOFTS':
            from cfad_.models import Model as model_class
        elif model_type == 'TCN':
            from cfad_.models import TemporalConvNet as model_class
        elif model_type == 'LSTM':
            from cfad_.models import LSTM as model_class
        elif model_type == 'GRU':
            from cfad_.models import GRU as model_class
        elif model_type == 'MLP':
            from cfad_.models import MLP as model_class
        elif model_type == 'Transformer':
            from cfad_.models import Transformer as model_class
        elif model_type == 'Informer':
            from cfad_.models import Informer as model_class
        elif model_type == 'Autoformer':
            from cfad_.models import Autoformer as model_class
        elif model_type == 'LightTS':
            from cfad_.models import LightTS as model_class
        elif model_type == 'SOFTS_Modified':
            from cfad_.models import Model as model_class
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_class

    def _build_model(self):
        if self.args.model == 'SOFTS':
            model = self.model_class(self.args)
        elif self.args.model == 'TCN':
            # num_channels 是配置参数的一部分
            num_channels = self.args.num_channels
            model = self.model_class(num_inputs=self.args.num_inputs, num_channels=num_channels,
                                     kernel_size=self.args.kernel_size, dropout=self.args.dropout).float()
        elif self.args.model == 'LSTM':
            model = self.model_class(device=self.device, input_size=self.args.num_inputs, hidden_size=self.args.lstm_hidden_size, num_layers=self.args.lstm_num_layers,
                                     dropout=self.args.dropout, bidirectional=False).float()
        elif self.args.model == 'GRU':
            model = self.model_class(device=self.device, input_size=self.args.num_inputs, hidden_size=self.args.gru_hidden_size, num_layers=self.args.gru_num_layers,
                                     ).float()
        elif self.args.model == 'MLP':
            model = self.model_class(device=self.device, input_size=self.args.num_inputs, hidden_sizes=self.args.mlp_hidden_size, output_size=self.args.mlp_output_size, num_layers=self.args.mlp_num_layers)
        elif self.args.model == 'Transformer':
            model = self.model_class(device=self.device, input_size=self.args.num_inputs, hidden_size=self.args.d_model, num_layers=self.args.e_layers, dropout=0.2)
        elif self.args.model == 'Informer':
            model = self.model_class(device=self.device, input_size=self.args.num_inputs, hidden_size=self.args.d_model, num_layers=self.args.e_layers, dropout=0.2)
        elif self.args.model == 'Autoformer':
            model = self.model_class(device=self.device, input_size=self.args.num_inputs, hidden_size=self.args.d_model, num_layers=self.args.e_layers, dropout=0.2)
        elif self.args.model == 'LightTS':
            model = self.model_class(device=self.device, input_size=self.args.num_inputs, hidden_size=self.args.d_model, num_layers=self.args.e_layers, dropout=0.2)
        elif self.args.model == 'SOFTS_Modified':
            model = self.model_class(self.args)
        else:
            raise ValueError(f"Unsupported model type: {self.args.model}")
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, data, mode, stride=1):
        if mode == 'train':
            dataset = Dataset_Custom(data, self.args.seq_len, self.args.pred_len, freq=self.args.freq, mode='train',
                                     stride=1)
            shuffle = True
        elif mode == 'test':
            dataset = Dataset_Custom(data, self.args.seq_len, self.args.pred_len, freq=self.args.freq, mode='test',
                                     stride=1)
            shuffle = False
        elif mode == 'pred':
            dataset = Dataset_Custom(data, self.args.seq_len, self.args.pred_len, freq=self.args.freq, mode='pred',
                                     stride=stride)
            shuffle = False
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers)
        return dataset, dataloader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        scaler = configs.scaler
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                if configs.model == 'SOFTS':
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                elif configs.model == 'TCN':
                    outputs = self.model(batch_x, batch_x_mark)
                elif configs.model == 'LSTM':
                    outputs = self.model(batch_x)
                elif configs.model == 'GRU':
                    outputs = self.model(batch_x)
                elif configs.model == 'MLP':
                    outputs = self.model(batch_x)
                elif configs.model == 'Transformer':
                    outputs = self.model(batch_x)
                elif configs.model == 'Informer':
                    outputs = self.model(batch_x, batch_x)
                elif configs.model == 'Autoformer':
                    outputs = self.model(batch_x, batch_x)
                elif configs.model == 'LightTS':
                    outputs = self.model(batch_x)
                elif configs.model == 'SOFTS_Modified':
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                # outputs = self.model(batch_x, batch_x_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))
        total_loss = total_loss.avg
        self.model.train()
        return total_loss

    def train(self, setting, train_data, vali_data=None, test_data=None):
        scaler = configs.scaler
        train_data, train_loader = self._get_data(train_data, mode='train')
        if vali_data is not None:
            vali_data, vali_loader = self._get_data(vali_data, mode='test')
        if test_data is not None:
            test_data, test_loader = self._get_data(test_data, mode='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                if configs.model == 'SOFTS':
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                elif configs.model == 'TCN':
                    outputs = self.model(batch_x, batch_x_mark)
                elif configs.model == 'LSTM':
                    outputs = self.model(batch_x)
                elif configs.model == 'GRU':
                    outputs = self.model(batch_x)
                elif configs.model == 'MLP':
                    outputs = self.model(batch_x)
                elif configs.model == 'Transformer':
                    outputs = self.model(batch_x)
                elif configs.model == 'Informer':
                    outputs = self.model(batch_x, batch_x)
                elif configs.model == 'Autoformer':
                    outputs = self.model(batch_x, batch_x)
                elif configs.model == 'LightTS':
                    outputs = self.model(batch_x)
                elif configs.model == 'SOFTS_Modified':
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                # outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = None
            test_loss = None
            if vali_data is not None:
                vali_loss = self.vali(vali_loader, criterion)
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if test_data is not None:
                test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
        return self.model


    def test(self, setting, test_data, stride=1):

        scaler = configs.scaler

        test_data, test_loader = self._get_data(test_data, mode='test', stride=stride)
        model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        print(f'loading model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))
        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        mse = AverageMeter()
        mae = AverageMeter()
        self.model.eval()

        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                if configs.model == 'SOFTS':
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                elif configs.model == 'TCN':
                    outputs = self.model(batch_x, batch_x_mark)
                elif configs.model == 'LSTM':
                    outputs = self.model(batch_x)
                elif configs.model == 'GRU':
                    outputs = self.model(batch_x)
                elif configs.model == 'MLP':
                    outputs = self.model(batch_x)
                elif configs.model == 'Transformer':
                    outputs = self.model(batch_x)
                elif configs.model == 'Informer':
                    outputs = self.model(batch_x, batch_x)
                elif configs.model == 'Autoformer':
                    outputs = self.model(batch_x, batch_x)
                elif configs.model == 'LightTS':
                    outputs = self.model(batch_x)
                elif configs.model == 'SOFTS_Modified':
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                # outputs = self.model(batch_x, batch_x_mark)


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                ##################################################
                # 将数据恢复到原始尺度
                # outputs = outputs.cpu().detach().numpy().squeeze(1)
                # batch_y = batch_y.cpu().detach().numpy().squeeze(1)
                # outputs = scaler.inverse_transform(outputs)
                # batch_y = scaler.inverse_transform(batch_y)
                # outputs = torch.tensor(outputs).unsqueeze(1)
                # batch_y = torch.tensor(batch_y).unsqueeze(1)
                ##################################################

                mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))

                y_true_list.append(batch_y.cpu().numpy())
                y_pred_list.append(outputs.cpu().numpy())

        mse = mse.avg
        mae = mae.avg


        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        rmse = np.sqrt(mse)

        y_true_non_zero = y_true[y_true != 0]
        y_pred_non_zero = y_pred[y_true != 0]
        mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100

        print('mse:{}, mae:{}, rmse:{}, mape:{}%'.format(mse, mae, rmse, mape))

        # 将 y_true 和 y_pred 保存到 CSV 文件
        results_df = pd.DataFrame({
            'True_Value': y_true.flatten(),
            'Predicted_Value': y_pred.flatten()
        })
        # results_df.to_csv('LightTS_pred_true.csv', index=False)
        # print("预测值和真实值已保存到 LightTS_pred_true.csv")

        return


    def predict(self, setting, pred_data, stride=1):
        pred_data, pred_loader = self._get_data(pred_data, mode='pred', stride=stride)
        model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        print(f'loading model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                if configs.model == 'SOFTS':
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                elif configs.model == 'TCN':
                    outputs = self.model(batch_x, batch_x_mark)
                # outputs = self.model(batch_x, batch_x_mark, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                preds.append(outputs.cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        return pred