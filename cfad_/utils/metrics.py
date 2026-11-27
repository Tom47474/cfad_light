# utils/metrics.py
import numpy as np
import torch


def MAE(pred, true):
    """
    Mean Absolute Error
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Mean Squared Error
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Root Mean Squared Error
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Mean Absolute Percentage Error
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    # 避免除以零的情况
    mask = true != 0
    if np.sum(mask) == 0:
        return np.inf

    mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100
    return mape


def MSPE(pred, true):
    """
    Mean Squared Percentage Error
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    # 避免除以零的情况
    mask = true != 0
    if np.sum(mask) == 0:
        return np.inf

    mspe = np.mean(((pred[mask] - true[mask]) / true[mask]) ** 2) * 100
    return mspe


def metric(pred, true):
    """
    Calculate all metrics
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
