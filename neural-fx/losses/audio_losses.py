import torch


def pre_emphasis_filter(x, coeff=0.95):
    return torch.concat([x, x - coeff * x], 1)


def ESR(y_pred, y_true):
    """
    Error to signal ratio with pre-emphasis filter:
    """
    y_true, y_pred = pre_emphasis_filter(y_true), pre_emphasis_filter(y_pred)
    return torch.sum(torch.pow(y_true - y_pred, 2)) / torch.sum(torch.pow(y_true, 2)) + 1e-10


def MSE(y_pred, y_true):
    return torch.nn.functional.mse_loss(y_pred, y_true)
