import torch


def pre_emphasis_filter(x, coeff=0.95):
    """
    Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]

    Args:
        x: Input tensor of shape [batch, channels, time]
        coeff: Pre-emphasis coefficient (default 0.95)

    Returns:
        Filtered tensor of same shape as input
    """
    # Keep first sample unchanged, apply filter to rest
    return torch.cat([x[..., :1], x[..., 1:] - coeff * x[..., :-1]], dim=-1)


def ESR(y_pred, y_true):
    """
    Error to signal ratio with pre-emphasis filter.
    """
    y_true_filtered = pre_emphasis_filter(y_true)
    y_pred_filtered = pre_emphasis_filter(y_pred)
    return (
        torch.sum(torch.pow(y_true_filtered - y_pred_filtered, 2))
        / torch.sum(torch.pow(y_true_filtered, 2))
        + 1e-10
    )


def MSE(y_pred, y_true):
    return torch.nn.functional.mse_loss(y_pred, y_true)
