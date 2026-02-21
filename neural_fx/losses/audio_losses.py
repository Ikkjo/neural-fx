import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def pre_emphasis_filter(x: Tensor, coeff: float = 0.95) -> Tensor:
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


def ESR(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Error to signal ratio with pre-emphasis filter.
    """
    y_true_filtered = pre_emphasis_filter(y_true)
    y_pred_filtered = pre_emphasis_filter(y_pred)
    return torch.sum(torch.pow(y_true_filtered - y_pred_filtered, 2)) / (
        torch.sum(torch.pow(y_true_filtered, 2)) + 1e-10
    )


def MSE(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return F.mse_loss(y_pred, y_true)


def safe_log(x: Tensor, eps: float = 1e-7) -> Tensor:
    """Safe log to avoid log(0)."""
    return torch.log(x + eps)


def stft_loss(
    y_pred: Tensor,
    y_true: Tensor,
    fft_size: int,
    hop_size: int,
    win_size: int,
    window: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute STFT loss between predicted and target signals.

    Args:
        y_pred: Predicted signal [batch, channels, time]
        y_true: Target signal [batch, channels, time]
        fft_size: FFT size
        hop_size: Hop size
        win_size: Window size
        window: Optional window tensor

    Returns:
        Tuple of (spectral_convergence_loss, log_magnitude_loss)
    """
    if window is None:
        window = torch.hann_window(win_size, device=y_pred.device)

    # Handle mono audio
    if y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)
        y_true = y_true.squeeze(1)
    else:
        # Multi-channel: compute mean across channels
        y_pred = y_pred.mean(dim=1)
        y_true = y_true.mean(dim=1)

    # Compute STFT
    y_pred_stft = torch.stft(
        y_pred,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        return_complex=True,
    )
    y_true_stft = torch.stft(
        y_true,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        return_complex=True,
    )

    # Magnitude spectrograms
    y_pred_mag = torch.abs(y_pred_stft)
    y_true_mag = torch.abs(y_true_stft)

    # Spectral convergence loss (Eq. 2 in Parallel WaveGAN paper)
    sc_loss = torch.norm(y_true_mag - y_pred_mag, p="fro") / (
        torch.norm(y_true_mag, p="fro") + 1e-8
    )

    # Log magnitude loss (Eq. 3 in Parallel WaveGAN paper)
    log_mag_loss = F.l1_loss(safe_log(y_pred_mag), safe_log(y_true_mag))

    return sc_loss, log_mag_loss


class SingleResolutionSTFTLoss(nn.Module):
    """Single resolution STFT loss module."""

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size

        self.register_buffer("window", torch.hann_window(win_size))

    def forward(self, y_pred: Tensor, y_true: Tensor) -> tuple[Tensor, Tensor]:
        """
        Calculate forward propagation.

        Args:
            y_pred: Predicted signal [batch, channels, time]
            y_true: Target signal [batch, channels, time]

        Returns:
            Tuple of (spectral_convergence_loss, log_magnitude_loss)
        """
        return stft_loss(
            y_pred,
            y_true,
            self.fft_size,
            self.hop_size,
            self.win_size,
            self.window,
        )


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss module.

    This loss combines STFT losses at multiple resolutions (FFT sizes)
    to capture both fine and coarse temporal structure.

    Reference:
        "Parallel WaveGAN: A Fast Waveform Generation Model Based on Generative Adversarial
        Networks with Multi-resolution Spectrogram" (Yamamoto et al., 2020)

    Args:
        fft_sizes: List of FFT sizes for different resolutions.
        hop_sizes: List of hop sizes corresponding to fft_sizes.
        win_sizes: List of window sizes corresponding to fft_sizes.
        sc_loss_weight: Weight for spectral convergence loss.
        mag_loss_weight: Weight for log magnitude loss.
    """

    def __init__(
        self,
        fft_sizes: list[int] = [512, 1024, 2048],
        hop_sizes: list[int] | None = None,
        win_sizes: list[int] | None = None,
        sc_loss_weight: float = 1.0,
        mag_loss_weight: float = 1.0,
    ):
        super().__init__()
        if hop_sizes is None:
            hop_sizes = [fft // 4 for fft in fft_sizes]
        if win_sizes is None:
            win_sizes = fft_sizes

        if not (len(fft_sizes) == len(hop_sizes) == len(win_sizes)):
            raise ValueError(
                "fft_sizes, hop_sizes, and win_sizes must have same length"
            )

        self.sc_loss_weight = sc_loss_weight
        self.mag_loss_weight = mag_loss_weight

        self.stft_losses = nn.ModuleList(
            [
                SingleResolutionSTFTLoss(fft, hop, win)
                for fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
            ]
        )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate multi-resolution STFT loss.

        Args:
            y_pred: Predicted signal [batch, channels, time]
            y_true: Target signal [batch, channels, time]

        Returns:
            Combined loss value.
        """
        total_loss = torch.tensor(0.0, device=y_pred.device)

        for stft_loss_fn in self.stft_losses:
            sc_loss, mag_loss = stft_loss_fn(y_pred, y_true)
            total_loss = (
                total_loss
                + self.sc_loss_weight * sc_loss
                + self.mag_loss_weight * mag_loss
            )

        return total_loss / len(self.stft_losses)
