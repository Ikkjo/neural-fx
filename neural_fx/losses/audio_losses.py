import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def pre_emphasis_filter(
    x: Tensor,
    coeff: float = 0.95,
    *,
    mode: str = "legacy",
) -> Tensor:
    """
    Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]

    Args:
        x: Input tensor of shape [batch, channels, time]
        coeff: Pre-emphasis coefficient (default 0.95)

    Returns:
        Filtered tensor. NAM mode drops the first sample.
    """
    if mode == "nam":
        return x[..., 1:] - coeff * x[..., :-1]
    if mode != "legacy":
        raise ValueError(f"Unknown ESR mode: {mode}")
    return torch.cat([x[..., :1], x[..., 1:] - coeff * x[..., :-1]], dim=-1)


def ESR(
    y_pred: Tensor,
    y_true: Tensor,
    pre_emphasis_coeff: float | None = 0.95,
    *,
    mode: str = "legacy",
) -> Tensor:
    """
    Error to signal ratio with optional pre-emphasis.

    ``legacy`` computes one ratio over the whole tensor. ``nam`` computes a
    ratio per mono batch item, then averages those ratios.
    """
    if mode not in {"legacy", "nam"}:
        raise ValueError(f"Unknown ESR mode: {mode}")
    if pre_emphasis_coeff is None:
        y_true_filtered = y_true
        y_pred_filtered = y_pred
    else:
        y_true_filtered = pre_emphasis_filter(y_true, pre_emphasis_coeff, mode=mode)
        y_pred_filtered = pre_emphasis_filter(y_pred, pre_emphasis_coeff, mode=mode)
    if mode == "nam":
        if y_pred_filtered.ndim == 3 and y_pred_filtered.shape[1] == 1:
            y_pred_filtered = y_pred_filtered.squeeze(1)
            y_true_filtered = y_true_filtered.squeeze(1)
        if y_pred_filtered.ndim == 1 and y_true_filtered.ndim == 1:
            y_pred_filtered = y_pred_filtered.unsqueeze(0)
            y_true_filtered = y_true_filtered.unsqueeze(0)
        if y_pred_filtered.ndim != 2 or y_true_filtered.ndim != 2:
            raise ValueError(
                "NAM ESR expects mono tensors shaped (time,), (batch, time), "
                "or (batch, 1, time)"
            )
        return torch.mean(
            torch.mean(torch.square(y_pred_filtered - y_true_filtered), dim=1)
            / torch.mean(torch.square(y_true_filtered), dim=1)
        )
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
    mode: str = "legacy",
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

    center = True
    if mode == "nam":
        padding = fft_size // 2
        y_pred = torch.cat(
            (
                y_pred[..., 1 : padding + 1].flip(-1),
                y_pred,
                y_pred[..., -padding - 1 : -1].flip(-1),
            ),
            dim=-1,
        )
        y_true = torch.cat(
            (
                y_true[..., 1 : padding + 1].flip(-1),
                y_true,
                y_true[..., -padding - 1 : -1].flip(-1),
            ),
            dim=-1,
        )
        center = False

    # Compute STFT
    y_pred_stft = torch.stft(
        y_pred,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        return_complex=True,
    )
    y_true_stft = torch.stft(
        y_true,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        return_complex=True,
    )

    # Magnitude spectrograms
    if mode == "nam":
        y_pred_mag = torch.sqrt(
            torch.clamp(torch.view_as_real(y_pred_stft).pow(2).sum(-1), min=1e-8)
        )
        y_true_mag = torch.sqrt(
            torch.clamp(torch.view_as_real(y_true_stft).pow(2).sum(-1), min=1e-8)
        )
    elif mode == "legacy":
        y_pred_mag = torch.abs(y_pred_stft)
        y_true_mag = torch.abs(y_true_stft)
    else:
        raise ValueError(f"Unknown STFT loss mode: {mode}")

    # Spectral convergence loss (Eq. 2 in Parallel WaveGAN paper)
    denominator = torch.norm(y_true_mag, p="fro")
    if mode == "legacy":
        denominator = denominator + 1e-8
    sc_loss = torch.norm(y_true_mag - y_pred_mag, p="fro") / denominator

    # Log magnitude loss (Eq. 3 in Parallel WaveGAN paper)
    log = torch.log if mode == "nam" else safe_log
    log_mag_loss = F.l1_loss(log(y_pred_mag), log(y_true_mag))

    return sc_loss, log_mag_loss


class SingleResolutionSTFTLoss(nn.Module):
    """Single resolution STFT loss module."""

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
        mode: str = "legacy",
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.mode = mode

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
        self.window = self.window.to(y_pred.device)
        return stft_loss(
            y_pred,
            y_true,
            self.fft_size,
            self.hop_size,
            self.win_size,
            self.window,
            self.mode,
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
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_sizes: list[int] | None = None,
        sc_loss_weight: float = 1.0,
        mag_loss_weight: float = 1.0,
        mode: str = "legacy",
    ):
        super().__init__()
        if mode == "nam":
            fft_sizes = fft_sizes or [1024, 2048, 512]
            hop_sizes = hop_sizes or [120, 240, 50]
            win_sizes = win_sizes or [600, 1200, 240]
        else:
            fft_sizes = fft_sizes or [512, 1024, 2048]
            hop_sizes = hop_sizes or [fft // 4 for fft in fft_sizes]
            win_sizes = win_sizes or fft_sizes

        if not (len(fft_sizes) == len(hop_sizes) == len(win_sizes)):
            raise ValueError(
                "fft_sizes, hop_sizes, and win_sizes must have same length"
            )

        self.sc_loss_weight = sc_loss_weight
        self.mag_loss_weight = mag_loss_weight

        self.stft_losses = nn.ModuleList(
            [
                SingleResolutionSTFTLoss(fft, hop, win, mode)
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
