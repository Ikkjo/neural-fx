from typing import Callable

import torch
from torch import Tensor


class RandomGain:
    """
    Apply random gain to audio within a specified range.

    This augmentation randomly scales the audio amplitude, simulating
    different input levels that the model might encounter.

    Args:
        min_gain_db: Minimum gain in decibels.
        max_gain_db: Maximum gain in decibels.
        p: Probability of applying the gain adjustment.
    """

    def __init__(
        self, min_gain_db: float = -12.0, max_gain_db: float = 0.0, p: float = 0.5
    ):
        if min_gain_db > max_gain_db:
            raise ValueError("min_gain_db must be <= max_gain_db")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")

        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.p = p

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply random gain to input and target.

        Args:
            x: Input audio tensor of shape [time].
            y: Target audio tensor of shape [time].

        Returns:
            Tuple of (augmented_input, augmented_target).
        """
        if torch.rand(1).item() >= self.p:
            return x, y

        # Random gain in dB
        gain_db = torch.empty(1).uniform_(self.min_gain_db, self.max_gain_db).item()
        gain_linear = 10 ** (gain_db / 20.0)

        # Apply gain to both input and target
        x = x * gain_linear
        y = y * gain_linear

        return x, y


class AddNoise:
    """
    Add Gaussian noise to audio.

    This augmentation adds noise to simulate real-world recording conditions
    and improve model robustness.

    Args:
        min_snr_db: Minimum signal-to-noise ratio in decibels.
        max_snr_db: Maximum signal-to-noise ratio in decibels.
        p: Probability of applying noise.
    """

    def __init__(
        self, min_snr_db: float = 20.0, max_snr_db: float = 40.0, p: float = 0.3
    ):
        if min_snr_db > max_snr_db:
            raise ValueError("min_snr_db must be <= max_snr_db")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")

        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """
        Add Gaussian noise to input.

        Args:
            x: Input audio tensor of shape [time].
            y: Target audio tensor of shape [time].

        Returns:
            Tuple of (noisy_input, target). Target is unchanged.
        """
        if torch.rand(1).item() >= self.p:
            return x, y

        # Random SNR in dB
        snr_db = torch.empty(1).uniform_(self.min_snr_db, self.max_snr_db).item()
        snr_linear = 10 ** (snr_db / 10.0)

        # Calculate noise power based on signal power
        signal_power = x.pow(2).mean()
        noise_power = signal_power / snr_linear

        # Generate Gaussian noise
        noise = torch.randn_like(x) * torch.sqrt(noise_power)

        # Add noise to input only (target should remain clean)
        x = x + noise

        return x, y


class Compose:
    """Compose multiple transforms to apply sequentially."""

    def __init__(
        self, transforms: list[Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]]
    ):
        self.transforms = transforms

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        for transform in self.transforms:
            x, y = transform(x, y)
        return x, y


def build_augmentation_transform(
    random_gain: bool = False,
    gain_min_db: float = -12.0,
    gain_max_db: float = 0.0,
    gain_p: float = 0.5,
    add_noise: bool = False,
    noise_min_snr_db: float = 20.0,
    noise_max_snr_db: float = 40.0,
    noise_p: float = 0.3,
) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None:
    """
    Build augmentation transform from config options.

    Args:
        random_gain: Whether to enable random gain augmentation.
        gain_min_db: Minimum gain in dB.
        gain_max_db: Maximum gain in dB.
        gain_p: Probability of applying gain.
        add_noise: Whether to enable noise augmentation.
        noise_min_snr_db: Minimum SNR in dB.
        noise_max_snr_db: Maximum SNR in dB.
        noise_p: Probability of adding noise.

    Returns:
        Composed transform or None if no augmentations enabled.
    """
    transforms = []

    if random_gain:
        transforms.append(RandomGain(gain_min_db, gain_max_db, gain_p))

    if add_noise:
        transforms.append(AddNoise(noise_min_snr_db, noise_max_snr_db, noise_p))

    if not transforms:
        return None

    return Compose(transforms)
