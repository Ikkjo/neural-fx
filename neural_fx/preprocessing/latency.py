"""Latency compensation module for aligning input and output audio signals."""

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from torch import Tensor


@dataclass
class LatencyCalibration:
    """Result of latency calibration.

    Attributes:
        delay_samples: Number of samples the output is delayed relative to input.
            Positive means output lags input, negative means output leads.
        method: Method used for calibration ('xcorr', 'blip', 'manual').
        correlation_score: Peak correlation coefficient (0-1, higher is better).
    """

    delay_samples: int
    method: str
    correlation_score: float


class LatencyCalibrator:
    """Calibrate and apply latency compensation between input and output audio."""

    def calibrate(
        self,
        input_audio: Tensor,
        output_audio: Tensor,
        method: Literal["xcorr", "blip", "manual"] = "xcorr",
        max_delay: int = 10000,
        manual_delay: int | None = None,
    ) -> LatencyCalibration:
        """Calibrate latency between input and output audio signals.

        Args:
            input_audio: Input audio tensor of shape [channels, samples] or [samples].
            output_audio: Output audio tensor of shape [channels, samples] or [samples].
            method: Calibration method. 'xcorr' uses FFT-based cross-correlation,
                'manual' uses provided manual_delay.
            max_delay: Maximum delay to search for in samples.
            manual_delay: Manual delay override when method='manual'.

        Returns:
            LatencyCalibration object with delay and correlation score.

        Raises:
            ValueError: If manual method selected but no delay provided.
        """
        # Flatten to 1D if needed
        if input_audio.ndim > 1:
            input_audio = input_audio.mean(dim=0)
        if output_audio.ndim > 1:
            output_audio = output_audio.mean(dim=0)

        # Ensure same length for correlation
        min_len = min(input_audio.shape[-1], output_audio.shape[-1])
        input_audio = input_audio[..., :min_len]
        output_audio = output_audio[..., :min_len]

        if method == "manual":
            if manual_delay is None:
                raise ValueError("manual_delay must be provided when method='manual'")
            return LatencyCalibration(
                delay_samples=manual_delay,
                method="manual",
                correlation_score=1.0,
            )

        elif method == "xcorr":
            return self._calibrate_xcorr(input_audio, output_audio, max_delay)

        elif method == "blip":
            return self._calibrate_blip(input_audio, output_audio, max_delay)

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def _calibrate_xcorr(
        self, input_audio: Tensor, output_audio: Tensor, max_delay: int
    ) -> LatencyCalibration:
        """Calibrate using FFT-based cross-correlation.

        Uses FFT-based cross-correlation for O(n log n) complexity instead
        of O(n^2) for time-domain correlation.
        """
        # Move to CPU and convert to numpy for processing
        x = input_audio.cpu().numpy()
        y = output_audio.cpu().numpy()

        # Ensure float64 for better precision in correlation
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        # Normalize for better correlation results
        x = (x - np.mean(x)) / (np.std(x) + 1e-10)
        y = (y - np.mean(y)) / (np.std(y) + 1e-10)

        # Compute cross-correlation using FFT
        # FFT cross-correlation: IFFT(FFT(x) * conj(FFT(y)))
        n = len(x) + len(y) - 1
        n_fft = 2 ** int(np.ceil(np.log2(n)))  # Next power of 2

        X = np.fft.fft(x, n_fft)
        Y = np.fft.fft(y, n_fft)

        # Cross-correlation: ifft(X * conj(Y))
        corr = np.fft.ifft(X * np.conj(Y)).real

        # Shift to center at zero lag
        corr = np.fft.fftshift(corr)

        # Find the peak within the allowed delay range
        # The center corresponds to zero lag
        center = len(corr) // 2
        search_start = max(0, center - max_delay)
        search_end = min(len(corr), center + max_delay + 1)

        search_corr = corr[search_start:search_end]
        peak_idx = np.argmax(np.abs(search_corr))
        peak_value = search_corr[peak_idx]

        # Convert back to delay in samples
        # Positive delay means output lags input (output needs to be shifted back)
        # Negative delay means output leads input
        delay = (center - search_start) - peak_idx

        # Normalize correlation score
        # Theoretical max is the overlap length at the optimal shift
        overlap_len = len(x) - abs(delay)
        if overlap_len > 0:
            correlation_score = abs(peak_value) / overlap_len
            correlation_score = min(correlation_score, 1.0)  # Cap at 1.0
        else:
            correlation_score = 0.0

        return LatencyCalibration(
            delay_samples=int(delay),
            method="xcorr",
            correlation_score=float(correlation_score),
        )

    def _calibrate_blip(
        self, input_audio: Tensor, output_audio: Tensor, max_delay: int
    ) -> LatencyCalibration:
        """Calibrate using blip detection (impulse detection).

        This method looks for sharp impulses in the input signal and finds
        corresponding impulses in the output.
        """
        # For now, fall back to xcorr
        # Blip detection requires specific input signals
        return self._calibrate_xcorr(input_audio, output_audio, max_delay)

    def apply_delay(
        self, x: Tensor, y: Tensor, delay: int
    ) -> Tuple[Tensor, Tensor]:
        """Apply delay compensation to align input and output.

        Args:
            x: Input audio tensor [channels, samples] or [samples].
            y: Output audio tensor [channels, samples] or [samples].
            delay: Delay in samples (positive = output lags input).

        Returns:
            Tuple of (compensated_input, compensated_output) tensors.
            Both will have the same length.
        """
        # Handle multi-dimensional inputs
        x_was_1d = x.ndim == 1
        y_was_1d = y.ndim == 1

        if x_was_1d:
            x = x.unsqueeze(0)
        if y_was_1d:
            y = y.unsqueeze(0)

        # Positive delay: output lags input, so we need to:
        # - Delay input by cutting off the start (or padding end)
        # - Advance output by cutting off the start
        if delay > 0:
            # Output lags input by 'delay' samples
            # Remove first 'delay' samples from output
            # Remove last 'delay' samples from input to match length
            y_aligned = y[..., delay:]
            x_aligned = x[..., : y_aligned.shape[-1]]
        elif delay < 0:
            # Output leads input by |delay| samples
            # Remove first |delay| samples from input
            # Remove last |delay| samples from output to match length
            x_aligned = x[..., -delay:]
            y_aligned = y[..., : x_aligned.shape[-1]]
        else:
            # No delay, just match lengths
            min_len = min(x.shape[-1], y.shape[-1])
            x_aligned = x[..., :min_len]
            y_aligned = y[..., :min_len]

        # Restore dimensions
        if x_was_1d:
            x_aligned = x_aligned.squeeze(0)
        if y_was_1d:
            y_aligned = y_aligned.squeeze(0)

        return x_aligned, y_aligned

    def compensate(
        self,
        input_audio: Tensor,
        output_audio: Tensor,
        calibration: LatencyCalibration,
    ) -> Tuple[Tensor, Tensor]:
        """Apply latency compensation using a calibration result.

        Args:
            input_audio: Input audio tensor.
            output_audio: Output audio tensor.
            calibration: LatencyCalibration from calibrate().

        Returns:
            Tuple of (compensated_input, compensated_output).
        """
        return self.apply_delay(input_audio, output_audio, calibration.delay_samples)


def create_latency_calibrator() -> LatencyCalibrator:
    """Factory function to create a LatencyCalibrator instance."""
    return LatencyCalibrator()
