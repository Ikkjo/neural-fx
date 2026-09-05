"""Shared quality-measurement policy for controlled evaluation and monitoring."""

from __future__ import annotations

import math
from copy import deepcopy
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

SILENCE_POLICY_ID = "digital_silence_v1"
RELATIVE_METRICS = frozenset({"esr", "multi_resolution_stft_distance"})
SILENCE_STFT = {
    "mode": "legacy",
    "fft_sizes": [512, 1024, 2048],
    "hop_sizes": [128, 256, 512],
    "win_sizes": [512, 1024, 2048],
}
SILENCE_POLICY: dict[str, Any] = {
    "id": SILENCE_POLICY_ID,
    "target_rule": "every target sample equals exactly zero",
    "unavailable_relative_metrics": sorted(RELATIVE_METRICS),
    "relative_aggregation": "mean of eligible scoring segments",
    "no_eligible_relative_score": None,
    "stft": SILENCE_STFT,
}


def silence_policy_metadata() -> dict[str, Any]:
    """Return a JSON-safe copy of the fixed silence policy metadata."""
    return deepcopy(SILENCE_POLICY)


def scoring_diagnostics(prediction: Tensor, target: Tensor) -> dict[str, Any]:
    """Measure absolute output behavior and classify target-only digital silence."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match: {prediction.shape} != {target.shape}"
        )
    if prediction.numel() == 0:
        raise ValueError("Prediction and target must not be empty")

    error = prediction - target
    values = {
        "mse": torch.mean(error.square()).item(),
        "prediction_rms": torch.mean(prediction.square()).sqrt().item(),
        "prediction_abs_peak": prediction.abs().max().item(),
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("Absolute scoring diagnostics must be finite")

    digital_silence = bool(torch.all(target == 0).item())
    return {
        "digital_silence": digital_silence,
        "relative_score_status": "excluded" if digital_silence else "eligible",
        **{name: float(value) for name, value in values.items()},
    }


def average_eligible(values: list[float | None]) -> tuple[float | None, int, int]:
    """Average finite relative scores and return mean, eligible, excluded counts."""
    eligible = [value for value in values if value is not None]
    if any(not math.isfinite(value) for value in eligible):
        raise ValueError("Relative scores must be finite or null")
    return (
        (float(fmean(eligible)) if eligible else None),
        len(eligible),
        len(values) - len(eligible),
    )
