"""Canonical parameter-count targets shared by model families."""

from typing import Final

MODEL_SIZE_TARGETS: Final[dict[str, int]] = {
    "nano": 3_000,
    "small": 18_000,
    "medium": 60_000,
    "large": 126_000,
    "xl": 217_000,
}


def nearest_size_target(parameter_count: int) -> str:
    """Return the canonical size label nearest to ``parameter_count``."""
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    return min(
        MODEL_SIZE_TARGETS,
        key=lambda label: (
            abs(MODEL_SIZE_TARGETS[label] - parameter_count),
            MODEL_SIZE_TARGETS[label],
        ),
    )
