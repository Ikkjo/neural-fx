"""Tests for shared contracts used by future model implementations."""

import pytest

from neural_fx.config import SSMParams, WaveNetParams, _load_model_params
from neural_fx.models.errors import (
    OptionalDependencyError,
    UnsupportedExportFormatError,
)
from neural_fx.models.sizing import MODEL_SIZE_TARGETS, nearest_size_target


def test_wavenet_layers_are_per_stack():
    params = WaveNetParams(layers=4, stacks=2)

    assert params.layers == 4
    assert params.stacks == 2
    assert 2 ** (params.layers - 1) == 8


@pytest.mark.parametrize(
    "field",
    [
        "layers",
        "stacks",
        "kernel_size",
        "dilation_channels",
        "residual_channels",
        "skip_channels",
    ],
)
def test_wavenet_rejects_non_positive_values(field):
    kwargs = {"layers": 4, field: 0}

    with pytest.raises(ValueError, match=field):
        WaveNetParams(**kwargs)


def test_ssm_params_load_for_s4_and_mamba():
    raw = {
        "d_model": 24,
        "d_state": 64,
        "num_layers": 3,
        "dropout": 0.1,
        "prenorm": False,
        "d_conv": 4,
        "expand": 2,
    }

    for model_type in ("s4", "mamba"):
        params = _load_model_params(model_type, raw)
        assert params == SSMParams(**raw)


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_ssm_rejects_invalid_dropout(dropout):
    with pytest.raises(ValueError, match="dropout"):
        SSMParams(dropout=dropout)


def test_model_size_targets_are_ordered_and_stable():
    assert list(MODEL_SIZE_TARGETS) == ["nano", "small", "medium", "large", "xl"]
    assert list(MODEL_SIZE_TARGETS.values()) == [
        3_000,
        18_000,
        60_000,
        126_000,
        217_000,
    ]
    assert nearest_size_target(58_000) == "medium"


def test_shared_model_errors_are_actionable():
    export_error = UnsupportedExportFormatError("wavenet", "rtneural")
    dependency_error = OptionalDependencyError(
        "Mamba", "mamba-ssm", "pip install -e .[mamba]"
    )

    assert "wavenet" in str(export_error)
    assert "rtneural" in str(export_error)
    assert "mamba-ssm" in str(dependency_error)
    assert "pip install -e .[mamba]" in str(dependency_error)
