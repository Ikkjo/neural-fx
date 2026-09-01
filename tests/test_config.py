"""Tests for authoritative sample-rate and latency configuration."""

from pathlib import Path

import pytest

from neural_fx.config import LSTMParams, config_from_dict, load_config


def _write_config(
    tmp_path: Path,
    *,
    model_sample_rate: int | None = None,
    data_sample_rate: int | None = None,
    latency_enabled: bool | None = None,
    calibration_duration_seconds: float | None = None,
) -> Path:
    lines = [
        'version: "1.0"',
        'name: "sample_rate_test"',
        "model:",
        '  type: "lstm"',
        "  input_size: 1",
        "  output_size: 1",
    ]
    if model_sample_rate is not None:
        lines.append(f"  sample_rate: {model_sample_rate}")
    lines.extend(
        [
            "  params:",
            "    hidden_size: 8",
            "    num_layers: 1",
            "training: {}",
            "loss:",
            '  type: "mse"',
            "data:",
            "  train:",
            '    input: "input.wav"',
            '    target: "target.wav"',
        ]
    )
    if data_sample_rate is not None:
        lines.append(f"  sample_rate: {data_sample_rate}")
    if latency_enabled is not None or calibration_duration_seconds is not None:
        lines.append("latency:")
    if latency_enabled is not None:
        lines.append(f"  enabled: {str(latency_enabled).lower()}")
    if calibration_duration_seconds is not None:
        lines.append(
            f"  calibration_duration_seconds: {calibration_duration_seconds}",
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def test_model_sample_rate_is_authoritative(tmp_path: Path) -> None:
    """Expose the model sample rate as the immutable authoritative value."""
    config = load_config(_write_config(tmp_path, model_sample_rate=44100))

    assert config.sample_rate == 44100
    assert config.model.sample_rate == 44100
    assert not hasattr(config.data, "sample_rate")

    with pytest.raises(AttributeError):
        config.sample_rate = 48000


@pytest.mark.parametrize("model_sample_rate", [None, 48000])
def test_data_sample_rate_is_rejected(
    tmp_path: Path,
    model_sample_rate: int | None,
) -> None:
    """Reject the removed data-level sample-rate setting immediately."""
    path = _write_config(
        tmp_path,
        model_sample_rate=model_sample_rate,
        data_sample_rate=44100,
    )

    with pytest.raises(ValueError, match="data.sample_rate is no longer supported"):
        load_config(path)


def test_sample_rate_defaults_to_48000(tmp_path: Path) -> None:
    """Default the authoritative model rate to 48 kHz."""
    config = load_config(_write_config(tmp_path))

    assert config.sample_rate == 48000


@pytest.mark.parametrize(
    ("training", "expected"),
    [({}, False), ({"compile": True}, True)],
)
def test_training_compile_setting(training: dict, expected: bool) -> None:
    config = config_from_dict(
        {
            "version": "1.0",
            "name": "compile-setting",
            "model": {"type": "lstm", "params": {"hidden_size": 8}},
            "training": training,
            "loss": {"type": "mse"},
            "data": {"train": {"input": "input.wav", "target": "target.wav"}},
        }
    )

    assert config.training.compile is expected


def test_relative_early_stopping_settings_are_loaded() -> None:
    config = config_from_dict(
        {
            "version": "1.0",
            "name": "relative-early-stopping",
            "model": {"type": "lstm", "params": {"hidden_size": 8}},
            "training": {
                "early_stopping": True,
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 0.005,
                "early_stopping_min_delta_mode": "relative",
            },
            "loss": {"type": "mse"},
            "data": {"train": {"input": "input.wav", "target": "target.wav"}},
        }
    )

    assert config.training.early_stopping is True
    assert config.training.early_stopping_patience == 20
    assert config.training.early_stopping_min_delta == 0.005
    assert config.training.early_stopping_min_delta_mode == "relative"


def test_nam_esr_mode_is_loaded_without_changing_the_default() -> None:
    base = {
        "version": "1.0",
        "name": "esr-mode",
        "model": {"type": "lstm", "params": {"hidden_size": 8}},
        "training": {},
        "data": {"train": {"input": "input.wav", "target": "target.wav"}},
    }

    assert config_from_dict({**base, "loss": {"type": "mse"}}).loss.esr_mode == "legacy"
    assert (
        config_from_dict({**base, "loss": {"type": "mse", "esr_mode": "nam"}})
        .loss.esr_mode
        == "nam"
    )
    with pytest.raises(ValueError, match="esr_mode"):
        config_from_dict({**base, "loss": {"type": "mse", "esr_mode": "other"}})


@pytest.mark.parametrize(
    ("training", "message"),
    [
        ({"early_stopping_patience": -1}, "patience cannot be negative"),
        ({"early_stopping_min_delta": -0.1}, "min_delta cannot be negative"),
        (
            {"early_stopping_min_delta_mode": "percentage"},
            "must be 'absolute' or 'relative'",
        ),
    ],
)
def test_invalid_early_stopping_settings_are_rejected(
    training: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_dict(
            {
                "version": "1.0",
                "name": "invalid-early-stopping",
                "model": {"type": "lstm", "params": {"hidden_size": 8}},
                "training": training,
                "loss": {"type": "mse"},
                "data": {
                    "train": {"input": "input.wav", "target": "target.wav"}
                },
            }
        )


@pytest.mark.parametrize(
    "path",
    [
        Path("configs/models/wavenet/wavenet_nano.yaml"),
        Path("configs/models/wavenet/wavenet_small.yaml"),
    ],
)
def test_shipped_wavenet_configs_enable_compiled_training(path: Path) -> None:
    assert load_config(path).training.compile is True


@pytest.mark.parametrize("sample_rate", [44100, 48000])
def test_calibration_duration_converts_to_five_seconds(
    tmp_path: Path,
    sample_rate: int,
) -> None:
    """Convert calibration duration to samples at the model rate."""
    config = load_config(_write_config(tmp_path, model_sample_rate=sample_rate))

    calibration_samples = round(
        config.sample_rate * config.latency.calibration_duration_seconds
    )

    assert calibration_samples == sample_rate * 5


def test_custom_calibration_duration_is_loaded(tmp_path: Path) -> None:
    """Load a positive custom calibration duration."""
    config = load_config(
        _write_config(
            tmp_path,
            model_sample_rate=44100,
            calibration_duration_seconds=2.5,
        )
    )

    assert config.latency.calibration_duration_seconds == 2.5


def test_zero_calibration_duration_disables_calibration(tmp_path: Path) -> None:
    """Accept zero as the sentinel for disabled latency calibration."""
    config = load_config(
        _write_config(tmp_path, calibration_duration_seconds=0),
    )

    assert config.latency.calibration_duration_seconds == 0


def test_latency_enabled_is_rejected(tmp_path: Path) -> None:
    """Reject the redundant latency enable flag with migration guidance."""
    path = _write_config(tmp_path, latency_enabled=False)

    with pytest.raises(ValueError, match="latency.enabled is no longer supported"):
        load_config(path)


def test_negative_calibration_duration_is_rejected(tmp_path: Path) -> None:
    """Reject calibration durations that cannot represent useful work."""
    path = _write_config(tmp_path, calibration_duration_seconds=-1)

    with pytest.raises(
        ValueError,
        match="calibration_duration_seconds cannot be negative",
    ):
        load_config(path)


def test_null_conv_config_loads_as_disabled() -> None:
    """Accept the natural YAML representation for an optional convolution."""
    config = config_from_dict(
        {
            "version": "1.0",
            "name": "no_conv",
            "model": {
                "type": "lstm",
                "params": {"hidden_size": 8, "conv1d": None},
            },
            "training": {},
            "loss": {"type": "mse"},
            "data": {"train": {"input": "input.wav", "target": "target.wav"}},
        }
    )

    assert isinstance(config.model.params, LSTMParams)
    assert config.model.params.conv1d is None


def test_legacy_replicability_setting_is_ignored() -> None:
    config = config_from_dict(
        {
            "version": "1.0",
            "name": "legacy_validation",
            "model": {
                "type": "lstm",
                "params": {"hidden_size": 8},
            },
            "training": {},
            "loss": {"type": "mse"},
            "data": {"train": {"input": "input.wav", "target": "target.wav"}},
            "validation": {"check_replicability": True},
        }
    )

    assert config.validation is not None
    assert not hasattr(config.validation, "check_replicability")
