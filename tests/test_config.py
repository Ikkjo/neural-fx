"""Tests for configuration loading and sample-rate migration."""

import pytest

from neural_fx.config import load_config


def _write_config(
    tmp_path,
    *,
    model_sample_rate: int | None = None,
    data_sample_rate: int | None = None,
    calibration_duration_seconds: float | None = None,
):
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
    if calibration_duration_seconds is not None:
        lines.extend(
            [
                "latency:",
                "  calibration_duration_seconds: "
                f"{calibration_duration_seconds}",
            ]
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def test_model_sample_rate_is_authoritative(tmp_path):
    config = load_config(_write_config(tmp_path, model_sample_rate=44100))

    assert config.sample_rate == 44100
    assert config.model.sample_rate == 44100
    assert not hasattr(config.data, "sample_rate")

    with pytest.raises(AttributeError):
        config.sample_rate = 48000


def test_legacy_data_sample_rate_is_migrated(tmp_path):
    path = _write_config(tmp_path, data_sample_rate=44100)

    with pytest.warns(DeprecationWarning, match="data.sample_rate is deprecated"):
        config = load_config(path)

    assert config.sample_rate == 44100


def test_matching_legacy_sample_rate_warns(tmp_path):
    path = _write_config(
        tmp_path, model_sample_rate=48000, data_sample_rate=48000
    )

    with pytest.warns(DeprecationWarning, match="data.sample_rate is deprecated"):
        config = load_config(path)

    assert config.sample_rate == 48000


def test_conflicting_sample_rates_fail(tmp_path):
    path = _write_config(
        tmp_path, model_sample_rate=48000, data_sample_rate=44100
    )

    with pytest.raises(ValueError, match="Conflicting sample rates"):
        load_config(path)


def test_sample_rate_defaults_to_48000(tmp_path):
    config = load_config(_write_config(tmp_path))

    assert config.sample_rate == 48000


@pytest.mark.parametrize("sample_rate", [44100, 48000])
def test_calibration_duration_converts_to_five_seconds(tmp_path, sample_rate):
    config = load_config(_write_config(tmp_path, model_sample_rate=sample_rate))

    calibration_samples = round(
        config.sample_rate * config.latency.calibration_duration_seconds
    )

    assert calibration_samples == sample_rate * 5


def test_custom_calibration_duration_is_loaded(tmp_path):
    config = load_config(
        _write_config(
            tmp_path,
            model_sample_rate=44100,
            calibration_duration_seconds=2.5,
        )
    )

    assert config.latency.calibration_duration_seconds == 2.5
