from pathlib import Path

import pytest
import torch
import torchaudio

from neural_fx.data.audio import load_audio_pair
from neural_fx.preprocessing.latency import LatencyCalibration


def test_load_audio_pair_resamples_each_source_rate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    target_path = tmp_path / "target.wav"
    torchaudio.save(input_path, torch.linspace(-0.5, 0.5, 16_000).unsqueeze(0), 16_000)
    torchaudio.save(target_path, torch.linspace(-0.5, 0.5, 8_000).unsqueeze(0), 8_000)

    pair = load_audio_pair(
        input_path,
        target_path,
        sample_rate=16_000,
        normalize=False,
    )

    assert pair.sample_rate == 16_000
    assert pair.input_audio.shape == pair.target_audio.shape == (1, 16_000)


def test_load_audio_pair_mixes_to_mono_and_preserves_relative_level(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.wav"
    target_path = tmp_path / "target.wav"
    input_audio = torch.stack((torch.full((4_000,), 0.8), torch.full((4_000,), 0.4)))
    target_audio = torch.full((1, 4_000), 0.3)
    torchaudio.save(input_path, input_audio, 8_000)
    torchaudio.save(target_path, target_audio, 8_000)

    pair = load_audio_pair(input_path, target_path, sample_rate=8_000)

    assert pair.input_audio.shape == pair.target_audio.shape == (1, 4_000)
    assert pair.input_audio.abs().max() == pytest.approx(1.0, abs=1e-4)
    assert pair.target_audio.abs().max() == pytest.approx(0.5, abs=1e-4)


def test_load_audio_pair_applies_latency_before_final_alignment(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    target_path = tmp_path / "target.wav"
    input_audio = torch.linspace(-0.5, 0.5, 4_000).unsqueeze(0)
    target_audio = torch.cat((torch.zeros(1, 4), input_audio), dim=-1)
    torchaudio.save(input_path, input_audio, 8_000)
    torchaudio.save(target_path, target_audio, 8_000)

    with pytest.warns(UserWarning, match="latency compensation of 4 samples"):
        pair = load_audio_pair(
            input_path,
            target_path,
            sample_rate=8_000,
            normalize=False,
            latency_calibration=LatencyCalibration(4, "manual", 1.0),
        )

    assert pair.input_audio.shape == pair.target_audio.shape == (1, 4_000)
    assert torch.allclose(pair.input_audio, pair.target_audio, atol=1e-4)
