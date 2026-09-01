"""Tests for reproducible experiment preparation and training artifacts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torchaudio

from neural_fx.config import config_from_dict
from neural_fx.preprocessing.experiment_data import SplitSpec, prepare_aligned_audio
from neural_fx.training.lightning_module import NeuralFXModule
from neural_fx.training.run import create_trainer, publish_best_checkpoint


def _write_delayed_pair(root: Path, delay: int = 4) -> tuple[Path, Path]:
    sample_rate = 8_000
    time = torch.arange(5_000) / sample_rate
    input_audio = (0.4 * torch.sin(2 * torch.pi * 173 * time)).unsqueeze(0)
    target_audio = torch.zeros_like(input_audio)
    if delay > 0:
        target_audio[..., delay:] = -0.5 * input_audio[..., :-delay]
    elif delay < 0:
        target_audio[..., :delay] = -0.5 * input_audio[..., -delay:]
    else:
        target_audio = -0.5 * input_audio
    input_path = root / "input.wav"
    target_path = root / "target.wav"
    torchaudio.save(
        input_path, input_audio, sample_rate, encoding="PCM_F", bits_per_sample=32
    )
    torchaudio.save(
        target_path, target_audio, sample_rate, encoding="PCM_F", bits_per_sample=32
    )
    return input_path, target_path


def _small_splits() -> tuple[SplitSpec, ...]:
    return (
        SplitSpec("train", 0, 2_000),
        SplitSpec("guard_train_val", 2_000, 200, write_audio=False),
        SplitSpec("val", 2_200, 1_000),
        SplitSpec("guard_val_test", 3_200, 200, write_audio=False),
        SplitSpec("test", 3_400, 1_000),
    )


def test_preparation_aligns_resamples_and_preserves_relative_gain(
    tmp_path: Path,
) -> None:
    input_path, target_path = _write_delayed_pair(tmp_path)
    output = tmp_path / "prepared"

    manifest = prepare_aligned_audio(
        input_path,
        target_path,
        output,
        target_sample_rate=16_000,
        target_delay_source_samples=4,
        splits=_small_splits(),
    )

    train_input, input_rate = torchaudio.load(output / "data/train/input.wav")
    train_target, target_rate = torchaudio.load(output / "data/train/target.wav")
    assert input_rate == target_rate == 16_000
    assert train_input.shape[-1] == train_target.shape[-1] == 2_000
    assert torch.allclose(train_target, -0.5 * train_input, atol=1e-4)
    assert manifest["normalization"] == "none"
    assert manifest["resampling"]["passes_per_signal"] == 1
    assert manifest["splits"]["train"]["relative_gain"][
        "target_to_input_rms_ratio"
    ] == pytest.approx(0.5, rel=1e-4)
    assert not (output / "data/guard_train_val").exists()


def test_preparation_aligns_a_target_that_leads_the_input(tmp_path: Path) -> None:
    input_path, target_path = _write_delayed_pair(tmp_path, delay=-4)
    output = tmp_path / "prepared"

    manifest = prepare_aligned_audio(
        input_path,
        target_path,
        output,
        target_sample_rate=16_000,
        target_delay_source_samples=-4,
        splits=_small_splits(),
    )

    train_input, _ = torchaudio.load(output / "data/train/input.wav")
    train_target, _ = torchaudio.load(output / "data/train/target.wav")
    assert torch.allclose(train_target, -0.5 * train_input, atol=1e-4)
    assert manifest["alignment"]["operation"] == "trim input head; trim target tail"


def test_preparation_reuses_exact_output_and_rejects_changes(tmp_path: Path) -> None:
    input_path, target_path = _write_delayed_pair(tmp_path)
    output = tmp_path / "prepared"
    kwargs = {
        "target_sample_rate": 16_000,
        "target_delay_source_samples": 4,
        "splits": _small_splits(),
    }
    first = prepare_aligned_audio(input_path, target_path, output, **kwargs)
    second = prepare_aligned_audio(input_path, target_path, output, **kwargs)
    assert second == first

    with (output / "data/train/input.wav").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        prepare_aligned_audio(input_path, target_path, output, **kwargs)

    target_audio, sample_rate = torchaudio.load(target_path)
    torchaudio.save(
        target_path,
        target_audio * 0.9,
        sample_rate,
        encoding="PCM_F",
        bits_per_sample=32,
    )
    with pytest.raises(FileExistsError, match="different specification"):
        prepare_aligned_audio(input_path, target_path, output, **kwargs)


def test_normalize_and_deterministic_flags_reach_runtime_construction() -> None:
    config = config_from_dict(
        {
            "version": "1.0",
            "name": "prepared-levels",
            "model": {"type": "lstm", "params": {"hidden_size": 4}},
            "training": {"deterministic": True},
            "loss": {"type": "mse"},
            "data": {
                "normalize": False,
                "train": {"input": "train-in.wav", "target": "train-out.wav"},
                "val": {"input": "val-in.wav", "target": "val-out.wav"},
            },
        }
    )
    module = NeuralFXModule(torch.nn.Identity(), config)

    with patch("neural_fx.training.lightning_module.AudioDataset") as dataset:
        module._create_train_dataset()
        module._create_val_dataset()
    assert dataset.call_args_list[0].kwargs["normalize"] is False
    assert dataset.call_args_list[1].kwargs["normalize"] is False

    trainer_config = SimpleNamespace(training=SimpleNamespace(deterministic=True))
    with patch("neural_fx.training.run.L.Trainer") as trainer:
        create_trainer(trainer_config, {"max_epochs": 1})
    trainer.assert_called_once_with(deterministic=True, max_epochs=1)


def test_best_checkpoint_is_published_at_manifest_path(tmp_path: Path) -> None:
    selected = tmp_path / "epoch=03-val_loss=0.1.ckpt"
    selected.write_bytes(b"checkpoint")

    canonical = publish_best_checkpoint(
        selected.as_posix(), tmp_path / "runs", "experiment"
    )

    assert canonical == (tmp_path / "runs/experiment/best.ckpt").resolve()
    assert canonical.read_bytes() == b"checkpoint"
