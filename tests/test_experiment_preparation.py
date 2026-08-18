"""Tests for the fixed issue-15 preparation and run contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torchaudio
import yaml

from neural_fx.config import config_from_dict, load_config
from neural_fx.experiments.issue15 import generate_issue15_run_files
from neural_fx.models import create_model_from_config
from neural_fx.preprocessing.experiment_data import SplitSpec, prepare_aligned_audio
from neural_fx.training.lightning_module import NeuralFXModule
from scripts.train import create_trainer, publish_best_checkpoint


def _write_delayed_pair(root: Path) -> tuple[Path, Path]:
    sample_rate = 8_000
    delay = 4
    time = torch.arange(5_000) / sample_rate
    input_audio = (0.4 * torch.sin(2 * torch.pi * 173 * time)).unsqueeze(0)
    target_audio = torch.zeros_like(input_audio)
    target_audio[..., delay:] = -0.5 * input_audio[..., :-delay]
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
    with patch("scripts.train.L.Trainer") as trainer:
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


def test_final_run_generator_writes_matched_reproducible_configs(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "dataset-manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "normalization": "none"})
    )
    output = tmp_path / "runs"
    selected_rates = {"lstm": 0.001, "gru": 0.003, "wavenet": 0.01, "s4": 0.001}

    written = generate_issue15_run_files(
        dataset_root,
        output,
        tmp_path / "checkpoints",
        selected_learning_rates=selected_rates,
    )

    assert len(written) == 24
    expected_parameters = {"lstm": 6_176, "gru": 6_211, "wavenet": 6_161, "s4": 6_161}
    configs = list((output / "configs").glob("*.yaml"))
    assert len(configs) == 12
    for config_path in configs:
        raw = yaml.safe_load(config_path.read_text())
        config = load_config(config_path)
        model = create_model_from_config(config.model)
        assert raw["training"]["deterministic"] is True
        assert raw["training"]["early_stopping"] is False
        assert raw["data"]["normalize"] is False
        assert config.training.deterministic is True
        assert config.training.early_stopping is False
        assert config.data.normalize is False
        assert model.num_params == expected_parameters[config.model.type]

    with pytest.raises(FileExistsError, match="already exists"):
        generate_issue15_run_files(
            dataset_root,
            output,
            tmp_path / "checkpoints",
            selected_learning_rates=selected_rates,
        )
