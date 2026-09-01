"""Regression tests for the 44.1 kHz nano experiment."""

from pathlib import Path

import pytest
import yaml

from neural_fx.analysis.evaluation import load_experiment_manifest
from neural_fx.config import load_config
from neural_fx.models import create_model_from_config
from scripts.prepare_experiment import (
    DEFAULT_EXPERIMENT,
    check_run_files,
    expected_run_files,
    experiment_splits,
    load_experiment,
    write_run_files,
)
from scripts.run_experiment import build_phase_commands

EXPECTED_PARAMETERS = {"lstm": 3_129, "gru": 2_369, "wavenet": 6_161}


def _run_manifest() -> dict:
    return yaml.safe_load((DEFAULT_EXPERIMENT.parent / "runs.yaml").read_text())


def test_generated_files_are_current_and_idempotent() -> None:
    assert check_run_files(DEFAULT_EXPERIMENT) == []
    assert write_run_files(DEFAULT_EXPERIMENT) == []


def test_experiment_has_eight_pairs_and_three_models_per_pair() -> None:
    experiment = load_experiment(DEFAULT_EXPERIMENT)
    manifest = _run_manifest()
    runs = manifest["runs"]

    assert len(experiment["targets"]) == 8
    assert len(runs) == 24
    assert len({run["id"] for run in runs}) == 24
    for target in experiment["targets"]:
        target_runs = [run for run in runs if run["target"] == target["id"]]
        assert {run["model_type"] for run in target_runs} == {
            "lstm",
            "gru",
            "wavenet",
        }


def test_experiment_supports_multiple_sizes_of_one_model_type(tmp_path: Path) -> None:
    experiment = load_experiment(DEFAULT_EXPERIMENT)
    experiment["targets"] = experiment["targets"][:1]
    large_lstm = dict(experiment["models"][0])
    large_lstm["id"] = "lstm_large"
    experiment["models"] = [experiment["models"][0], large_lstm]
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False))

    files = expected_run_files(experiment_path, output_dir=DEFAULT_EXPERIMENT.parent)
    run_manifest = yaml.safe_load(files[DEFAULT_EXPERIMENT.parent / "runs.yaml"])

    assert [run["model_id"] for run in run_manifest["runs"]] == [
        "lstm_nano",
        "lstm_large",
    ]


def test_training_configs_match_the_reviewed_protocol() -> None:
    for run in _run_manifest()["runs"]:
        config = load_config(Path(run["config"]))
        model = create_model_from_config(config.model)

        assert config.name == run["id"]
        assert config.sample_rate == 44_100
        assert model.num_params == EXPECTED_PARAMETERS[run["model_type"]]
        assert config.training.epochs == 400
        assert config.training.seed == 42
        assert config.training.deterministic is True
        assert config.training.early_stopping is False
        assert config.loss.type == "mse"
        assert config.data.normalize is True
        assert config.data.val is not None
        assert config.latency.calibration_duration_seconds == 0
        assert config.latency.manual_delay == 0
        if run["model_type"] == "wavenet":
            assert config.training.compile is True


def test_evaluation_manifests_use_the_common_held_out_recipe() -> None:
    test_samples = next(
        split.num_samples
        for split in experiment_splits(load_experiment(DEFAULT_EXPERIMENT))
        if split.name == "test"
    )
    for run in _run_manifest()["runs"]:
        manifest = load_experiment_manifest(run["evaluation_manifest"])

        assert manifest["run_kind"] == "final"
        assert manifest["dataset"]["split"] == "test"
        assert manifest["dataset"]["num_samples"] == test_samples
        assert manifest["dataset"]["latency_samples"] == 0
        assert manifest["dataset"]["normalization"] == "paired_peak"
        assert manifest["dataset"]["metric_mask_first"] == 4_096
        assert manifest["burn_in_samples"] == 4_096
        assert manifest["esr_mode"] == "nam"
        assert manifest["esr_pre_emphasis"] is None
        assert manifest["model"]["checkpoint_policy"] == "best"


def test_split_plan_has_segment_aligned_data_and_one_second_guards() -> None:
    experiment = load_experiment(DEFAULT_EXPERIMENT)
    splits = experiment_splits(experiment)
    segment_length = 8_192
    by_name = {split.name: split for split in splits}

    assert by_name["guard_train_val"].num_samples == 44_100
    assert by_name["guard_val_test"].num_samples == 44_100
    assert by_name["val"].start_sample == by_name["guard_train_val"].end_sample
    assert by_name["test"].start_sample == by_name["guard_val_test"].end_sample
    for name in ("train", "val", "test"):
        assert by_name[name].num_samples % segment_length == 0

    segment_counts = {
        name: by_name[name].num_samples // segment_length
        for name in ("train", "val", "test")
    }
    total = sum(segment_counts.values())
    assert segment_counts["train"] / total == pytest.approx(0.8, abs=0.001)
    assert segment_counts["val"] / total == pytest.approx(0.1, abs=0.001)
    assert segment_counts["test"] / total == pytest.approx(0.1, abs=0.001)


@pytest.mark.parametrize(
    ("phase", "expected_count"),
    [
        ("benchmark-initial", 3),
        ("smoke", 3),
        ("train", 24),
        ("benchmark-final", 24),
        ("evaluate", 24),
        ("compare", 8),
    ],
)
def test_experiment_phase_command_counts(phase: str, expected_count: int) -> None:
    commands = build_phase_commands(phase)

    assert len(commands) == expected_count
    assert all("github.com" not in argument for command in commands for argument in command.argv)


def test_smoke_phase_uses_one_epoch_and_only_full_rig() -> None:
    commands = build_phase_commands("smoke")

    assert {
        tuple(
            command.run_id.removeprefix("smoke-nano_44100_").split("_")[:2]
        )
        for command in commands
    } == {
        ("full", "rig")
    }
    for command in commands:
        assert "--max_epochs" in command.argv
        index = command.argv.index("--max_epochs")
        assert command.argv[index + 1] == "1"


def test_full_training_disables_progress_bars() -> None:
    for command in build_phase_commands("train"):
        assert "--no-progress-bar" in command.argv
        assert "--num-workers" in command.argv
        index = command.argv.index("--num-workers")
        assert command.argv[index + 1] == "0"
