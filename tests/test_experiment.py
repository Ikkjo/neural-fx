"""Regression tests for the 44.1 kHz nano experiment."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from neural_fx.analysis.evaluation import load_experiment_manifest
from neural_fx.config import load_config
from neural_fx.models import create_model_from_config
from scripts.prepare_experiment import (
    DEFAULT_EXPERIMENT,
    adopt_experiment_checkpoints,
    check_run_files,
    expected_run_files,
    experiment_splits,
    load_experiment,
    write_run_files,
)
from scripts.run_experiment import _completed_training_epochs, build_phase_commands

EXPECTED_PARAMETERS = {"lstm": 3_129, "gru": 2_369, "wavenet": 6_161}
NAM_RECIPE_EXPERIMENT = Path(
    "configs/experiments/nam_recipe_capacity_44100/experiment.yaml"
).resolve()
NEGATIVE_DELAY_EXPERIMENT = Path(
    "configs/experiments/full_rig_delay_neg041_44100/experiment.yaml"
).resolve()
GEAR_COMPARISON_EXPERIMENT = Path(
    "configs/experiments/gear_comparison_44100/experiment.yaml"
).resolve()


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
    experiment["smoke"]["target"] = experiment["targets"][0]["id"]
    large_lstm = dict(experiment["models"][0])
    large_lstm["id"] = "lstm_large"
    experiment["models"] = [experiment["models"][0], large_lstm]
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False))

    files = expected_run_files(experiment_path, output_dir=tmp_path)
    run_manifest = yaml.safe_load(files[tmp_path / "runs.yaml"])
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    assert [run["model_id"] for run in run_manifest["runs"]] == [
        "lstm_nano",
        "lstm_large",
    ]
    assert [
        command.run_id
        for command in build_phase_commands("benchmark-initial", experiment_path)
    ] == ["initialized-lstm_nano", "initialized-lstm_large"]


def test_adopted_checkpoint_counts_as_completed_training(tmp_path: Path) -> None:
    run_id = "adopted-run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "adopted.meta.json").write_text(
        '{"training_info": {"epochs_trained": 100}}\n'
    )

    experiment = {"paths": {"checkpoint_root": str(tmp_path)}}
    assert _completed_training_epochs(experiment, run_id) == 100


def test_checkpoint_adoption_verifies_and_records_source(tmp_path: Path) -> None:
    experiment = load_experiment(DEFAULT_EXPERIMENT)
    experiment["targets"] = experiment["targets"][:1]
    experiment["models"] = experiment["models"][:1]
    experiment["paths"]["checkpoint_root"] = str(tmp_path / "checkpoints")
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"selected checkpoint")
    source_evaluation = tmp_path / "evaluation.json"
    source_evaluation.write_text(json.dumps({"metrics": {"esr": 0.25}}))
    experiment["adoptions"] = [
        {
            "target": experiment["targets"][0]["id"],
            "model_id": experiment["models"][0]["id"],
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_evaluation": str(source_evaluation),
            "source_training_epochs": 100,
            "selected_epoch": 85,
            "global_step": 20468,
            "validation_score": 0.1,
            "held_out_esr": 0.25,
        }
    ]
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False))

    changed = adopt_experiment_checkpoints(experiment_path)
    destination = next(path for path in changed if path.name == "best.ckpt")
    metadata = json.loads((destination.parent / "adopted.meta.json").read_text())

    assert destination.read_bytes() == source.read_bytes()
    assert metadata["training_info"]["epochs_trained"] == 100
    assert metadata["adoption"]["selected_epoch"] == 85
    source.unlink()
    source_evaluation.unlink()
    assert adopt_experiment_checkpoints(experiment_path) == []


def test_negative_delay_screen_is_one_controlled_wavenet_run() -> None:
    experiment = load_experiment(NEGATIVE_DELAY_EXPERIMENT)
    files = expected_run_files(NEGATIVE_DELAY_EXPERIMENT)
    run_manifest = yaml.safe_load(files[NEGATIVE_DELAY_EXPERIMENT.parent / "runs.yaml"])
    run = run_manifest["runs"][0]
    config = load_config(Path(run["config"]))
    evaluation = load_experiment_manifest(run["evaluation_manifest"])

    assert len(run_manifest["runs"]) == 1
    assert check_run_files(NEGATIVE_DELAY_EXPERIMENT) == []
    assert experiment["targets"] == [
        {
            "id": "full_rig_delay_neg041",
            "file": "full_rig.wav",
            "sha256": "30c67e596387e0231315393e97c188350452ad2113e302978636eb82a0ff5756",
            "delay_samples": -41,
        }
    ]
    assert run["model_id"] == "wavenet_12k"
    assert config.training.epochs == 100
    assert evaluation["dataset"]["latency_samples"] == 0
    assert evaluation["dataset"]["preparation_delay_samples"] == -41
    smoke = build_phase_commands("smoke", NEGATIVE_DELAY_EXPERIMENT)
    worker_flag = smoke[0].argv.index("--num-workers")
    assert smoke[0].argv[worker_flag + 1] == "4"


def test_gear_comparison_has_twelve_runs_and_three_adoptions() -> None:
    experiment = load_experiment(GEAR_COMPARISON_EXPERIMENT)
    manifest = yaml.safe_load(
        (GEAR_COMPARISON_EXPERIMENT.parent / "runs.yaml").read_text()
    )

    assert check_run_files(GEAR_COMPARISON_EXPERIMENT) == []
    assert len(manifest["runs"]) == 12
    assert {target["delay_samples"] for target in experiment["targets"]} == {-41}
    assert {model["id"] for model in experiment["models"]} == {
        "wavenet_12k",
        "lstm_7k",
        "gru_7k",
    }
    assert len(experiment["adoptions"]) == 3


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


def test_nam_recipe_capacity_configs_match_the_approved_plan() -> None:
    assert check_run_files(NAM_RECIPE_EXPERIMENT) == []
    manifest = yaml.safe_load((NAM_RECIPE_EXPERIMENT.parent / "runs.yaml").read_text())
    expected = {
        "wavenet_nano_nam_recipe": (6_161, 127, 0, True),
        "wavenet_nam_budget": (12_129, 6_142, 0, True),
        "lstm_proteus_40": (6_921, 1, 200, False),
    }

    assert len(manifest["runs"]) == 6
    for run in manifest["runs"]:
        config = load_config(run["config"])
        model = create_model_from_config(config.model)
        parameters, receptive_field, burn_in, uses_context = expected[run["model_id"]]
        assert (model.num_params, model.receptive_field) == (
            parameters,
            receptive_field,
        )
        assert config.training.batch_size == 16
        assert config.training.segment_length == 8192
        assert config.training.validation_segment_length == 65_536
        assert config.training.use_receptive_field_context is uses_context
        assert config.optimizer.lr == 0.004
        assert config.optimizer.weight_decay == pytest.approx(3.17e-7)
        assert config.lr_scheduler.gamma == 0.994
        assert config.loss.mask_first == burn_in
        assert config.loss.stft is not None and config.loss.stft.mode == "nam"
        assert config.validation_loss is not None
        assert config.validation_loss.esr_mode == "nam"
        assert config.validation_loss.mask_first == 4096
        assert config.data.normalize is False


def test_nam_recipe_capacity_commands_cover_each_run_without_state_collisions() -> None:
    assert len(build_phase_commands("smoke", NAM_RECIPE_EXPERIMENT)) == 3
    assert len(build_phase_commands("train", NAM_RECIPE_EXPERIMENT)) == 6
    initial = build_phase_commands("benchmark-initial", NAM_RECIPE_EXPERIMENT)
    assert len({command.run_id for command in initial}) == len(initial) == 3
