#!/usr/bin/env python3
"""Generate, validate, and prepare an experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from neural_fx.config import load_config
from neural_fx.models import create_model_from_config
from neural_fx.preprocessing.experiment_data import (
    SplitSpec,
    prepare_aligned_audio,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = REPO_ROOT / "configs/experiments/nano_44100/experiment.yaml"
RUNS_FILE = "runs.yaml"


def load_experiment(path: Path) -> dict[str, Any]:
    """Load the fixed experiment description and reject incomplete inputs."""
    path = path.expanduser().resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("Experiment must contain a mapping")
    if data.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported experiment schema: {data.get('schema_version')}")
    for key in (
        "experiment_id",
        "run_id_prefix",
        "paths",
        "audio",
        "targets",
        "models",
        "training",
        "evaluation",
        "benchmark",
        "smoke",
    ):
        if key not in data:
            raise ValueError(f"Experiment is missing '{key}'")
    if not data["targets"]:
        raise ValueError("Experiment requires at least one target")
    if not data["models"]:
        raise ValueError("Experiment requires at least one model")
    if len({target["id"] for target in data["targets"]}) != len(data["targets"]):
        raise ValueError("Experiment target ids must be unique")
    if len({model["id"] for model in data["models"]}) != len(data["models"]):
        raise ValueError("Experiment model ids must be unique")
    return data


def experiment_splits(experiment: dict[str, Any]) -> tuple[SplitSpec, ...]:
    """Parse and validate the shared split plan."""
    splits = tuple(SplitSpec(**item) for item in experiment["audio"]["splits"])
    audio_splits = [split for split in splits if split.write_audio]
    if [split.name for split in audio_splits] != ["train", "val", "test"]:
        raise ValueError("Experiment must write train, val, and test splits")
    return splits


def _repo_path(value: str, repo_root: Path) -> Path:
    return (repo_root / value).resolve()


def _relative_path(path: Path, origin: Path) -> str:
    return Path(os.path.relpath(path, start=origin)).as_posix()


def _repository_reference(path: Path, repo_root: Path) -> str:
    """Return a repository-relative path when possible, else an absolute path."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _prepared_split_path(
    experiment: dict[str, Any], target_id: str, split: str, role: str
) -> str:
    prepared_root = Path(experiment["paths"]["prepared_root"])
    return (prepared_root / target_id / "data" / split / f"{role}.wav").as_posix()


def _run_id(
    experiment: dict[str, Any], target_id: str, model_spec: dict[str, Any]
) -> str:
    seed = experiment["training"]["seed"]
    return f"{experiment['run_id_prefix']}_{target_id}_{model_spec['id']}_seed{seed}"


def _training_config(
    experiment: dict[str, Any],
    target: dict[str, Any],
    model_spec: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    base_path = _repo_path(model_spec["base_config"], repo_root)
    config = copy.deepcopy(yaml.safe_load(base_path.read_text()))
    config["name"] = _run_id(experiment, target["id"], model_spec)
    config["model"]["sample_rate"] = experiment["audio"]["sample_rate"]

    training = config["training"]
    experiment_training = experiment["training"]
    stopping = experiment_training["early_stopping"]
    training.update(
        {
            "epochs": experiment_training["epochs"],
            "seed": experiment_training["seed"],
            "deterministic": experiment_training["deterministic"],
            "early_stopping": stopping["enabled"],
        }
    )
    if stopping["enabled"]:
        training.update(
            {
                "early_stopping_patience": stopping["patience"],
                "early_stopping_min_delta": stopping["min_delta"],
                "early_stopping_min_delta_mode": stopping["min_delta_mode"],
            }
        )
    config["data"] = {
        "normalize": experiment_training["normalize"],
        "train": {
            "input": _prepared_split_path(experiment, target["id"], "train", "input"),
            "target": _prepared_split_path(
                experiment, target["id"], "train", "target"
            ),
        },
        "val": {
            "input": _prepared_split_path(experiment, target["id"], "val", "input"),
            "target": _prepared_split_path(experiment, target["id"], "val", "target"),
        },
    }
    config["latency"] = {
        "method": "manual",
        "manual_delay": 0,
        "calibration_duration_seconds": 0,
    }
    config["validation"] = {
        "enabled": True,
        "check_clipping": True,
        "check_dc_offset": True,
        "ignore_warnings": False,
    }
    return config


def _evaluation_manifest(
    experiment: dict[str, Any],
    target: dict[str, Any],
    model_spec: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    model_type = model_spec["type"]
    run_id = _run_id(experiment, target["id"], model_spec)
    paths = experiment["paths"]
    checkpoint = _repo_path(
        f"{paths['checkpoint_root']}/{run_id}/best.ckpt", repo_root
    )
    benchmark = _repo_path(
        f"{paths['results_root']}/benchmarks/final/{run_id}.json", repo_root
    )
    test_input = _repo_path(
        _prepared_split_path(experiment, target["id"], "test", "input"), repo_root
    )
    test_target = _repo_path(
        _prepared_split_path(experiment, target["id"], "test", "target"), repo_root
    )
    test_samples = next(
        split["num_samples"]
        for split in experiment["audio"]["splits"]
        if split["name"] == "test"
    )
    evaluation = experiment["evaluation"]
    training = experiment["training"]
    stopping = training["early_stopping"]
    manifest = {
        "schema_version": "1.0",
        "experiment_id": run_id,
        "run_kind": "final",
        "inference_chunk_size": evaluation["inference_chunk_size"],
        "esr_mode": evaluation["esr_mode"],
        "burn_in_samples": evaluation["burn_in_samples"],
        "esr_pre_emphasis": evaluation["esr_pre_emphasis"],
        "model": {
            "type": model_type,
            "config": _relative_path(config_path, manifest_path.parent),
            "checkpoint": _relative_path(checkpoint, manifest_path.parent),
            "checkpoint_policy": "best",
            "benchmark_result": _relative_path(benchmark, manifest_path.parent),
        },
        "dataset": {
            "input_audio": _relative_path(test_input, manifest_path.parent),
            "target_audio": _relative_path(test_target, manifest_path.parent),
            "split": "test",
            "start_sample": 0,
            "num_samples": test_samples,
            "latency_samples": 0,
            "preparation_delay_samples": target["delay_samples"],
            "normalization": "paired_peak" if training["normalize"] else "none",
            "metric_mask_first": evaluation["burn_in_samples"],
        },
        "training": {
            "seed": training["seed"],
            "epochs": training["epochs"],
            "deterministic": training["deterministic"],
            "selection_metric": "val_loss",
            "early_stopping": stopping["enabled"],
        },
    }
    if stopping["enabled"]:
        manifest["training"].update(
            {
                "early_stopping_patience": stopping["patience"],
                "early_stopping_min_delta": stopping["min_delta"],
                "early_stopping_min_delta_mode": stopping["min_delta_mode"],
            }
        )
    return manifest


def expected_run_files(
    experiment_path: Path,
    *,
    output_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[Path, str]:
    """Build every generated YAML file without writing it."""
    experiment_path = experiment_path.expanduser().resolve()
    experiment = load_experiment(experiment_path)
    output_dir = (output_dir or experiment_path.parent).resolve()
    files: dict[Path, str] = {}
    runs: list[dict[str, Any]] = []
    for target in experiment["targets"]:
        for model_spec in experiment["models"]:
            model_type = model_spec["type"]
            run_id = _run_id(experiment, target["id"], model_spec)
            model_id = model_spec["id"]
            config_path = output_dir / "runs" / target["id"] / f"{model_id}.yaml"
            manifest_path = (
                output_dir
                / "evaluations"
                / target["id"]
                / f"{model_id}.yaml"
            )
            config = _training_config(experiment, target, model_spec, repo_root)
            manifest = _evaluation_manifest(
                experiment,
                target,
                model_spec,
                config_path,
                manifest_path,
                repo_root,
            )
            files[config_path] = yaml.safe_dump(config, sort_keys=False)
            files[manifest_path] = yaml.safe_dump(manifest, sort_keys=False)
            runs.append(
                {
                    "id": run_id,
                    "target": target["id"],
                    "model_id": model_id,
                    "model_type": model_type,
                    "config": _repository_reference(config_path, repo_root),
                    "evaluation_manifest": _repository_reference(
                        manifest_path, repo_root
                    ),
                    "checkpoint": (
                        Path(experiment["paths"]["checkpoint_root"])
                        / run_id
                        / "best.ckpt"
                    ).as_posix(),
                }
            )
    run_manifest = {
        "schema_version": "1.0",
        "experiment_id": experiment["experiment_id"],
        "experiment": _repository_reference(experiment_path, repo_root),
        "runs": runs,
    }
    files[output_dir / RUNS_FILE] = yaml.safe_dump(run_manifest, sort_keys=False)
    return files


def write_run_files(
    experiment_path: Path,
    *,
    output_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Write the deterministic run set and return paths that changed."""
    changed: list[Path] = []
    for path, content in expected_run_files(
        experiment_path, output_dir=output_dir, repo_root=repo_root
    ).items():
        if path.is_file() and path.read_text() == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        changed.append(path)
    return changed


def check_run_files(
    experiment_path: Path,
    *,
    output_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Return generated paths that are missing, stale, or invalid."""
    stale: list[Path] = []
    expected = expected_run_files(
        experiment_path, output_dir=output_dir, repo_root=repo_root
    )
    for path, content in expected.items():
        if not path.is_file() or path.read_text() != content:
            stale.append(path)
    if stale:
        return stale

    experiment = load_experiment(experiment_path)
    expected_parameters = {
        model["id"]: model["expected_parameters"] for model in experiment["models"]
    }
    for path in expected:
        if "/runs/" not in path.as_posix():
            continue
        config = load_config(path)
        model = create_model_from_config(config.model)
        if model.num_params != expected_parameters[path.stem]:
            stale.append(path)
    return stale


def prepare_experiment_audio(
    experiment_path: Path, *, repo_root: Path = REPO_ROOT
) -> list[Path]:
    """Prepare every aligned audio pair using the experiment split plan."""
    experiment = load_experiment(experiment_path)
    source_root = _repo_path(experiment["paths"]["source_root"], repo_root)
    prepared_root = _repo_path(experiment["paths"]["prepared_root"], repo_root)
    input_path = source_root / experiment["audio"]["input_file"]
    written: list[Path] = []
    for target in experiment["targets"]:
        output = prepared_root / target["id"]
        existed = output.exists()
        prepare_aligned_audio(
            input_path,
            source_root / target["file"],
            output,
            splits=experiment_splits(experiment),
            target_sample_rate=experiment["audio"]["sample_rate"],
            target_delay_source_samples=target["delay_samples"],
            expected_input_sha256=experiment["audio"]["input_sha256"],
            expected_target_sha256=target["sha256"],
            experiment_id=f"{experiment['experiment_id']}-{target['id']}",
        )
        if not existed:
            written.append(output)
    return written


def adopt_experiment_checkpoints(
    experiment_path: Path, *, repo_root: Path = REPO_ROOT
) -> list[Path]:
    """Copy verified completed checkpoints into their final run directories."""
    experiment = load_experiment(experiment_path)
    written: list[Path] = []
    targets = {target["id"]: target for target in experiment["targets"]}
    models = {model["id"]: model for model in experiment["models"]}
    checkpoint_root = _repo_path(experiment["paths"]["checkpoint_root"], repo_root)
    for adoption in experiment.get("adoptions", []):
        target = targets[adoption["target"]]
        model = models[adoption["model_id"]]
        run_id = _run_id(experiment, target["id"], model)
        destination_dir = checkpoint_root / run_id
        destination = destination_dir / "best.ckpt"
        metadata_path = destination_dir / "adopted.meta.json"
        expected_hash = adoption["source_checkpoint_sha256"]
        if destination.is_file() and metadata_path.is_file():
            if sha256_file(destination) != expected_hash:
                raise FileExistsError(f"Adopted checkpoint differs: {destination}")
            continue

        source = _repo_path(adoption["source_checkpoint"], repo_root)
        source_hash = sha256_file(source)
        if source_hash != expected_hash:
            raise ValueError(f"Adopted checkpoint hash does not match: {source}")
        source_evaluation = _repo_path(adoption["source_evaluation"], repo_root)
        evaluation = json.loads(source_evaluation.read_text())
        if evaluation["metrics"]["esr"] != adoption["held_out_esr"]:
            raise ValueError(f"Adopted evaluation ESR does not match: {source_evaluation}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if sha256_file(destination) != source_hash:
                raise FileExistsError(f"Adopted checkpoint differs: {destination}")
        else:
            temporary = destination.with_suffix(".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            written.append(destination)

        metadata = {
            "schema_version": "1.0",
            "checkpoint_file": "best.ckpt",
            "run_id": run_id,
            "training_info": {
                "epochs_trained": adoption["source_training_epochs"],
            },
            "adoption": {
                "source_checkpoint": _repository_reference(source, repo_root),
                "source_checkpoint_sha256": source_hash,
                "source_evaluation": _repository_reference(
                    source_evaluation, repo_root
                ),
                "selected_epoch": adoption["selected_epoch"],
                "global_step": adoption["global_step"],
                "validation_score": adoption["validation_score"],
                "held_out_esr": adoption["held_out_esr"],
                "preparation_delay_samples": target["delay_samples"],
            },
        }
        content = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        if not metadata_path.is_file() or metadata_path.read_text() != content:
            metadata_path.write_text(content)
            written.append(metadata_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate", help="Write the tracked run files")
    subparsers.add_parser("check", help="Check generated files without changing them")
    subparsers.add_parser("prepare-audio", help="Write ignored aligned audio splits")
    subparsers.add_parser(
        "adopt-checkpoints", help="Copy verified completed checkpoints into this experiment"
    )
    args = parser.parse_args()

    if args.command == "generate":
        changed = write_run_files(args.experiment)
        print(f"Generated experiment run files; changed {len(changed)} path(s)")
        return 0
    if args.command == "check":
        stale = check_run_files(args.experiment)
        if stale:
            for path in stale:
                print(f"STALE {path}")
            return 1
        print("Experiment run files match the experiment")
        return 0
    if args.command == "prepare-audio":
        written = prepare_experiment_audio(args.experiment)
        print(f"Prepared experiment audio; created {len(written)} target directorie(s)")
        return 0
    written = adopt_experiment_checkpoints(args.experiment)
    print(f"Adopted experiment checkpoints; changed {len(written)} path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
