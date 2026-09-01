#!/usr/bin/env python3
"""Run an experiment serially with resumable status files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__:
    from scripts.prepare_experiment import (
        DEFAULT_EXPERIMENT,
        REPO_ROOT,
        check_run_files,
        load_experiment,
    )
else:
    from prepare_experiment import (
        DEFAULT_EXPERIMENT,
        REPO_ROOT,
        check_run_files,
        load_experiment,
    )


@dataclass(frozen=True)
class ExperimentCommand:
    """One resumable external command in an experiment phase."""

    run_id: str
    argv: tuple[str, ...]


def _repo_path(value: str, repo_root: Path = REPO_ROOT) -> Path:
    return (repo_root / value).resolve()


def _load_runs(experiment_path: Path) -> list[dict[str, Any]]:
    path = experiment_path.resolve().parent / "runs.yaml"
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise ValueError("Run manifest must use schema version 1.0")
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Run manifest must contain at least one run")
    return runs


def _benchmark_args(experiment: dict[str, Any]) -> tuple[str, ...]:
    benchmark = experiment["benchmark"]
    return (
        "--device",
        str(benchmark["device"]),
        "--threads",
        str(benchmark["threads"]),
        "--block-sizes",
        *(str(size) for size in benchmark["block_sizes"]),
        "--num-samples",
        str(benchmark["num_samples"]),
        "--warmup-runs",
        str(benchmark["warmup_runs"]),
        "--measurement-runs",
        str(benchmark["measurement_runs"]),
    )


def build_phase_commands(
    phase: str,
    experiment_path: Path = DEFAULT_EXPERIMENT,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[ExperimentCommand]:
    """Build commands for one phase without executing or writing state."""
    experiment_path = experiment_path.expanduser().resolve()
    experiment = load_experiment(experiment_path)
    runs = _load_runs(experiment_path)
    paths = experiment["paths"]
    python = sys.executable

    if phase == "smoke":
        selected = [run for run in runs if run["target"] == experiment["smoke"]["target"]]
        checkpoint_root = _repo_path(paths["smoke_checkpoint_root"], repo_root)
        return [
            ExperimentCommand(
                f"smoke-{run['id']}",
                (
                    python,
                    str(repo_root / "scripts/train.py"),
                    "--config",
                    str(_repo_path(run["config"], repo_root)),
                    "--checkpoint_dir",
                    str(checkpoint_root),
                    "--max_epochs",
                    str(experiment["smoke"]["epochs"]),
                    "--log_every_n_steps",
                    "1",
                ),
            )
            for run in selected
        ]

    if phase == "train":
        checkpoint_root = _repo_path(paths["checkpoint_root"], repo_root)
        commands = []
        for run in runs:
            argv = [
                python,
                str(repo_root / "scripts/train.py"),
                "--config",
                str(_repo_path(run["config"], repo_root)),
                "--checkpoint_dir",
                str(checkpoint_root),
                "--num-workers",
                str(experiment["training"]["num_workers"]),
                "--no-progress-bar",
            ]
            resume = checkpoint_root / run["id"] / "last.ckpt"
            if resume.is_file():
                argv.extend(("--resume", str(resume)))
            commands.append(ExperimentCommand(run["id"], tuple(argv)))
        return commands

    results_root = _repo_path(paths["results_root"], repo_root)
    if phase == "benchmark-initial":
        selected = [run for run in runs if run["target"] == experiment["smoke"]["target"]]
        return [
            ExperimentCommand(
                f"initialized-{run['model_type']}",
                (
                    python,
                    str(repo_root / "scripts/benchmark.py"),
                    "--config",
                    str(_repo_path(run["config"], repo_root)),
                    "--output",
                    str(
                        results_root
                        / "benchmarks/initialized"
                        / f"{run['model_id']}.json"
                    ),
                    *_benchmark_args(experiment),
                ),
            )
            for run in selected
        ]

    if phase == "benchmark-final":
        return [
            ExperimentCommand(
                run["id"],
                (
                    python,
                    str(repo_root / "scripts/benchmark.py"),
                    "--config",
                    str(_repo_path(run["config"], repo_root)),
                    "--checkpoint",
                    str(_repo_path(run["checkpoint"], repo_root)),
                    "--output",
                    str(results_root / "benchmarks/final" / f"{run['id']}.json"),
                    *_benchmark_args(experiment),
                ),
            )
            for run in runs
        ]

    if phase == "evaluate":
        return [
            ExperimentCommand(
                run["id"],
                (
                    python,
                    str(repo_root / "scripts/evaluate_experiment.py"),
                    "--manifest",
                    str(_repo_path(run["evaluation_manifest"], repo_root)),
                    "--output-dir",
                    str(results_root / "evaluations" / run["id"]),
                    "--device",
                    "cpu",
                ),
            )
            for run in runs
        ]

    if phase == "compare":
        commands = []
        for target in experiment["targets"]:
            target_runs = [run for run in runs if run["target"] == target["id"]]
            result_paths = [
                str(results_root / "evaluations" / run["id"] / "evaluation.json")
                for run in target_runs
            ]
            commands.append(
                ExperimentCommand(
                    f"compare-{target['id']}",
                    (
                        python,
                        str(repo_root / "scripts/compare_evaluations.py"),
                        *result_paths,
                        "--output-dir",
                        str(results_root / "comparisons" / target["id"]),
                    ),
                )
            )
        return commands
    raise ValueError(f"Unknown experiment phase: {phase}")


def _load_state(path: Path, phase: str) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "1.0", "phase": phase, "runs": {}}
    data = json.loads(path.read_text())
    if data.get("schema_version") != "1.0" or data.get("phase") != phase:
        raise ValueError(f"Invalid experiment state file: {path}")
    return data


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _completed_training_epochs(experiment: dict[str, Any], run_id: str) -> int:
    checkpoint_root = _repo_path(experiment["paths"]["checkpoint_root"])
    metadata_path = checkpoint_root / run_id / "last.meta.json"
    if not metadata_path.is_file():
        return 0
    metadata = json.loads(metadata_path.read_text())
    training_info = metadata.get("training_info", {})
    epochs = training_info.get("epochs_trained", 0)
    return epochs if isinstance(epochs, int) else 0


def execute_phase(
    phase: str,
    experiment_path: Path = DEFAULT_EXPERIMENT,
    *,
    dry_run: bool = False,
    rerun_completed: bool = False,
    fail_fast: bool = False,
) -> int:
    """Execute one serial phase, recording failures and continuing by default."""
    stale = check_run_files(experiment_path)
    if stale:
        raise RuntimeError(
            "Generated experiment files are stale; run prepare_experiment.py generate"
        )
    experiment = load_experiment(experiment_path)
    commands = build_phase_commands(phase, experiment_path)
    if dry_run:
        for command in commands:
            print(command.run_id, " ".join(command.argv))
        return 0

    state_path = _repo_path(experiment["paths"]["state_root"]) / f"{phase}.json"
    state = _load_state(state_path, phase)
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    failures = 0
    for command in commands:
        previous = state["runs"].get(command.run_id, {})
        if previous.get("status") == "completed" and not rerun_completed:
            if phase != "train" or _completed_training_epochs(
                experiment, command.run_id
            ) >= experiment["training"]["epochs"]:
                print(f"Skipping completed {command.run_id}")
                continue
            print(f"Resuming incomplete {command.run_id}")
        state["runs"][command.run_id] = {
            "status": "running",
            "command": list(command.argv),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_state(state_path, state)
        result = subprocess.run(command.argv, cwd=REPO_ROOT, env=environment)
        state["runs"][command.run_id].update(
            {
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_state(state_path, state)
        if result.returncode != 0:
            failures += 1
            if fail_fast:
                break
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "benchmark-initial",
            "smoke",
            "train",
            "benchmark-final",
            "evaluate",
            "compare",
        ),
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    return execute_phase(
        args.phase,
        args.experiment,
        dry_run=args.dry_run,
        rerun_completed=args.rerun_completed,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    raise SystemExit(main())
