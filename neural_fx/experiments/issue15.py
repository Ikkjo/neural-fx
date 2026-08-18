"""Generate the preregistered training and evaluation files for issue 15."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "lstm": {
        "hidden_size": 31,
        "num_layers": 1,
        "conv1d": {"filters": 16, "kernel_size": 3, "stride": 4},
        "skip_connection": False,
        "dropout": 0.0,
    },
    "gru": {
        "hidden_size": 37,
        "num_layers": 1,
        "conv1d": {"filters": 16, "kernel_size": 3, "stride": 4},
        "skip_connection": False,
        "dropout": 0.0,
    },
    "wavenet": {
        "layers": 6,
        "stacks": 2,
        "kernel_size": 2,
        "dilation_channels": 8,
        "residual_channels": 8,
        "skip_channels": 16,
    },
    "s4": {"d_model": 20, "d_state": 12, "num_layers": 5, "dropout": 0.0},
}
FINAL_SEEDS = (17, 42, 137)
PILOT_LEARNING_RATES = (0.001, 0.003, 0.01)


def _training_config(
    model_type: str,
    seed: int,
    learning_rate: float,
    epochs: int,
    dataset_root: Path,
    name: str,
) -> dict[str, Any]:
    return {
        "version": "1.0",
        "name": name,
        "model": {
            "type": model_type,
            "input_size": 1,
            "output_size": 1,
            "sample_rate": 48_000,
            "params": MODEL_PARAMETERS[model_type],
        },
        "training": {
            "batch_size": 8,
            "epochs": epochs,
            "segment_length": 16_384,
            "random_segments": True,
            "tbptt": {"enabled": False, "burn_in": 4_096},
            "seed": seed,
            "deterministic": True,
            "early_stopping": False,
            "augmentation": {"enabled": False},
            "num_workers": 4,
        },
        "optimizer": {"type": "adam", "lr": learning_rate},
        "lr_scheduler": {"type": "exponential", "gamma": 0.98},
        "loss": {
            "type": "mse",
            "weights": {"esr": 0.0, "mse": 1.0, "stft": 0.0},
            "pre_emphasis": {"enabled": False, "coef": 0.85},
            "mask_first": 4_096,
        },
        "data": {
            "normalize": False,
            "train": {
                "input": str((dataset_root / "data/train/input.wav").resolve()),
                "target": str((dataset_root / "data/train/target.wav").resolve()),
            },
            "val": {
                "input": str((dataset_root / "data/val/input.wav").resolve()),
                "target": str((dataset_root / "data/val/target.wav").resolve()),
            },
        },
        "latency": {"calibration_duration_seconds": 0},
        "validation": {
            "enabled": True,
            "check_clipping": True,
            "check_dc_offset": True,
        },
    }


def _evaluation_manifest(
    run_id: str,
    run_kind: str,
    model_type: str,
    seed: int,
    learning_rate: float,
    config_path: Path,
    checkpoint_path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    split = "test" if run_kind == "final" else "val"
    return {
        "schema_version": "1.0",
        "experiment_id": run_id,
        "run_kind": run_kind,
        "model": {
            "type": model_type,
            "config": str(config_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
        },
        "dataset": {
            "input_audio": str((dataset_root / f"data/{split}/input.wav").resolve()),
            "target_audio": str((dataset_root / f"data/{split}/target.wav").resolve()),
            "split": "test" if run_kind == "final" else "validation",
            "start_sample": 0,
            "num_samples": 6_240_000,
            "latency_samples": 0,
            "normalization": "none",
            "metric_mask_first": 4_096,
            "inference_chunk_size": 65_536,
            "burn_in_samples": 4_096,
        },
        "training": {
            "seed": seed,
            "learning_rate": learning_rate,
            "deterministic": True,
            "epochs": 40 if run_kind == "final" else 15,
            "selection_metric": "val_loss",
        },
        "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    }


def generate_issue15_run_files(
    dataset_root: Path,
    output_dir: Path,
    checkpoint_root: Path,
    *,
    selected_learning_rates: dict[str, float] | None = None,
) -> list[Path]:
    """Create immutable pilot or final run configs plus evaluation manifests.

    Passing selected learning rates creates the 12 final runs. Omitting them
    creates the 12 preregistered learning-rate pilot runs.
    """
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    checkpoint_root = checkpoint_root.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Run-file output already exists: {output_dir}")
    dataset_manifest_path = dataset_root / "dataset-manifest.json"
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset manifest not found under {dataset_root}"
        )
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    if (
        dataset_manifest.get("schema_version") != "1.0"
        or dataset_manifest.get("normalization") != "none"
    ):
        raise ValueError(
            "Run files require a version 1.0, non-normalized prepared dataset"
        )

    if selected_learning_rates is None:
        run_specs = [
            (model_type, 2026, learning_rate)
            for model_type in MODEL_PARAMETERS
            for learning_rate in PILOT_LEARNING_RATES
        ]
        run_kind = "smoke"
        epochs = 15
    else:
        if set(selected_learning_rates) != set(MODEL_PARAMETERS):
            raise ValueError(
                "Final learning rates must be provided for lstm, gru, wavenet, and s4"
            )
        if any(
            rate not in PILOT_LEARNING_RATES
            for rate in selected_learning_rates.values()
        ):
            raise ValueError(
                "Each selected learning rate must be one of 0.001, 0.003, or 0.01"
            )
        run_specs = [
            (model_type, seed, selected_learning_rates[model_type])
            for model_type in MODEL_PARAMETERS
            for seed in FINAL_SEEDS
        ]
        run_kind = "final"
        epochs = 40

    config_dir = output_dir / "configs"
    manifest_dir = output_dir / "manifests"
    config_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    written: list[Path] = []
    for model_type, seed, learning_rate in run_specs:
        lr_label = str(learning_rate).replace(".", "p")
        run_id = f"issue15-{run_kind}-{model_type}-seed{seed}-lr{lr_label}"
        config_path = config_dir / f"{run_id}.yaml"
        checkpoint_path = checkpoint_root / run_id / "best.ckpt"
        config = _training_config(
            model_type, seed, learning_rate, epochs, dataset_root, run_id
        )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = _evaluation_manifest(
            run_id,
            run_kind,
            model_type,
            seed,
            learning_rate,
            config_path,
            checkpoint_path,
            dataset_root,
        )
        manifest_path = manifest_dir / f"{run_id}.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
        written.extend((config_path, manifest_path))
    return written
