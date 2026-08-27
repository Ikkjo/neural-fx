#!/usr/bin/env python3
"""Train a neural audio-effects model from a YAML configuration."""

import argparse
from pathlib import Path

from neural_fx.config import load_config
from neural_fx.training.run import (
    TrainingDataValidationError,
    TrainingRun,
    create_trainer,
    publish_best_checkpoint,
    run_data_validation,
    run_latency_calibration,
    run_training,
)

__all__ = [
    "create_trainer",
    "publish_best_checkpoint",
    "run_data_validation",
    "run_latency_calibration",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--max_epochs", type=int, help="Override max epochs")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable compiled model training",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("./lightning_logs"),
        help="Checkpoint directory",
    )
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume from")
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Validation check interval as a fraction of an epoch",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="Write CSV and TensorBoard metrics every N training steps",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training even when a GPU is available",
    )
    parser.add_argument(
        "--latency_method",
        choices=("xcorr", "manual"),
        help="Override the latency calibration method",
    )
    parser.add_argument(
        "--latency_manual",
        type=int,
        help="Override the manual latency delay in samples",
    )
    parser.add_argument(
        "--ignore_checks",
        action="store_true",
        help="Continue after data validation failures",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Override early-stopping patience in epochs",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate an analysis report after training",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run = TrainingRun.resolve(
        load_config(args.config),
        checkpoint_dir=args.checkpoint_dir,
        gpus=args.gpus,
        force_cpu=args.cpu,
        resume_path=args.resume,
        ignore_checks=args.ignore_checks,
        patience=args.patience,
        plot=args.plot,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.max_epochs,
        compile=args.compile,
        latency_method=args.latency_method,
        latency_manual=args.latency_manual,
    )
    try:
        run_training(run)
    except TrainingDataValidationError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
