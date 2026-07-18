#!/usr/bin/env python3
"""Training script for neural audio effects models with validation support."""

import argparse
import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from neural_fx.config import LatencyConfig, LoggingConfig, load_config
from neural_fx.data.dataset import AudioDataset
from neural_fx.models import create_model_from_config
from neural_fx.preprocessing.latency import LatencyCalibrator
from neural_fx.preprocessing.validation import DataValidator
from neural_fx.training.callbacks import NeuralFXCheckpoint, ValidationEarlyStopping
from neural_fx.training.lightning_module import NeuralFXModule


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for command-line options."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _existing_logger_versions(save_dir: Path, experiment_name: str) -> list[int]:
    """Return numeric Lightning logger versions already on disk."""
    experiment_dir = save_dir / experiment_name
    if not experiment_dir.is_dir():
        return []

    versions = []
    for path in experiment_dir.iterdir():
        prefix, separator, suffix = path.name.partition("_")
        if path.is_dir() and prefix == "version" and separator and suffix.isdigit():
            versions.append(int(suffix))
    return versions


def resolve_logger_version(
    save_dir: Path, experiment_name: str, resume_path: str | None
) -> int | str:
    """Choose one shared run version for every configured logger.

    A resumed checkpoint within a ``version_*`` directory reuses that exact
    directory. Legacy checkpoints without version information reuse the latest
    existing version, while a fresh run always receives the next numeric version.
    """
    versions = _existing_logger_versions(save_dir, experiment_name)

    if resume_path:
        for part in reversed(Path(resume_path).parts):
            prefix, separator, suffix = part.partition("_")
            if prefix == "version" and separator and suffix:
                return int(suffix) if suffix.isdigit() else suffix
        return max(versions) if versions else 0

    return max(versions, default=-1) + 1


def build_loggers(
    logging_config: LoggingConfig, experiment_name: str, version: int | str
) -> list[CSVLogger | TensorBoardLogger]:
    """Build all requested loggers with a shared experiment directory."""
    loggers: list[CSVLogger | TensorBoardLogger] = []
    common_kwargs = {
        "save_dir": logging_config.save_dir,
        "name": experiment_name,
        "version": version,
    }
    for backend in logging_config.backends:
        if backend == "csv":
            loggers.append(CSVLogger(**common_kwargs))
        elif backend == "tensorboard":
            loggers.append(
                TensorBoardLogger(**common_kwargs, default_hp_metric=False)
            )
    return loggers


def run_latency_calibration(config, input_path: str, target_path: str):
    """Run latency calibration if enabled."""
    if config.latency is None or not config.latency.enabled:
        return None

    print(f"Running latency calibration using {config.latency.method} method...")

    # Load audio for calibration
    dataset = AudioDataset(
        input_path=input_path,
        target_path=target_path,
        segment_length=round(
            config.sample_rate * config.latency.calibration_duration_seconds
        ),
        sample_rate=config.sample_rate,
        normalize=True,
    )

    # Get first segment
    input_audio, target_audio = dataset[0]

    calibrator = LatencyCalibrator()
    calibration = calibrator.calibrate(
        input_audio,
        target_audio,
        method=config.latency.method,
        max_delay=config.latency.max_delay,
        manual_delay=config.latency.manual_delay,
    )

    print(f"  Delay: {calibration.delay_samples} samples")
    print(f"  Correlation score: {calibration.correlation_score:.4f}")
    print(f"  Method: {calibration.method}")

    return calibration


def run_data_validation(config, input_path: str, target_path: str, ignore_checks: bool):
    """Run data validation if enabled."""
    if config.validation is None or not config.validation.enabled:
        return True

    print("Running data validation...")

    check_replicability = (
        config.validation.check_replicability
        if config.validation.check_replicability is not None
        else False
    )
    validator = DataValidator(
        check_clipping=config.validation.check_clipping,
        check_dc_offset=config.validation.check_dc_offset,
        check_replicability=check_replicability,
    )

    report = validator.validate(input_path, target_path)

    print(report.get_summary())

    if not report.passed and not ignore_checks:
        print("\nValidation failed. Use --ignore_checks to bypass.")
        return False

    if report.warnings and not ignore_checks and not config.validation.ignore_warnings:
        print(
            "\nWarnings detected. Use --ignore_checks to bypass or set ignore_warnings in config."
        )
        # Don't fail on warnings alone unless explicitly requested

    return True


def main():
    parser = argparse.ArgumentParser(description="Train neural audio effects model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--max_epochs", type=int, default=None, help="Override max epochs"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override the config directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Validation check interval (fraction of epoch)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU training even if GPU is available"
    )
    # New CLI arguments
    parser.add_argument(
        "--latency_method",
        type=str,
        choices=["xcorr", "manual"],
        default=None,
        help="Latency calibration method (overrides config)",
    )
    parser.add_argument(
        "--latency_manual",
        type=int,
        default=None,
        help="Manual latency delay in samples (overrides config)",
    )
    parser.add_argument(
        "--ignore_checks",
        action="store_true",
        help="Ignore data validation warnings",
    )
    parser.add_argument(
        "--esr_threshold",
        type=float,
        default=None,
        help="ESR threshold for early stopping (overrides config)",
    )
    #TODO: check if all of the arguments actually exist in the config and
    # are properly overridden, especially the nested ones like latency.method and latency.manual_delay.
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs (overrides config)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after training",
    )
    parser.add_argument(
        "--loggers",
        nargs="+",
        choices=["csv", "tensorboard"],
        default=None,
        help="Override logging backends (for example: csv tensorboard)",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=_positive_int,
        default=None,
        help="Override the training metric logging interval",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.latency_method:
        if config.latency is None:
            config.latency = LatencyConfig()
        config.latency.method = args.latency_method

    if args.latency_manual is not None:
        if config.latency is None:
            config.latency = LatencyConfig()
        config.latency.manual_delay = args.latency_manual
        config.latency.method = "manual"

    if args.loggers is not None:
        config.logging.backends = args.loggers
    if args.log_every_n_steps is not None:
        config.logging.log_every_n_steps = args.log_every_n_steps

    output_dir = Path(args.checkpoint_dir or config.logging.save_dir)
    config.logging.save_dir = str(output_dir)
    logger_version = resolve_logger_version(output_dir, config.name, args.resume)
    run_dir = output_dir / config.name / f"version_{logger_version}"

    L.seed_everything(config.training.seed, workers=True)

    # Run data validation
    input_path = config.data.train.input
    target_path = config.data.train.target

    if not run_data_validation(config, input_path, target_path, args.ignore_checks):
        sys.exit(1)

    # Run latency calibration
    latency_calibration = run_latency_calibration(config, input_path, target_path)

    # Create model using registry
    model = create_model_from_config(config.model)

    # Create Lightning module
    module = NeuralFXModule(model, config)

    epochs = args.max_epochs if args.max_epochs else config.training.epochs

    # Setup callbacks
    callbacks = [LearningRateMonitor(logging_interval="epoch")]

    # Enhanced checkpoint callback
    latency_cal_dict = None
    if latency_calibration is not None:
        latency_cal_dict = {
            "delay_samples": latency_calibration.delay_samples,
            "method": latency_calibration.method,
            "correlation_score": latency_calibration.correlation_score,
        }

    checkpoint_callback = NeuralFXCheckpoint(
        config=config,
        input_file=input_path,
        target_file=target_path,
        latency_calibration=latency_cal_dict,
        dirpath=run_dir / "checkpoints",
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss" if config.data.val else "train_loss",
        mode="min",
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    patience = args.patience if args.patience else 10
    early_stop_callback = ValidationEarlyStopping(
        monitor="val_loss" if config.data.val else "train_loss",
        min_delta=0.0,
        patience=patience,
        mode="min",
    )
    callbacks.append(early_stop_callback)

    # Determine device
    use_gpu = not args.cpu and args.gpus > 0 and torch.cuda.is_available()

    # Keep all backends in the same version directory. TensorBoard creates a new
    # event file on resume, preserving the earlier event history.
    loggers = build_loggers(config.logging, config.name, logger_version)

    # Setup trainer kwargs
    trainer_kwargs = {
        "max_epochs": epochs,
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": args.gpus if use_gpu else 1,
        "callbacks": callbacks,
        "gradient_clip_val": 1.0,
        "enable_progress_bar": True,
        "val_check_interval": args.val_check_interval,
        "logger": loggers,
        "log_every_n_steps": config.logging.log_every_n_steps,
    }

    # Add validation if available
    has_validation = config.data.val is not None
    if has_validation:
        trainer_kwargs["check_val_every_n_epoch"] = 1
    else:
        # Disable validation sanity check when no validation data
        trainer_kwargs["num_sanity_val_steps"] = 0
        trainer_kwargs["limit_val_batches"] = 0

    # Setup resume from checkpoint if specified
    resume_path = args.resume
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")

    trainer = L.Trainer(**trainer_kwargs)

    # Train
    if resume_path:
        trainer.fit(module, ckpt_path=resume_path)
    else:
        trainer.fit(module)

    print(f"Training complete. Best checkpoint: {checkpoint_callback.best_model_path}")

    # Generate plots if requested
    if args.plot:
        print("Generating analysis plots...")
        try:
            from neural_fx.analysis.plotting import TrainingAnalyzer

            # Load best model
            if checkpoint_callback.best_model_path:
                checkpoint = torch.load(
                    checkpoint_callback.best_model_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if "state_dict" in checkpoint:
                    module.load_state_dict(checkpoint["state_dict"])

            analyzer = TrainingAnalyzer(model, config)

            # Create dataset for analysis
            analysis_dataset = AudioDataset(
                input_path=input_path,
                target_path=target_path,
                segment_length=config.training.segment_length,
                sample_rate=config.sample_rate,
                latency_calibration=latency_calibration,
                random_segments=False,
            )

            # Generate report
            plot_dir = run_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            report = analyzer.generate_report(analysis_dataset, plot_dir)

            print("\nAnalysis Results:")
            print(f"  ESR: {report['esr']:.6f} - {report['esr_comment']}")
            print(f"  Plots saved to: {plot_dir}")

        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    main()
