#!/usr/bin/env python3
"""Training script for neural audio effects models with validation support."""

import argparse
import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger

from neural_fx.config import load_config, LatencyConfig
from neural_fx.models import create_model_from_config
from neural_fx.training.lightning_module import NeuralFXModule
from neural_fx.training.callbacks import NeuralFXCheckpoint, ValidationEarlyStopping
from neural_fx.preprocessing.latency import LatencyCalibrator
from neural_fx.preprocessing.validation import DataValidator
from neural_fx.data.dataset import AudioDataset


def run_latency_calibration(config, input_path: str, target_path: str):
    """Run latency calibration if enabled."""
    if config.latency is None or not config.latency.enabled:
        return None

    print(f"Running latency calibration using {config.latency.method} method...")

    # Load audio for calibration
    dataset = AudioDataset(
        input_path=input_path,
        target_path=target_path,
        segment_length=48000 * 5,  # Use 5 seconds for calibration
        sample_rate=config.data.sample_rate,
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
        default="./lightning_logs",
        help="Checkpoint directory",
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
    callbacks = []

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
        dirpath=Path(args.checkpoint_dir) / config.name,
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

    # Setup logger - use CSVLogger to avoid TensorBoard -1 value issues
    logger = CSVLogger(
        save_dir=args.checkpoint_dir,
        name=config.name,
    )

    # Setup trainer kwargs
    trainer_kwargs = {
        "max_epochs": epochs,
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": args.gpus if use_gpu else 1,
        "callbacks": callbacks,
        "gradient_clip_val": 1.0,
        "enable_progress_bar": True,
        "val_check_interval": args.val_check_interval,
        "logger": logger,
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
                sample_rate=config.data.sample_rate,
                latency_calibration=latency_calibration,
                random_segments=False,
            )

            # Generate report
            plot_dir = Path(args.checkpoint_dir) / config.name / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            report = analyzer.generate_report(analysis_dataset, plot_dir)

            print("\nAnalysis Results:")
            print(f"  ESR: {report['esr']:.6f} - {report['esr_comment']}")
            print(f"  Plots saved to: {plot_dir}")

        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    main()
