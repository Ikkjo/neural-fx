import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from ..artifacts import load_model
from ..config import NeuralFXConfig
from ..data.dataset import AudioDataset
from ..models import BaseNeuralFXModel, create_model_from_config
from ..preprocessing.latency import LatencyCalibration, LatencyCalibrator
from ..preprocessing.validation import DataValidator
from .callbacks import NeuralFXCheckpoint, ValidationEarlyStopping
from .lightning_module import NeuralFXModule
from .logging import create_training_loggers


@dataclass(frozen=True)
class TrainingRun:
    config: NeuralFXConfig
    checkpoint_dir: Path
    gpus: int
    force_cpu: bool
    resume_path: Path | None
    ignore_checks: bool
    patience: int
    min_delta: float
    min_delta_mode: str
    plot: bool
    val_check_interval: float
    log_every_n_steps: int
    enable_progress_bar: bool = True

    @classmethod
    def resolve(
        cls,
        config: NeuralFXConfig,
        *,
        checkpoint_dir: str | Path = "./lightning_logs",
        gpus: int = 1,
        force_cpu: bool = False,
        resume_path: str | Path | None = None,
        ignore_checks: bool = False,
        patience: int | None = None,
        min_delta: float | None = None,
        min_delta_mode: str | None = None,
        plot: bool = False,
        val_check_interval: float = 1.0,
        log_every_n_steps: int = 50,
        enable_progress_bar: bool = True,
        max_epochs: int | None = None,
        compile: bool | None = None,
        num_workers: int | None = None,
        latency_method: str | None = None,
        latency_manual: int | None = None,
    ) -> "TrainingRun":
        epochs = max_epochs if max_epochs is not None else config.training.epochs
        manual_delay = (
            latency_manual
            if latency_manual is not None
            else config.latency.manual_delay
        )
        latency_method = (
            "manual"
            if latency_manual is not None
            else latency_method or config.latency.method
        )
        resolved_config = replace(
            config,
            training=replace(
                config.training,
                epochs=epochs,
                compile=config.training.compile if compile is None else compile,
                num_workers=(
                    config.training.num_workers
                    if num_workers is None
                    else num_workers
                ),
            ),
            latency=replace(
                config.latency,
                method=latency_method,
                manual_delay=manual_delay,
            ),
        )
        return cls(
            config=resolved_config,
            checkpoint_dir=Path(checkpoint_dir),
            gpus=gpus,
            force_cpu=force_cpu,
            resume_path=Path(resume_path) if resume_path is not None else None,
            ignore_checks=ignore_checks,
            patience=(
                patience
                if patience is not None
                else config.training.early_stopping_patience
            ),
            min_delta=(
                min_delta
                if min_delta is not None
                else config.training.early_stopping_min_delta
            ),
            min_delta_mode=(
                min_delta_mode
                if min_delta_mode is not None
                else config.training.early_stopping_min_delta_mode
            ),
            plot=plot,
            val_check_interval=val_check_interval,
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=enable_progress_bar,
        )


@dataclass(frozen=True)
class TrainingResult:
    best_checkpoint: Path | None
    terminal_checkpoint: Path


class TrainingDataValidationError(RuntimeError):
    """Training cannot start because required data validation failed."""


def _validate_compile_request(run: TrainingRun) -> None:
    if not run.config.training.compile:
        return
    if run.gpus > 1:
        raise ValueError("Compiled training supports at most one GPU")
    tbptt = run.config.training.tbptt
    if tbptt is not None and tbptt.enabled:
        raise ValueError("Compiled training does not support TBPTT")


def _create_training_model(config: NeuralFXConfig) -> BaseNeuralFXModel:
    model = create_model_from_config(config.model)
    if config.training.compile:
        model.compile()
    return model


def run_latency_calibration(
    config: NeuralFXConfig,
    input_path: str,
    target_path: str,
) -> LatencyCalibration | None:
    if config.latency.calibration_duration_seconds == 0:
        return None

    print(f"Running latency calibration using {config.latency.method} method...")
    dataset = AudioDataset(
        input_path=input_path,
        target_path=target_path,
        segment_length=round(
            config.sample_rate * config.latency.calibration_duration_seconds
        ),
        sample_rate=config.sample_rate,
        normalize=config.data.normalize,
    )
    input_audio, target_audio = dataset[0]
    calibration = LatencyCalibrator().calibrate(
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


def run_data_validation(
    config: NeuralFXConfig,
    input_path: str,
    target_path: str,
    ignore_checks: bool,
) -> bool:
    if config.validation is None or not config.validation.enabled:
        return True

    print("Running data validation...")
    validator = DataValidator(
        check_clipping=config.validation.check_clipping,
        check_dc_offset=config.validation.check_dc_offset,
    )
    report = validator.validate(input_path, target_path)
    print(report.get_summary())
    if not report.passed and not ignore_checks:
        print("\nValidation failed. Use --ignore_checks to bypass.")
        return False
    if report.warnings and not ignore_checks and not config.validation.ignore_warnings:
        print(
            "\nWarnings detected. Use --ignore_checks to bypass or set "
            "ignore_warnings in config."
        )
    return True


def create_trainer(
    config: NeuralFXConfig,
    trainer_kwargs: dict[str, object],
) -> L.Trainer:
    return L.Trainer(deterministic=config.training.deterministic, **trainer_kwargs)


def publish_best_checkpoint(
    best_model_path: str,
    checkpoint_dir: str | Path,
    config_name: str,
) -> Path | None:
    if not best_model_path:
        return None
    source = Path(best_model_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {source}")
    destination = (Path(checkpoint_dir) / config_name / "best.ckpt").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination:
        shutil.copy2(source, destination)
    return destination


def _configure_determinism(config: NeuralFXConfig) -> None:
    if config.training.deterministic:
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in (None, ":4096:8"):
            raise ValueError(
                "Deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8; "
                f"found {workspace_config!r}"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    L.seed_everything(config.training.seed, workers=True)


def _training_callbacks(
    run: TrainingRun,
    latency_calibration: LatencyCalibration | None,
) -> tuple[list[Callback], NeuralFXCheckpoint]:
    config = run.config
    monitor = "val_loss" if config.data.val else "train_loss"
    checkpoint = NeuralFXCheckpoint(
        config=config,
        input_file=config.data.train.input,
        target_file=config.data.train.target,
        latency_calibration=(
            asdict(latency_calibration) if latency_calibration is not None else None
        ),
        dirpath=run.checkpoint_dir / config.name,
        filename=f"{{epoch:02d}}-{{{monitor}:.4f}}",
        save_top_k=3,
        monitor=monitor,
        mode="min",
        save_last=True,
    )
    callbacks: list[Callback] = [checkpoint]
    if config.training.early_stopping:
        callbacks.append(
            ValidationEarlyStopping(
                monitor=monitor,
                min_delta=run.min_delta,
                min_delta_mode=run.min_delta_mode,
                patience=run.patience,
                mode="min",
            )
        )
    return callbacks, checkpoint


def _trainer_kwargs(run: TrainingRun, callbacks: list[Callback]) -> dict[str, object]:
    config = run.config
    use_gpu = not run.force_cpu and run.gpus > 0 and torch.cuda.is_available()
    kwargs: dict[str, object] = {
        "max_epochs": config.training.epochs,
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": run.gpus if use_gpu else 1,
        "callbacks": callbacks,
        "gradient_clip_val": 1.0,
        "enable_progress_bar": run.enable_progress_bar,
        "val_check_interval": run.val_check_interval,
        "logger": create_training_loggers(run.checkpoint_dir, config.name),
        "log_every_n_steps": run.log_every_n_steps,
    }
    if config.data.val is not None:
        kwargs["check_val_every_n_epoch"] = 1
    else:
        kwargs["num_sanity_val_steps"] = 0
        kwargs["limit_val_batches"] = 0
    return kwargs


def _generate_plots(
    run: TrainingRun,
    model: torch.nn.Module,
    best_model_path: str,
    latency_calibration: LatencyCalibration | None,
) -> None:
    print("Generating analysis plots...")
    try:
        from ..analysis.plotting import generate_analysis_report

        if best_model_path:
            model = load_model(checkpoint_path=best_model_path).model
        config = run.config
        dataset = AudioDataset(
            input_path=config.data.train.input,
            target_path=config.data.train.target,
            segment_length=config.training.segment_length,
            sample_rate=config.sample_rate,
            normalize=config.data.normalize,
            latency_calibration=latency_calibration,
            random_segments=False,
        )
        plot_dir = run.checkpoint_dir / config.name / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        report = generate_analysis_report(model, dataset, plot_dir, config=config)
        print("\nAnalysis Results:")
        print(f"  ESR: {report['esr']:.6f} - {report['esr_comment']}")
        print(f"  Plots saved to: {plot_dir}")
    except Exception as error:
        # A completed training run remains valid when optional report generation fails.
        print(f"Warning: Could not generate plots: {error}")


def run_training(run: TrainingRun) -> TrainingResult:
    config = run.config
    _validate_compile_request(run)
    execution = "compiled" if config.training.compile else "eager"
    print(f"Training execution: {execution}")
    _configure_determinism(config)
    input_path = config.data.train.input
    target_path = config.data.train.target
    if not run_data_validation(config, input_path, target_path, run.ignore_checks):
        raise TrainingDataValidationError("Training data validation failed")

    train_latency = run_latency_calibration(config, input_path, target_path)
    val_latency = (
        run_latency_calibration(config, config.data.val.input, config.data.val.target)
        if config.data.val is not None
        else None
    )
    model = _create_training_model(config)
    module = NeuralFXModule(
        model,
        config,
        train_latency=train_latency,
        val_latency=val_latency,
    )
    callbacks, checkpoint = _training_callbacks(run, train_latency)
    trainer = create_trainer(config, _trainer_kwargs(run, callbacks))
    if run.resume_path is not None:
        print(f"Resuming from checkpoint: {run.resume_path}")
        trainer.fit(module, ckpt_path=str(run.resume_path))
    else:
        trainer.fit(module)

    terminal_checkpoint = checkpoint.save_terminal_checkpoint(trainer)
    best_checkpoint = publish_best_checkpoint(
        checkpoint.best_model_path,
        run.checkpoint_dir,
        config.name,
    )
    print(f"Training complete. Best checkpoint: {best_checkpoint}")
    print(f"Terminal checkpoint: {terminal_checkpoint}")
    if run.plot:
        _generate_plots(run, model, checkpoint.best_model_path, train_latency)
    return TrainingResult(best_checkpoint, terminal_checkpoint)
