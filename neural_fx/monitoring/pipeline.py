"""Execute one fixed-suite offline monitoring run."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch import Tensor

from ..data.audio import load_audio_pair
from ..losses.audio_losses import MultiResolutionSTFTLoss
from .execution import (
    latency_summary,
    load_artifact,
    measure_latency,
    peak_rss_bytes,
    quality_metrics,
)
from .schema import (
    MonitoringCase,
    MonitoringCaseResult,
    MonitoringError,
    MonitoringManifest,
    MonitoringReport,
    ValidationCheck,
    fingerprint_monitoring_suite,
    load_monitoring_manifest,
    monitoring_case_hashes,
    sha256_file,
)


def _check(
    case_id: str,
    name: str,
    passed: bool,
    message: str,
    *,
    severity: str = "error",
    value: float | int | str | None = None,
) -> ValidationCheck:
    return ValidationCheck(case_id, name, passed, severity, message, value)


def _require(check: ValidationCheck) -> None:
    if not check.passed and check.severity == "error":
        raise MonitoringError(
            f"Case '{check.case_id}' failed {check.name}: {check.message}",
            category="validation",
            diagnostics=(check,),
        )


def _load_case(
    case: MonitoringCase, manifest: MonitoringManifest
) -> tuple[Tensor, Tensor, list[ValidationCheck]]:
    try:
        pair = load_audio_pair(
            case.input_path,
            case.target_path,
            sample_rate=manifest.sample_rate,
            strict=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise MonitoringError(
            f"Case '{case.case_id}' audio is invalid: {exc}",
            category="validation",
        ) from exc

    checks = [
        _check(case.case_id, "files_exist", True, "Input and target files exist"),
        _check(
            case.case_id,
            "sample_rate",
            True,
            f"Input and target use {manifest.sample_rate} Hz",
            value=manifest.sample_rate,
        ),
    ]
    channels_match = (
        pair.input_audio.shape[0]
        == pair.target_audio.shape[0]
        == manifest.channels
    )
    channel_check = _check(
        case.case_id,
        "channels",
        channels_match,
        (
            f"Expected {manifest.channels} channels; "
            f"input={pair.input_audio.shape[0]}, target={pair.target_audio.shape[0]}"
        ),
    )
    checks.append(channel_check)
    _require(channel_check)

    required_length = case.start_sample + case.num_samples
    available = min(pair.input_audio.shape[-1], pair.target_audio.shape[-1])
    length_check = _check(
        case.case_id,
        "usable_length",
        required_length <= available,
        f"Need {required_length} samples and found {available}",
        value=available,
    )
    checks.append(length_check)
    _require(length_check)

    start = case.start_sample
    end = start + case.num_samples
    input_audio = pair.input_audio[..., start:end].to(torch.float32)
    target_audio = pair.target_audio[..., start:end].to(torch.float32)
    for role, audio in (("input", input_audio), ("target", target_audio)):
        finite = bool(torch.isfinite(audio).all())
        finite_check = _check(
            case.case_id,
            f"{role}_finite",
            finite,
            f"{role} audio contains only finite values"
            if finite
            else f"{role} audio contains NaN or Inf",
        )
        checks.append(finite_check)
        _require(finite_check)

        peak = float(audio.abs().max())
        amplitude_check = _check(
            case.case_id,
            f"{role}_amplitude",
            peak <= manifest.max_abs,
            f"{role} peak absolute amplitude is {peak:.6f}",
            value=peak,
        )
        checks.append(amplitude_check)
        _require(amplitude_check)

        if role == "target":
            full_scale_samples = int((audio.abs() >= 1.0).sum())
            full_scale_check = _check(
                case.case_id,
                "target_full_scale",
                full_scale_samples == 0 or manifest.allow_target_full_scale,
                f"target has {full_scale_samples} samples at or above 1.0",
                value=full_scale_samples,
            )
            checks.append(full_scale_check)
            _require(full_scale_check)

        clipped = int((audio.abs() >= manifest.clipping_threshold).sum())
        checks.append(
            _check(
                case.case_id,
                f"{role}_clipping",
                clipped == 0,
                (
                    f"{role} has {clipped} samples at or above "
                    f"{manifest.clipping_threshold}"
                ),
                severity="warning",
                value=clipped,
            )
        )
    return input_audio, target_audio, checks


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def monitor_artifact(
    manifest_path: str | Path,
    artifact_path: str | Path,
    *,
    artifact_type: str | None = None,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> MonitoringReport:
    """Validate a fixed suite and measure one checkpoint or TorchScript artifact."""
    manifest = load_monitoring_manifest(manifest_path)
    case_hashes = monitoring_case_hashes(manifest)
    suite_fingerprint = fingerprint_monitoring_suite(manifest, case_hashes)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise MonitoringError("CUDA is not available", category="execution")
    artifact = load_artifact(
        artifact_path,
        artifact_type=artifact_type,
        config_path=config_path,
        device=resolved_device,
    )
    if artifact.config.sample_rate != manifest.sample_rate:
        raise MonitoringError(
            "Artifact and suite sample rates do not match", category="artifact"
        )
    if (
        artifact.config.model.input_size != manifest.channels
        or artifact.config.model.output_size != manifest.channels
    ):
        raise MonitoringError(
            "Artifact and suite channel counts do not match", category="artifact"
        )

    loaded_cases = []
    validation_checks: list[ValidationCheck] = []
    for case in manifest.cases:
        input_audio, target_audio, checks = _load_case(case, manifest)
        loaded_cases.append((case, input_audio, target_audio))
        validation_checks.extend(checks)

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    stft_loss = MultiResolutionSTFTLoss().to(resolved_device)
    case_results = []
    aggregate_measurements: list[float] = []
    try:
        for (case, input_audio, target_audio), hashes in zip(
            loaded_cases, case_hashes
        ):
            input_batch = input_audio.unsqueeze(0).to(resolved_device)
            target_batch = target_audio.unsqueeze(0).to(resolved_device)
            with torch.inference_mode():
                prediction = artifact.run(
                    input_batch, manifest.inference_chunk_size
                )
            if prediction.shape != target_batch.shape:
                raise MonitoringError(
                    f"Case '{case.case_id}' output shape does not match target",
                    category="execution",
                )
            if not bool(torch.isfinite(prediction).all()):
                raise MonitoringError(
                    f"Case '{case.case_id}' prediction contains NaN or Inf",
                    category="execution",
                )
            metrics = quality_metrics(
                prediction, target_batch, manifest, stft_loss
            )
            full_latency = measure_latency(
                artifact,
                input_batch,
                chunk_size=manifest.inference_chunk_size,
                manifest=manifest,
                device=resolved_device,
            )
            latency = {"full": full_latency}
            aggregate_measurements.extend(full_latency["measurements_ms"])
            for block_size in manifest.latency_block_sizes:
                latency[f"block_{block_size}"] = measure_latency(
                    artifact,
                    input_batch[..., :block_size],
                    chunk_size=None,
                    manifest=manifest,
                    device=resolved_device,
                )
            case_results.append(
                MonitoringCaseResult(
                    case_id=case.case_id,
                    input_sha256=hashes["input_sha256"],
                    target_sha256=hashes["target_sha256"],
                    evaluated_samples=manifest.segment_length,
                    metric_samples=(
                        manifest.segment_length - manifest.burn_in_samples
                    ),
                    metrics=metrics,
                    latency=latency,
                )
            )
    except MonitoringError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise MonitoringError(str(exc), category="execution") from exc

    aggregate_quality = {
        metric: statistics.fmean(case.metrics[metric] for case in case_results)
        for metric in manifest.quality_metrics
    }
    full_latency = latency_summary(
        aggregate_measurements, manifest.segment_length, manifest.sample_rate
    )
    process_peak = peak_rss_bytes()
    if resolved_device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(resolved_device))
        memory_kind = "cuda_peak_allocated_bytes"
    elif process_peak is None:
        peak_memory = None
        memory_kind = "unavailable"
    else:
        peak_memory = process_peak
        memory_kind = "process_peak_rss_bytes"
    artifact_size = artifact.path.stat().st_size
    comparison_metrics = {
        **aggregate_quality,
        "p50_latency_ms": float(full_latency["p50_latency_ms"]),
        "p95_latency_ms": float(full_latency["p95_latency_ms"]),
        "real_time_factor": float(full_latency["real_time_factor"]),
        "peak_memory_bytes": (
            float(peak_memory) if peak_memory is not None else None
        ),
        "artifact_size_bytes": float(artifact_size),
    }
    device_name = (
        torch.cuda.get_device_name(resolved_device)
        if resolved_device.type == "cuda"
        else platform.processor() or platform.machine()
    )
    warning_count = sum(
        not check.passed and check.severity == "warning"
        for check in validation_checks
    )
    return MonitoringReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        suite={
            "id": manifest.suite_id,
            "fingerprint": suite_fingerprint,
            "manifest_path": str(manifest.manifest_path),
            "manifest_sha256": sha256_file(manifest.manifest_path),
            "cases": case_hashes,
            "validation_passed": True,
            "validation_warnings": warning_count,
        },
        artifact={
            "path": str(artifact.path),
            "type": artifact.artifact_type,
            "inference_category": artifact.inference_category,
            "sha256": sha256_file(artifact.path),
            "size_bytes": artifact_size,
            "config_path": str(Path(config_path).resolve()) if config_path else None,
            "config_sha256": _sha256_json(asdict(artifact.config)),
            "model_name": artifact.config.name,
            "model_type": artifact.config.model.type,
            "trainable_parameters": artifact.parameter_count,
        },
        runtime={
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchaudio_version": torchaudio.__version__,
            "platform": platform.platform(),
            "device": str(resolved_device),
            "device_class": resolved_device.type,
            "device_name": device_name,
            "dtype": "float32",
        },
        workload={
            **manifest.settings_dict(),
            "case_ids": [case.case_id for case in manifest.cases],
        },
        validation=tuple(validation_checks),
        cases=tuple(case_results),
        aggregate={
            "metrics": comparison_metrics,
            "quality": aggregate_quality,
            "full_latency": full_latency,
            "memory": {
                "peak_memory_bytes": peak_memory,
                "kind": memory_kind,
                "process_peak_rss_bytes": process_peak,
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(resolved_device))
                    if resolved_device.type == "cuda"
                    else None
                ),
                "cuda_peak_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(resolved_device))
                    if resolved_device.type == "cuda"
                    else None
                ),
            },
        },
    )
