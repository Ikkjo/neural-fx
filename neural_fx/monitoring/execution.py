"""Private artifact execution and measurement helpers."""

from __future__ import annotations

import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..artifacts import load_model
from ..config import NeuralFXConfig, load_config
from ..inference import run_inference
from ..losses.audio_losses import ESR, MSE, MultiResolutionSTFTLoss
from .schema import MonitoringError, MonitoringManifest

try:
    import resource
except ModuleNotFoundError:  # Windows does not provide this module.
    resource = None


@dataclass(frozen=True)
class LoadedArtifact:
    path: Path
    artifact_type: str
    inference_category: str
    config: NeuralFXConfig
    parameter_count: int
    run: Callable[[Tensor, int | None], Tensor]


def load_artifact(
    artifact_path: str | Path,
    *,
    artifact_type: str | None,
    config_path: str | Path | None,
    device: torch.device,
) -> LoadedArtifact:
    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        raise MonitoringError(f"Artifact not found: {path}", category="artifact")
    resolved_type = artifact_type
    if resolved_type is None:
        if path.suffix != ".ckpt":
            raise MonitoringError(
                "--artifact-type is required unless the artifact ends in .ckpt",
                category="artifact",
            )
        resolved_type = "checkpoint"
    if resolved_type not in {"checkpoint", "torchscript"}:
        raise MonitoringError(
            f"Unsupported artifact type: {resolved_type}", category="artifact"
        )

    try:
        if resolved_type == "checkpoint":
            loaded = load_model(
                checkpoint_path=path,
                config_path=config_path,
                device=device,
            )
            model = loaded.model

            def run(audio: Tensor, chunk_size: int | None) -> Tensor:
                return run_inference(
                    model,
                    audio,
                    chunk_size=chunk_size or audio.shape[-1],
                ).output

            return LoadedArtifact(
                path,
                "checkpoint",
                "native_stateful",
                loaded.config,
                _parameter_count(model),
                run,
            )

        if config_path is None:
            raise ValueError("TorchScript monitoring requires an explicit config")
        config = load_config(config_path)
        if int(getattr(config.model.params, "conditioning_size", 0)):
            raise ValueError("Conditioned TorchScript monitoring is not supported")
        module = torch.jit.load(str(path), map_location=device).eval()

        if config.model.type == "s4":
            params = config.model.params

            def run(audio: Tensor, chunk_size: int | None) -> Tensor:
                del chunk_size
                state = torch.zeros(
                    int(params.num_layers),
                    audio.shape[0],
                    int(params.d_model),
                    int(params.d_state),
                    device=device,
                )
                outputs = []
                for index in range(audio.shape[-1]):
                    output, state = module(audio[..., index : index + 1], state)
                    outputs.append(output)
                return torch.cat(outputs, dim=-1)

            inference_category = "stateful_sample"
        else:

            def run(audio: Tensor, chunk_size: int | None) -> Tensor:
                del chunk_size
                return module(audio)

            inference_category = "stateless_sequence"

        return LoadedArtifact(
            path,
            "torchscript",
            inference_category,
            config,
            _parameter_count(module),
            run,
        )
    except MonitoringError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise MonitoringError(str(exc), category="artifact") from exc


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_run(
    artifact: LoadedArtifact,
    audio: Tensor,
    chunk_size: int | None,
    device: torch.device,
) -> float:
    _synchronize(device)
    start = time.perf_counter_ns()
    with torch.inference_mode():
        artifact.run(audio, chunk_size)
    _synchronize(device)
    return (time.perf_counter_ns() - start) / 1_000_000.0


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(
    measurements_ms: list[float], samples: int, sample_rate: int
) -> dict[str, Any]:
    p50_ms = statistics.median(measurements_ms)
    return {
        "runs": len(measurements_ms),
        "samples_per_run": samples,
        "measurements_ms": measurements_ms,
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": _percentile(measurements_ms, 0.95),
        "min_latency_ms": min(measurements_ms),
        "max_latency_ms": max(measurements_ms),
        "real_time_factor": p50_ms / (samples / sample_rate * 1000.0),
    }


def measure_latency(
    artifact: LoadedArtifact,
    audio: Tensor,
    *,
    chunk_size: int | None,
    manifest: MonitoringManifest,
    device: torch.device,
) -> dict[str, Any]:
    for _ in range(manifest.warmup_runs):
        _time_run(artifact, audio, chunk_size, device)
    measurements = [
        _time_run(artifact, audio, chunk_size, device)
        for _ in range(manifest.measurement_runs)
    ]
    return latency_summary(measurements, audio.shape[-1], manifest.sample_rate)


def quality_metrics(
    prediction: Tensor,
    target: Tensor,
    manifest: MonitoringManifest,
    stft_loss: MultiResolutionSTFTLoss,
) -> dict[str, float]:
    prediction = prediction[..., manifest.burn_in_samples :]
    target = target[..., manifest.burn_in_samples :]
    metrics: dict[str, float] = {}
    if "esr" in manifest.quality_metrics:
        metrics["esr"] = float(
            ESR(prediction, target, pre_emphasis_coeff=manifest.esr_pre_emphasis)
        )
    if "mse" in manifest.quality_metrics:
        metrics["mse"] = float(MSE(prediction, target))
    if "multi_resolution_stft_distance" in manifest.quality_metrics:
        metrics["multi_resolution_stft_distance"] = float(
            stft_loss(prediction, target)
        )
    if any(not math.isfinite(value) for value in metrics.values()):
        raise MonitoringError(
            "Monitoring produced a non-finite quality metric", category="execution"
        )
    return metrics


def peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)
