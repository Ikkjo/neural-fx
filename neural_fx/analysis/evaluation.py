"""Controlled checkpoint evaluation and cross-model comparison."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import yaml

from ..config import NeuralFXConfig
from ..data.audio import load_audio_pair
from ..inference import run_inference
from ..losses.audio_losses import ESR, MultiResolutionSTFTLoss
from ..preprocessing.latency import LatencyCalibration
from .benchmarking import load_benchmark_result, load_model_for_evaluation

EVALUATION_SCHEMA_VERSION = "1.0"
COMPARISON_SCHEMA_VERSION = "1.0"
DEFAULT_INFERENCE_CHUNK_SIZE = 65_536


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    """Load a YAML/JSON experiment manifest and resolve its file inputs."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("Experiment manifest must contain a mapping")
    if data.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {data.get('schema_version')}")
    for key in ("experiment_id", "run_kind", "model", "dataset", "training"):
        if key not in data:
            raise ValueError(f"Experiment manifest is missing '{key}'")
    if data["run_kind"] not in {"smoke", "final"}:
        raise ValueError("run_kind must be 'smoke' or 'final'")
    if "checkpoint" not in data["model"]:
        raise ValueError("Experiment model is missing 'checkpoint'")
    data["model"]["checkpoint"] = str(_resolve_path(data["model"]["checkpoint"], path))
    if "config" in data["model"]:
        data["model"]["config"] = str(_resolve_path(data["model"]["config"], path))
    for key in ("input_audio", "target_audio", "split"):
        if key not in data["dataset"]:
            raise ValueError(f"Experiment dataset is missing '{key}'")
    for key in ("input_audio", "target_audio"):
        data["dataset"][key] = str(_resolve_path(data["dataset"][key], path))
    if "seed" not in data["training"]:
        raise ValueError("Experiment training section is missing 'seed'")
    benchmark_path = data["model"].get("benchmark_result")
    if benchmark_path is not None:
        data["model"]["benchmark_result"] = str(_resolve_path(benchmark_path, path))
    data["manifest_path"] = str(path)
    return data


def _prepare_evaluation_audio(
    manifest: dict[str, Any], sample_rate: int
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = manifest["dataset"]
    delay_samples = int(dataset.get("latency_samples", 0))
    normalization = dataset.get("normalization", "paired_peak")
    if normalization not in {"paired_peak", "none"}:
        raise ValueError("normalization must be 'paired_peak' or 'none'")
    audio_pair = load_audio_pair(
        input_path=dataset["input_audio"],
        target_path=dataset["target_audio"],
        sample_rate=sample_rate,
        normalize=normalization == "paired_peak",
        latency_calibration=LatencyCalibration(
            delay_samples=delay_samples,
            method="manual",
            correlation_score=1.0,
        ),
    )
    input_audio = audio_pair.input_audio
    target_audio = audio_pair.target_audio

    start = int(dataset.get("start_sample", 0))
    available = min(input_audio.shape[-1], target_audio.shape[-1])
    num_samples = int(dataset.get("num_samples", available - start))
    if start < 0 or num_samples <= 0 or start + num_samples > available:
        raise ValueError(
            f"Requested samples [{start}, {start + num_samples}) exceed aligned audio length {available}"
        )
    input_audio = input_audio[..., start : start + num_samples]
    target_audio = target_audio[..., start : start + num_samples]

    return input_audio, target_audio


def _safe_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = prediction.detach().cpu().numpy().ravel()
    target_np = target.detach().cpu().numpy().ravel()
    if np.std(pred_np) == 0 or np.std(target_np) == 0:
        return 0.0
    return float(np.corrcoef(pred_np, target_np)[0, 1])


def run_chunked_inference(
    model: torch.nn.Module,
    audio: torch.Tensor,
    chunk_size: int = DEFAULT_INFERENCE_CHUNK_SIZE,
) -> torch.Tensor:
    """Run stateful inference, resetting once and carrying state across chunks."""
    return run_inference(model, audio, chunk_size=chunk_size).output


def _stft_window_starts(
    num_samples: int,
    sample_rate: int,
    window_seconds: float = 3.0,
    max_windows: int = 10,
) -> tuple[list[int], int]:
    """Select fixed, uniformly spaced, non-overlapping STFT windows."""
    window_samples = min(num_samples, round(window_seconds * sample_rate))
    window_count = min(max_windows, max(1, num_samples // window_samples))
    if window_count == 1:
        return [0], window_samples
    starts = np.linspace(
        0,
        num_samples - window_samples,
        num=window_count,
        dtype=np.int64,
    ).tolist()
    return [int(start) for start in starts], window_samples


@dataclass(frozen=True)
class _EvaluationSignals:
    input_audio: torch.Tensor
    target_audio: torch.Tensor
    prediction: torch.Tensor
    chunk_size: int

    @property
    def num_samples(self) -> int:
        return self.prediction.shape[-1]


def _execute_evaluation(
    model: torch.nn.Module,
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    *,
    device: str | torch.device,
    chunk_size: int,
) -> _EvaluationSignals:
    input_batch = input_audio.unsqueeze(0).to(device)
    prediction = run_chunked_inference(model, input_batch, chunk_size).detach().cpu()
    target_batch = target_audio.unsqueeze(0)
    input_batch = input_batch.cpu()
    common_length = min(
        input_batch.shape[-1],
        target_batch.shape[-1],
        prediction.shape[-1],
    )
    if common_length < 2048:
        raise ValueError("Evaluation segment must contain at least 2048 samples")
    return _EvaluationSignals(
        input_audio=input_batch[..., :common_length],
        target_audio=target_batch[..., :common_length],
        prediction=prediction[..., :common_length],
        chunk_size=chunk_size,
    )


def _calculate_evaluation_metrics(
    signals: _EvaluationSignals,
    manifest: dict[str, Any],
    config: NeuralFXConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    mask_first = int(
        manifest.get(
            "burn_in_samples",
            manifest["dataset"].get("metric_mask_first", config.loss.mask_first),
        )
    )
    if mask_first < 0 or mask_first >= signals.num_samples:
        raise ValueError(
            f"loss.mask_first ({mask_first}) must be smaller than the evaluation "
            f"segment ({signals.num_samples})"
        )

    prediction = signals.prediction[..., mask_first:]
    target = signals.target_audio[..., mask_first:]
    if prediction.shape[-1] < 2048:
        raise ValueError(
            "Evaluation segment after loss masking must contain 2048 samples"
        )

    pre_emphasis_coeff = manifest.get("esr_pre_emphasis")
    if pre_emphasis_coeff is not None:
        pre_emphasis_coeff = float(pre_emphasis_coeff)
    esr_mode = manifest.get("esr_mode", "legacy")
    stft_starts, stft_window_samples = _stft_window_starts(
        prediction.shape[-1],
        config.sample_rate,
    )
    stft_loss = MultiResolutionSTFTLoss()
    stft_values = [
        stft_loss(
            prediction[..., start : start + stft_window_samples],
            target[..., start : start + stft_window_samples],
        ).item()
        for start in stft_starts
    ]
    metrics = {
        "esr": ESR(
            prediction,
            target,
            pre_emphasis_coeff=pre_emphasis_coeff,
            mode=esr_mode,
        ).item(),
        "mse": torch.mean((prediction - target) ** 2).item(),
        "correlation": _safe_correlation(prediction, target),
        "multi_resolution_stft_distance": statistics.fmean(stft_values),
    }
    recipe = {
        "mask_first": mask_first,
        "esr_mode": esr_mode,
        "esr_pre_emphasis": pre_emphasis_coeff,
        "configured_loss_mask_first": config.loss.mask_first,
        "metric_samples": prediction.shape[-1],
        "stft_window_starts": stft_starts,
        "stft_window_samples": stft_window_samples,
        "stft_window_values": stft_values,
    }
    return metrics, recipe


def _write_listening_samples(
    signals: _EvaluationSignals,
    output_dir: Path,
    sample_rate: int,
) -> dict[str, str]:
    artifacts = {}
    for name, audio in (
        ("input", signals.input_audio),
        ("target", signals.target_audio),
        ("prediction", signals.prediction),
    ):
        path = output_dir / f"{name}.wav"
        torchaudio.save(str(path), audio.squeeze(0).clamp(-1.0, 1.0), sample_rate)
        artifacts[f"{name}_audio"] = str(path.resolve())
    return artifacts


def _load_evaluation_performance(
    model_spec: dict[str, Any],
    trainable_parameters: int,
) -> dict[str, Any] | None:
    benchmark_path = model_spec.get("benchmark_result")
    if benchmark_path is None:
        return None

    benchmark = load_benchmark_result(benchmark_path)
    if benchmark["model"]["trainable_parameters"] != trainable_parameters:
        raise ValueError("Benchmark parameter count does not match the evaluated model")
    return {
        "result_path": str(Path(benchmark_path).resolve()),
        "runtime": benchmark["runtime"],
        "offline": benchmark["offline"],
        "blocks": benchmark["blocks"],
        "memory": benchmark["memory"],
    }


def evaluate_experiment(
    manifest: dict[str, Any],
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    inference_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint and write the complete schema-1.0 artifact set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = manifest["model"]
    model, config = load_model_for_evaluation(
        model_spec.get("config"),
        model_spec["checkpoint"],
        device=device,
    )
    input_audio, target_audio = _prepare_evaluation_audio(manifest, config.sample_rate)
    chunk_size = int(
        inference_chunk_size
        if inference_chunk_size is not None
        else manifest.get("inference_chunk_size", DEFAULT_INFERENCE_CHUNK_SIZE)
    )
    signals = _execute_evaluation(
        model,
        input_audio,
        target_audio,
        device=device,
        chunk_size=chunk_size,
    )
    metrics, metric_recipe = _calculate_evaluation_metrics(signals, manifest, config)
    artifacts = _write_listening_samples(signals, output_dir, config.sample_rate)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest["experiment_id"],
        "run_kind": manifest["run_kind"],
        "sources": {
            "manifest": manifest["manifest_path"],
            "config": (
                str(Path(model_spec["config"]).resolve())
                if model_spec.get("config") is not None
                else None
            ),
            "config_source": (
                "explicit_file"
                if model_spec.get("config") is not None
                else "checkpoint"
            ),
            "checkpoint": str(Path(model_spec["checkpoint"]).resolve()),
        },
        "model": {
            "name": config.name,
            "type": config.model.type,
            "trainable_parameters": trainable_parameters,
            **(
                {"checkpoint_policy": model_spec["checkpoint_policy"]}
                if "checkpoint_policy" in model_spec
                else {}
            ),
        },
        "training": manifest["training"],
        "inference": {
            "chunk_size": signals.chunk_size,
            "chunks": math.ceil(signals.num_samples / signals.chunk_size),
            "state_reset_count": 1,
        },
        "dataset": {
            **manifest["dataset"],
            "evaluated_samples": signals.num_samples,
            **metric_recipe,
            "sample_rate": config.sample_rate,
        },
        "metrics": metrics,
        "performance": _load_evaluation_performance(
            model_spec,
            trainable_parameters,
        ),
        "artifacts": artifacts,
        "notes": manifest.get("notes"),
    }


def write_evaluation_result(result: dict[str, Any], path: str | Path) -> None:
    """Write a checkpoint evaluation result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def load_evaluation_result(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an evaluation result."""
    result = json.loads(Path(path).read_text())
    if result.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported evaluation schema: {result.get('schema_version')}"
        )
    for key in ("experiment_id", "run_kind", "sources", "model", "metrics"):
        if key not in result:
            raise ValueError(f"Evaluation result is missing '{key}'")
    return result


def _size_matched_groups(
    results: list[dict[str, Any]], tolerance: float
) -> list[dict[str, Any]]:
    if tolerance < 1.0:
        raise ValueError("Size tolerance must be at least 1.0")
    ordered = sorted(
        results, key=lambda result: result["model"]["trainable_parameters"]
    )
    groups: list[list[dict[str, Any]]] = []
    for result in ordered:
        parameters = result["model"]["trainable_parameters"]
        if not groups:
            groups.append([result])
            continue
        group_min = groups[-1][0]["model"]["trainable_parameters"]
        if parameters / group_min <= tolerance:
            groups[-1].append(result)
        else:
            groups.append([result])
    return [
        {
            "id": f"size-{index + 1}",
            "min_parameters": group[0]["model"]["trainable_parameters"],
            "max_parameters": group[-1]["model"]["trainable_parameters"],
            "experiments": [result["experiment_id"] for result in group],
        }
        for index, group in enumerate(groups)
    ]




def build_comparison_report(
    results: list[dict[str, Any]], size_tolerance: float = 1.35
) -> tuple[dict[str, Any], str]:
    """Combine quality and performance results into JSON and Markdown reports."""
    if not results:
        raise ValueError("At least one evaluation result is required")
    dataset_keys = (
        "input_audio",
        "target_audio",
        "split",
        "start_sample",
        "evaluated_samples",
        "sample_rate",
        "latency_samples",
        "normalization",
        "mask_first",
        "metric_samples",
    )
    expected_dataset = {
        key: results[0].get("dataset", {}).get(key) for key in dataset_keys
    }
    for result in results[1:]:
        candidate = {key: result.get("dataset", {}).get(key) for key in dataset_keys}
        if candidate != expected_dataset:
            raise ValueError(
                "Evaluation results must use the same aligned dataset segment"
            )
    groups = _size_matched_groups(results, size_tolerance)
    group_by_experiment = {
        experiment: group["id"]
        for group in groups
        for experiment in group["experiments"]
    }
    rows = []
    for result in results:
        performance = result.get("performance") or {}
        offline = performance.get("offline") or {}
        blocks = performance.get("blocks") or []
        block_128 = next(
            (block for block in blocks if block["block_size"] == 128), None
        )
        rows.append(
            {
                "experiment_id": result["experiment_id"],
                "size_group": group_by_experiment[result["experiment_id"]],
                "run_kind": result["run_kind"],
                "model": result["model"],
                "metrics": result["metrics"],
                "offline_real_time_factor": offline.get("real_time_factor"),
                "block_128_p95_ms": block_128.get("p95_ms") if block_128 else None,
                "sources": result["sources"],
                "listening_samples": result.get("artifacts", {}),
            }
        )
    rows.sort(key=lambda row: row["metrics"]["esr"])
    for rank, row in enumerate(rows, start=1):
        row["esr_rank"] = rank
    report = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "workflow_validation_only"
            if any(result["run_kind"] == "smoke" for result in results)
            else "final_experiment"
        ),
        "size_tolerance_ratio": size_tolerance,
        "primary_metric": {
            "name": "esr",
            "direction": "lower_is_better",
            "secondary_metrics": [
                "mse",
                "correlation",
                "multi_resolution_stft_distance",
            ],
        },
        "size_groups": groups,
        "results": rows,
    }

    header = (
        "| ESR rank | Experiment | Size group | Kind | Type | Parameters | ESR | MSE | "
        "MR-STFT | Correlation | Offline RTF | 128 p95 ms | Inputs | Samples |\n"
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
    )
    lines = [header]
    for row in rows:
        sources = row["sources"]
        config_link = (
            f"[config]({sources['config']})"
            if sources.get("config") is not None
            else "config embedded in checkpoint"
        )
        source_links = (
            f"[manifest]({sources['manifest']}), {config_link}, "
            f"[checkpoint]({sources['checkpoint']})"
        )
        samples = row["listening_samples"]
        sample_links = (
            f"[input]({samples['input_audio']}), [target]({samples['target_audio']}), "
            f"[prediction]({samples['prediction_audio']})"
            if samples
            else "—"
        )
        metrics = row["metrics"]
        lines.append(
            "| {esr_rank} | {experiment_id} | {size_group} | {run_kind} | {model_type} | "
            "{parameters:,} | {esr:.6f} | {mse:.6f} | {stft:.6f} | "
            "{correlation:.4f} | {rtf} | {p95} | {sources} | {samples} |".format(
                esr_rank=row["esr_rank"],
                experiment_id=row["experiment_id"],
                size_group=row["size_group"],
                run_kind=row["run_kind"],
                model_type=row["model"]["type"],
                parameters=row["model"]["trainable_parameters"],
                esr=metrics["esr"],
                mse=metrics["mse"],
                stft=metrics["multi_resolution_stft_distance"],
                correlation=metrics["correlation"],
                rtf=(
                    f"{row['offline_real_time_factor']:.4f}"
                    if row["offline_real_time_factor"] is not None
                    else "—"
                ),
                p95=(
                    f"{row['block_128_p95_ms']:.3f}"
                    if row["block_128_p95_ms"] is not None
                    else "—"
                ),
                sources=source_links,
                samples=sample_links,
            )
        )
    prefix = "# Model comparison\n\n"
    prefix += "> ESR is the primary ranking metric. Lower ESR is better. MSE, correlation, and MR-STFT are secondary metrics.\n\n"
    prefix += (
        "> These results include smoke runs. They validate the workflow and must not be used as a final quality ranking.\n\n"
        if report["interpretation"] == "workflow_validation_only"
        else "> Final experiment results. Interpret them with the recorded dataset, seeds, and hardware.\n\n"
    )
    return report, prefix + "\n".join(lines) + "\n"
