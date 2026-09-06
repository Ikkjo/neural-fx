"""Controlled checkpoint evaluation and cross-model comparison."""

from __future__ import annotations

import json
import math
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
from ..metrics import (
    RELATIVE_METRICS,
    average_eligible,
    scoring_diagnostics,
    silence_policy_metadata,
)
from ..preprocessing.latency import LatencyCalibration
from .benchmarking import load_benchmark_result, load_model_for_evaluation

EVALUATION_MANIFEST_SCHEMA_VERSION = "1.0"
EVALUATION_RESULT_SCHEMA_VERSION = "1.1"
EVALUATION_SCHEMA_VERSION = EVALUATION_RESULT_SCHEMA_VERSION
COMPARISON_SCHEMA_VERSION = "1.2"
DEFAULT_INFERENCE_CHUNK_SIZE = 65_536


def _format_metric(value: float | None, digits: int) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    """Load a YAML/JSON experiment manifest and resolve its file inputs."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("Experiment manifest must contain a mapping")
    if data.get("schema_version") != EVALUATION_MANIFEST_SCHEMA_VERSION:
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
) -> tuple[dict[str, float | None], dict[str, Any]]:
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

    complete_diagnostics = scoring_diagnostics(prediction, target)
    pre_emphasis_coeff = manifest.get("esr_pre_emphasis")
    if pre_emphasis_coeff is not None:
        pre_emphasis_coeff = float(pre_emphasis_coeff)
    esr_mode = manifest.get("esr_mode", "legacy")
    stft_starts, stft_window_samples = _stft_window_starts(
        prediction.shape[-1],
        config.sample_rate,
    )
    stft_loss = MultiResolutionSTFTLoss()
    stft_values: list[float | None] = []
    stft_windows = []
    for start in stft_starts:
        window_prediction = prediction[..., start : start + stft_window_samples]
        window_target = target[..., start : start + stft_window_samples]
        diagnostics = scoring_diagnostics(window_prediction, window_target)
        value = (
            None
            if diagnostics["digital_silence"]
            else float(stft_loss(window_prediction, window_target).item())
        )
        stft_values.append(value)
        stft_windows.append(
            {
                "start_sample": start,
                "num_samples": stft_window_samples,
                "multi_resolution_stft_distance": value,
                **diagnostics,
            }
        )
    stft_mean, stft_scored_count, stft_excluded_count = average_eligible(stft_values)
    esr = (
        None
        if complete_diagnostics["digital_silence"]
        else float(
            ESR(
                prediction,
                target,
                pre_emphasis_coeff=pre_emphasis_coeff,
                mode=esr_mode,
            ).item()
        )
    )
    metrics = {
        "esr": esr,
        "mse": complete_diagnostics["mse"],
        "correlation": _safe_correlation(prediction, target),
        "multi_resolution_stft_distance": stft_mean,
    }
    if any(
        value is not None and not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError("Evaluation produced a non-finite metric")
    recipe = {
        "mask_first": mask_first,
        "esr_mode": esr_mode,
        "esr_pre_emphasis": pre_emphasis_coeff,
        "configured_loss_mask_first": config.loss.mask_first,
        "metric_samples": prediction.shape[-1],
        "stft_window_starts": stft_starts,
        "stft_window_samples": stft_window_samples,
        "stft_window_values": stft_values,
        "stft_windows": stft_windows,
        "stft_scored_count": stft_scored_count,
        "stft_excluded_count": stft_excluded_count,
        "scoring_diagnostics": complete_diagnostics,
        "silence_policy": silence_policy_metadata(),
    }
    return metrics, recipe


def _checkpoint_training_state(
    checkpoint_path: str | Path,
) -> dict[str, int | float | str | None]:
    """Read training state when the artifact is a Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_state: dict[str, int | float | str | None] = {
        "epoch": None,
        "global_step": None,
        "monitor": None,
        "monitor_value": None,
    }
    if not isinstance(checkpoint, dict):
        return training_state

    for key in ("epoch", "global_step"):
        value = checkpoint.get(key)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        if isinstance(value, int):
            training_state[key] = value

    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, dict):
        return training_state
    for state in callbacks.values():
        if (
            not isinstance(state, dict)
            or "monitor" not in state
            or "current_score" not in state
        ):
            continue
        monitor = state["monitor"]
        score = state["current_score"]
        if isinstance(score, torch.Tensor) and score.numel() == 1:
            score = score.item()
        training_state["monitor"] = monitor if isinstance(monitor, str) else None
        training_state["monitor_value"] = (
            float(score) if isinstance(score, (int, float)) else None
        )
        break
    return training_state


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
        "workload": benchmark["workload"],
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
        "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
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
        "checkpoint": _checkpoint_training_state(model_spec["checkpoint"]),
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
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def load_evaluation_result(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an evaluation result."""
    result = json.loads(Path(path).read_text())
    if result.get("schema_version") != EVALUATION_RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported evaluation schema: {result.get('schema_version')}"
        )
    for key in ("experiment_id", "run_kind", "sources", "model", "metrics"):
        if key not in result:
            raise ValueError(f"Evaluation result is missing '{key}'")
    for name, value in result["metrics"].items():
        if value is None:
            if name not in RELATIVE_METRICS:
                raise ValueError(f"Evaluation metric '{name}' cannot be null")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Evaluation metric '{name}' must be finite or null")
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


def _relative_value(
    value: float | int | None, reference: float | int | None
) -> dict[str, float | None]:
    if value is None or reference is None or reference == 0:
        return {"ratio": None, "percent_change": None}
    ratio = float(value) / float(reference)
    return {"ratio": ratio, "percent_change": (ratio - 1.0) * 100.0}


def _format_relative(value: dict[str, float | None]) -> str:
    ratio = value["ratio"]
    change = value["percent_change"]
    return (
        "N/A" if ratio is None or change is None else f"{ratio:.2f}x ({change:+.0f}%)"
    )


def _comparison_performance(result: dict[str, Any]) -> dict[str, Any] | None:
    performance = result.get("performance")
    if performance is None:
        return None
    required = ("runtime", "offline", "blocks", "memory")
    if not all(key in performance for key in required):
        raise ValueError("Evaluation performance is incomplete")
    return performance


def _validate_performance(results: list[dict[str, Any]]) -> None:
    performances = [_comparison_performance(result) for result in results]
    if not any(performances):
        return
    if not all(performances):
        raise ValueError("Evaluation results must all include benchmark performance")
    reference = performances[0]
    assert reference is not None
    expected = {
        "runtime": {
            key: reference["runtime"].get(key)
            for key in (
                "device",
                "device_name",
                "dtype",
                "torch_num_threads",
                "platform",
                "python_version",
                "torch_version",
            )
        },
        "workload": None,
    }
    benchmark_workload = reference.get("workload")
    if benchmark_workload is not None:
        expected["workload"] = benchmark_workload
    for performance in performances[1:]:
        assert performance is not None
        candidate = {
            "runtime": {
                key: performance["runtime"].get(key)
                for key in (
                    "device",
                    "device_name",
                    "dtype",
                    "torch_num_threads",
                    "platform",
                    "python_version",
                    "torch_version",
                )
            },
            "workload": performance.get("workload"),
        }
        if candidate != expected:
            raise ValueError(
                "Evaluation results must use the same benchmark environment and workload"
            )


def build_comparison_report(
    results: list[dict[str, Any]],
    *,
    reference_experiment_id: str,
    size_tolerance: float = 1.35,
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
        "preparation_delay_samples",
        "normalization",
        "mask_first",
        "metric_samples",
        "esr_mode",
        "esr_pre_emphasis",
        "silence_policy",
        "stft_window_starts",
        "stft_window_samples",
    )
    for result in results:
        dataset = result.get("dataset", {})
        if "esr_mode" not in dataset or "esr_pre_emphasis" not in dataset:
            raise ValueError(
                "Evaluation results must record esr_mode and esr_pre_emphasis before comparison"
            )
        if "silence_policy" not in dataset:
            raise ValueError(
                "Evaluation results must record the silence policy before comparison"
            )
        if dataset["silence_policy"] != silence_policy_metadata():
            raise ValueError("Evaluation results must use the same silence policy")
        if "stft_window_starts" not in dataset or "stft_window_samples" not in dataset:
            raise ValueError(
                "Evaluation results must record the STFT window recipe before comparison"
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
    references = [
        result
        for result in results
        if result["experiment_id"] == reference_experiment_id
    ]
    if len(references) != 1:
        raise ValueError(
            f"Expected exactly one reference result: {reference_experiment_id}"
        )
    _validate_performance(results)
    reference = references[0]
    reference_performance = _comparison_performance(reference) or {}
    groups = _size_matched_groups(results, size_tolerance)
    group_by_experiment = {
        experiment: group["id"]
        for group in groups
        for experiment in group["experiments"]
    }
    rows = []
    for result in results:
        performance = _comparison_performance(result) or {}
        offline = performance.get("offline") or {}
        blocks = performance.get("blocks") or []
        memory = performance.get("memory") or {}
        block_128 = next(
            (block for block in blocks if block["block_size"] == 128), None
        )
        row = {
            "experiment_id": result["experiment_id"],
            "size_group": group_by_experiment[result["experiment_id"]],
            "run_kind": result["run_kind"],
            "model": result["model"],
            "metrics": result["metrics"],
            "diagnostics": result.get("dataset", {}).get("scoring_diagnostics"),
            "stft_scored_count": result.get("dataset", {}).get("stft_scored_count"),
            "stft_excluded_count": result.get("dataset", {}).get("stft_excluded_count"),
            "checkpoint": result.get("checkpoint"),
            "offline_real_time_factor": offline.get("real_time_factor"),
            "block_128_p95_ms": block_128.get("p95_ms") if block_128 else None,
            "blocks": blocks,
            "model_state_bytes": memory.get("model_state_bytes"),
            "sources": result["sources"],
            "listening_samples": result.get("artifacts", {}),
        }
        rows.append(row)
    reference_blocks = {
        block["block_size"]: block for block in reference_performance.get("blocks", [])
    }
    for row in rows:
        row["relative_to_reference"] = {
            "esr": _relative_value(
                row["metrics"].get("esr"), reference["metrics"].get("esr")
            ),
            "mse": _relative_value(
                row["metrics"].get("mse"), reference["metrics"].get("mse")
            ),
            "multi_resolution_stft_distance": _relative_value(
                row["metrics"].get("multi_resolution_stft_distance"),
                reference["metrics"].get("multi_resolution_stft_distance"),
            ),
            "trainable_parameters": _relative_value(
                row["model"].get("trainable_parameters"),
                reference["model"].get("trainable_parameters"),
            ),
            "offline_real_time_factor": _relative_value(
                row["offline_real_time_factor"],
                (reference_performance.get("offline") or {}).get("real_time_factor"),
            ),
            "block_p95_ms": {
                str(block["block_size"]): _relative_value(
                    block.get("p95_ms"),
                    reference_blocks.get(block["block_size"], {}).get("p95_ms"),
                )
                for block in row["blocks"]
            },
        }
    rows.sort(
        key=lambda row: (
            row["metrics"]["esr"] is None,
            row["metrics"]["esr"] if row["metrics"]["esr"] is not None else math.inf,
        )
    )
    rank = 1
    for row in rows:
        if row["metrics"]["esr"] is None:
            row["esr_rank"] = None
        else:
            row["esr_rank"] = rank
            rank += 1
    report = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "workflow_validation_only"
            if any(result["run_kind"] == "smoke" for result in results)
            else "final_experiment"
        ),
        "size_tolerance_ratio": size_tolerance,
        "reference_experiment_id": reference_experiment_id,
        "relative_value_convention": {
            "reference": "candidate / LSTM-40 reference",
            "percent_change": "(candidate / reference - 1) * 100",
            "lower_is_better": [
                "esr",
                "mse",
                "multi_resolution_stft_distance",
                "trainable_parameters",
                "offline_real_time_factor",
                "block_p95_ms",
            ],
            "unavailable": "null when the reference or candidate value is unavailable, or the reference is zero",
        },
        "primary_metric": {
            "name": "esr",
            "direction": "lower_is_better",
            "secondary_metrics": [
                "mse",
                "correlation",
                "multi_resolution_stft_distance",
            ],
        },
        "silence_policy": silence_policy_metadata(),
        "size_groups": groups,
        "results": rows,
    }

    quality_header = (
        "| ESR rank | Experiment | Type | Parameters (relative) | State (bytes) | ESR (relative) | MSE (relative) | MR-STFT (relative) | STFT scored/excluded | Correlation |\n"
        "| ---: | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |"
    )
    quality_lines = [quality_header]
    cpu_header = (
        "| Experiment | Offline RTF (relative) | 64 p95/deadline/misses (relative) | 128 p95/deadline/misses (relative) | 256 p95/deadline/misses (relative) | 512 p95/deadline/misses (relative) |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    cpu_lines = [cpu_header]
    for row in rows:
        metrics = row["metrics"]
        relative = row["relative_to_reference"]
        stft_counts = (
            f"{row['stft_scored_count']}/{row['stft_excluded_count']}"
            if row["stft_scored_count"] is not None
            and row["stft_excluded_count"] is not None
            else "—"
        )

        quality_lines.append(
            "| {esr_rank} | {experiment_id} | {model_type} | {parameters:,} ({parameters_relative}) | "
            "{state} | {esr} ({esr_relative}) | {mse} ({mse_relative}) | {stft} ({stft_relative}) | "
            "{stft_counts} | {correlation:.4f} |".format(
                esr_rank=(row["esr_rank"] if row["esr_rank"] is not None else "—"),
                experiment_id=row["experiment_id"],
                model_type=row["model"].get("type", "unknown"),
                parameters=row["model"]["trainable_parameters"],
                parameters_relative=_format_relative(relative["trainable_parameters"]),
                state=(
                    row["model_state_bytes"]
                    if row["model_state_bytes"] is not None
                    else "N/A"
                ),
                esr=_format_metric(metrics["esr"], 6),
                esr_relative=_format_relative(relative["esr"]),
                mse=_format_metric(metrics["mse"], 6),
                mse_relative=_format_relative(relative["mse"]),
                stft=_format_metric(metrics["multi_resolution_stft_distance"], 6),
                stft_relative=_format_relative(
                    relative["multi_resolution_stft_distance"]
                ),
                stft_counts=stft_counts,
                correlation=metrics["correlation"],
            )
        )
        blocks = {block["block_size"]: block for block in row["blocks"]}

        def block_cell(size: int) -> str:
            block = blocks.get(size)
            if block is None:
                return "N/A"
            status = "met" if block.get("deadline_misses") == 0 else "missed"
            return (
                f"{block['p95_ms']:.3f}/{block['deadline_ms']:.3f}/{block['deadline_misses']} "
                f"({_format_relative(relative['block_p95_ms'].get(str(size), {'ratio': None, 'percent_change': None}))}; {status})"
            )

        cpu_lines.append(
            "| {experiment} | {rtf} ({relative_rtf}) | {b64} | {b128} | {b256} | {b512} |".format(
                experiment=row["experiment_id"],
                rtf=_format_metric(row["offline_real_time_factor"], 4),
                relative_rtf=_format_relative(relative["offline_real_time_factor"]),
                b64=block_cell(64),
                b128=block_cell(128),
                b256=block_cell(256),
                b512=block_cell(512),
            )
        )
    prefix = "# Model comparison\n\n"
    prefix += "> ESR is the primary ranking metric. Lower ESR is better. MSE, correlation, and MR-STFT are secondary metrics. Ratios use candidate / LSTM-40; for ratio metrics, lower is better.\n\n"
    prefix += (
        "> These results include smoke runs. They validate the workflow and must not be used as a final quality ranking.\n\n"
        if report["interpretation"] == "workflow_validation_only"
        else "> Final experiment results. Interpret them with the recorded dataset, seeds, and hardware.\n\n"
    )
    environment = reference_performance.get("runtime")
    workload = reference_performance.get("workload")
    if environment and workload:
        prefix += (
            f"> Environment: {environment.get('device_name')}; {environment.get('platform')}; "
            f"PyTorch {environment.get('torch_version')}; {environment.get('dtype')}; "
            f"{environment.get('torch_num_threads')} thread(s); {workload.get('warmup_runs')} warm-ups; "
            f"{workload.get('measurement_runs')} measurements.\n\n"
        )
    prefix += "> Block measurements cover stateful model-forward compute only. They exclude audio-interface latency, buffering, operating-system scheduling, and device round-trip. `met` means zero recorded deadline misses; it is not a hard real-time guarantee.\n\n"
    return report, prefix + "## Quality and model\n\n" + "\n".join(
        quality_lines
    ) + "\n\n## CPU\n\n" + "\n".join(cpu_lines) + "\n"
