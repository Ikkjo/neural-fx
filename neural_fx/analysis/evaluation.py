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

from ..data.dataset import AudioDataset
from ..losses.audio_losses import ESR, MultiResolutionSTFTLoss
from ..preprocessing.latency import LatencyCalibration
from .benchmarking import load_benchmark_result, load_model_for_evaluation

EVALUATION_SCHEMA_VERSION = "1.0"
COMPARISON_SCHEMA_VERSION = "1.0"
ARCHITECTURE_REPORT_SCHEMA_VERSION = "1.0"
DEFAULT_INFERENCE_CHUNK_SIZE = 65_536
QUALITY_METRICS = (
    "esr",
    "mse",
    "correlation",
    "multi_resolution_stft_distance",
)


@dataclass(frozen=True)
class ArchitectureComparisonPolicy:
    """Rules for one controlled, repeated architecture comparison."""

    decision_rule_id: str
    required_seeds: tuple[int, ...]
    required_architectures: tuple[str, ...]
    size_tolerance_ratio: float
    median_esr_relative_improvement: float
    require_lower_esr_for_all_matched_seeds: bool
    require_esr_standard_deviation_below_median_gap: bool
    maximum_median_mse_regression: float
    maximum_median_mr_stft_regression: float
    maximum_median_correlation_regression: float
    minimum_performance_relative_improvement: float
    require_non_overlapping_performance_ranges: bool
    real_time_block_size: int
    real_time_p95_deadline_fraction: float
    benchmark_runs_per_device: int


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
    audio_dataset = AudioDataset(
        input_path=dataset["input_audio"],
        target_path=dataset["target_audio"],
        segment_length=1,
        sample_rate=sample_rate,
        normalize=normalization == "paired_peak",
        random_segments=False,
        latency_calibration=LatencyCalibration(
            delay_samples=delay_samples,
            method="manual",
            correlation_score=1.0,
        ),
    )
    input_audio = audio_dataset.input_audio
    target_audio = audio_dataset.target_audio

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
    if chunk_size <= 0:
        raise ValueError("inference chunk size must be positive")
    model.reset_state()
    chunks = []
    with torch.inference_mode():
        for start in range(0, audio.shape[-1], chunk_size):
            chunks.append(model(audio[..., start : start + chunk_size]))
    if not chunks:
        raise ValueError("evaluation audio must contain at least one sample")
    return torch.cat(chunks, dim=-1)


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


def evaluate_experiment(
    manifest: dict[str, Any],
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    inference_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint and write aligned listening samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = manifest["model"]
    model, config = load_model_for_evaluation(
        model_spec.get("config"), model_spec["checkpoint"], device=device
    )
    input_audio, target_audio = _prepare_evaluation_audio(manifest, config.sample_rate)

    input_batch = input_audio.unsqueeze(0).to(device)
    chunk_size = int(
        inference_chunk_size
        if inference_chunk_size is not None
        else manifest.get("inference_chunk_size", DEFAULT_INFERENCE_CHUNK_SIZE)
    )
    prediction = run_chunked_inference(model, input_batch, chunk_size).detach().cpu()
    target_batch = target_audio.unsqueeze(0)
    input_batch = input_batch.cpu()
    common_length = min(
        input_batch.shape[-1], target_batch.shape[-1], prediction.shape[-1]
    )
    input_batch = input_batch[..., :common_length]
    target_batch = target_batch[..., :common_length]
    prediction = prediction[..., :common_length]
    if common_length < 2048:
        raise ValueError("Evaluation segment must contain at least 2048 samples")

    mask_first = int(
        manifest.get(
            "burn_in_samples",
            manifest["dataset"].get("metric_mask_first", config.loss.mask_first),
        )
    )
    if mask_first < 0 or mask_first >= common_length:
        raise ValueError(
            f"loss.mask_first ({mask_first}) must be smaller than the evaluation segment ({common_length})"
        )
    metric_prediction = prediction[..., mask_first:]
    metric_target = target_batch[..., mask_first:]
    if metric_prediction.shape[-1] < 2048:
        raise ValueError(
            "Evaluation segment after loss masking must contain 2048 samples"
        )
    pre_emphasis = config.loss.pre_emphasis
    pre_emphasis_coeff = (
        pre_emphasis.coef if pre_emphasis is not None and pre_emphasis.enabled else None
    )
    stft_starts, stft_window_samples = _stft_window_starts(
        metric_prediction.shape[-1], config.sample_rate
    )
    stft_loss = MultiResolutionSTFTLoss()
    stft_values = [
        stft_loss(
            metric_prediction[..., start : start + stft_window_samples],
            metric_target[..., start : start + stft_window_samples],
        ).item()
        for start in stft_starts
    ]
    stft_distance = statistics.fmean(stft_values)
    metrics = {
        "esr": ESR(
            metric_prediction,
            metric_target,
            pre_emphasis_coeff=pre_emphasis_coeff,
        ).item(),
        "mse": torch.mean((metric_prediction - metric_target) ** 2).item(),
        "correlation": _safe_correlation(metric_prediction, metric_target),
        "multi_resolution_stft_distance": stft_distance,
    }

    artifacts = {}
    for name, audio in (
        ("input", input_batch),
        ("target", target_batch),
        ("prediction", prediction),
    ):
        path = output_dir / f"{name}.wav"
        torchaudio.save(
            str(path), audio.squeeze(0).clamp(-1.0, 1.0), config.sample_rate
        )
        artifacts[f"{name}_audio"] = str(path.resolve())

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    performance = None
    benchmark_path = model_spec.get("benchmark_result")
    if benchmark_path is not None:
        benchmark = load_benchmark_result(benchmark_path)
        if benchmark["model"]["trainable_parameters"] != trainable_parameters:
            raise ValueError(
                "Benchmark parameter count does not match the evaluated model"
            )
        performance = {
            "result_path": str(Path(benchmark_path).resolve()),
            "runtime": benchmark["runtime"],
            "offline": benchmark["offline"],
            "blocks": benchmark["blocks"],
            "memory": benchmark["memory"],
        }

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
        },
        "training": manifest["training"],
        "inference": {
            "chunk_size": chunk_size,
            "chunks": math.ceil(common_length / chunk_size),
            "state_reset_count": 1,
        },
        "dataset": {
            **manifest["dataset"],
            "evaluated_samples": common_length,
            "mask_first": mask_first,
            "configured_loss_mask_first": config.loss.mask_first,
            "metric_samples": metric_prediction.shape[-1],
            "stft_window_starts": stft_starts,
            "stft_window_samples": stft_window_samples,
            "stft_window_values": stft_values,
            "sample_rate": config.sample_rate,
        },
        "metrics": metrics,
        "performance": performance,
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


def _numeric_summary(values_by_key: dict[str, float]) -> dict[str, Any]:
    """Summarize raw values without hiding any individual observation."""
    values = list(values_by_key.values())
    return {
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "raw": values_by_key,
    }


def _aggregate_benchmarks(
    benchmark_results: list[dict[str, Any]],
    representative_checkpoint: str | None,
    policy: ArchitectureComparisonPolicy,
) -> dict[str, Any]:
    """Aggregate fresh benchmark processes by device."""
    by_device: dict[str, list[dict[str, Any]]] = {}
    for result in benchmark_results:
        device = str(result["runtime"]["device"]).split(":", maxsplit=1)[0]
        by_device.setdefault(device, []).append(result)

    devices = {}
    for device, runs in sorted(by_device.items()):
        checkpoints_match = representative_checkpoint is not None and all(
            result["model"].get("checkpoint_path") == representative_checkpoint
            for result in runs
        )
        block_size_sets = [
            {block["block_size"] for block in result["blocks"]} for result in runs
        ]
        block_recipe_matches = bool(block_size_sets) and all(
            sizes == block_size_sets[0] for sizes in block_size_sets
        )
        blocks = {}
        block_sizes = (
            sorted(set.intersection(*block_size_sets)) if block_size_sets else []
        )
        for block_size in block_sizes:
            matching = [
                next(
                    block
                    for block in result["blocks"]
                    if block["block_size"] == block_size
                )
                for result in runs
            ]
            timing = {
                output_key: _numeric_summary(
                    {
                        str(index + 1): float(block[input_key])
                        for index, block in enumerate(matching)
                    }
                )
                for output_key, input_key in (
                    ("p50_ms", "median_ms"),
                    ("p95_ms", "p95_ms"),
                    ("max_ms", "max_ms"),
                )
                if all(input_key in block for block in matching)
            }
            deadline_misses = {
                str(index + 1): float(block["deadline_misses"])
                for index, block in enumerate(matching)
            }
            deadline_headroom = {
                str(index + 1): float(block["deadline_ms"] - block["p95_ms"])
                for index, block in enumerate(matching)
            }
            blocks[str(block_size)] = {
                **timing,
                "deadline_ms": matching[0]["deadline_ms"],
                "deadline_misses": _numeric_summary(deadline_misses),
                "deadline_headroom_ms": _numeric_summary(deadline_headroom),
                "real_time_capable": all(
                    block["deadline_misses"] == 0
                    and block["p95_ms"]
                    <= policy.real_time_p95_deadline_fraction * block["deadline_ms"]
                    for block in matching
                ),
            }
        memory = {}
        for key in (
            "process_peak_rss_bytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            values = {
                str(index + 1): float(result["memory"][key])
                for index, result in enumerate(runs)
                if key in result["memory"]
            }
            if values:
                memory[key] = _numeric_summary(values)
        offline = {
            key: _numeric_summary(
                {
                    str(index + 1): float(result["offline"][key])
                    for index, result in enumerate(runs)
                }
            )
            for key in ("median_ms", "p95_ms", "max_ms", "real_time_factor")
            if all(key in result["offline"] for result in runs)
        }
        model_resources = {
            key: _numeric_summary(
                {
                    str(index + 1): float(result["model"][key])
                    for index, result in enumerate(runs)
                }
            )
            for key in ("checkpoint_size_bytes", "trainable_parameters")
            if all(result["model"].get(key) is not None for result in runs)
        }
        if all("model_state_bytes" in result["memory"] for result in runs):
            model_resources["model_state_bytes"] = _numeric_summary(
                {
                    str(index + 1): float(result["memory"]["model_state_bytes"])
                    for index, result in enumerate(runs)
                }
            )
        devices[device] = {
            "status": (
                "complete"
                if len(runs) == policy.benchmark_runs_per_device
                and checkpoints_match
                and block_recipe_matches
                else "incomplete"
            ),
            "run_count": len(runs),
            "representative_checkpoint_matches": checkpoints_match,
            "block_recipe_matches": block_recipe_matches,
            "result_paths": [result.get("_result_path") for result in runs],
            "offline": offline,
            "offline_real_time_factor": offline["real_time_factor"],
            "blocks": blocks,
            "model_resources": model_resources,
            "memory": memory,
        }
    return devices


def _quality_conclusion(
    architectures: list[dict[str, Any]],
    policy: ArchitectureComparisonPolicy,
    *,
    size_matched: bool,
) -> dict[str, Any]:
    complete = [item for item in architectures if item["status"] == "complete"]
    if len(complete) != len(architectures) or len(complete) < 2 or not size_matched:
        return {
            "status": "incomplete",
            "winner": None,
            "statement": (
                "architecture comparison is incomplete; no quality conclusion is allowed"
            ),
            "conditions": {},
        }

    ordered = sorted(complete, key=lambda item: item["metrics"]["esr"]["median"])
    candidate, next_best = ordered[:2]
    candidate_esr = candidate["metrics"]["esr"]
    next_esr = next_best["metrics"]["esr"]
    median_gap = next_esr["median"] - candidate_esr["median"]
    matched_seed_lower = all(
        candidate_esr["raw"][seed] < next_esr["raw"][seed]
        for seed in candidate_esr["raw"]
    )
    conditions = {
        "median_esr_relative_improvement_met": (
            median_gap / next_esr["median"] >= policy.median_esr_relative_improvement
        ),
        "esr_lower_for_all_matched_seeds": (
            matched_seed_lower or not policy.require_lower_esr_for_all_matched_seeds
        ),
        "esr_standard_deviation_smaller_than_median_gap": (
            candidate_esr["standard_deviation"] < median_gap
            or not policy.require_esr_standard_deviation_below_median_gap
        ),
        "median_mse_regression_within_limit": (
            candidate["metrics"]["mse"]["median"]
            <= next_best["metrics"]["mse"]["median"]
            * (1.0 + policy.maximum_median_mse_regression)
        ),
        "median_mr_stft_regression_within_limit": (
            candidate["metrics"]["multi_resolution_stft_distance"]["median"]
            <= next_best["metrics"]["multi_resolution_stft_distance"]["median"]
            * (1.0 + policy.maximum_median_mr_stft_regression)
        ),
        "median_correlation_regression_within_limit": (
            candidate["metrics"]["correlation"]["median"]
            >= next_best["metrics"]["correlation"]["median"]
            - policy.maximum_median_correlation_regression
        ),
    }
    winner = candidate["architecture"] if all(conditions.values()) else None
    return {
        "status": "complete",
        "candidate": candidate["architecture"],
        "next_best": next_best["architecture"],
        "winner": winner,
        "statement": (
            f"{winner} is the clear quality winner under this budget"
            if winner is not None
            else "no clear quality winner under this budget"
        ),
        "median_esr_gap": median_gap,
        "conditions": conditions,
    }


def _performance_conclusion(
    architectures: list[dict[str, Any]],
    quality_winner: str | None,
    policy: ArchitectureComparisonPolicy,
) -> dict[str, Any]:
    block_size = str(policy.real_time_block_size)
    cpu_ready = [
        item
        for item in architectures
        if item.get("benchmarks", {}).get("cpu", {}).get("status") == "complete"
        and block_size in item["benchmarks"]["cpu"]["blocks"]
    ]
    if len(cpu_ready) != len(architectures) or len(cpu_ready) < 2:
        return {
            "status": "incomplete",
            "material_differences": {},
            "pareto_winner": None,
            "statement": "performance comparison is incomplete",
        }

    def material(metric_getter: Any) -> dict[str, Any]:
        ordered = sorted(cpu_ready, key=lambda item: metric_getter(item)["median"])
        fastest, next_fastest = ordered[:2]
        fastest_values = metric_getter(fastest)
        next_values = metric_getter(next_fastest)
        improvement = (next_values["median"] - fastest_values["median"]) / next_values[
            "median"
        ]
        is_material = (
            improvement >= policy.minimum_performance_relative_improvement
            and (
                fastest_values["maximum"] < next_values["minimum"]
                or not policy.require_non_overlapping_performance_ranges
            )
        )
        return {
            "winner": fastest["architecture"] if is_material else None,
            "candidate": fastest["architecture"],
            "next_best": next_fastest["architecture"],
            "relative_improvement": improvement,
            "ranges_do_not_overlap": (
                fastest_values["maximum"] < next_values["minimum"]
            ),
            "material": is_material,
        }

    material_differences = {
        f"cpu_{block_size}_sample_p95": material(
            lambda item: item["benchmarks"]["cpu"]["blocks"][block_size]["p95_ms"]
        ),
        "cpu_offline_real_time_factor": material(
            lambda item: item["benchmarks"]["cpu"]["offline_real_time_factor"]
        ),
    }

    pareto_winner = None
    if quality_winner is not None:
        candidate = next(
            item for item in architectures if item["architecture"] == quality_winner
        )
        candidate_cpu = candidate["benchmarks"]["cpu"]
        candidate_values = (
            candidate["parameters"],
            candidate_cpu["blocks"][block_size]["p95_ms"]["median"],
            candidate_cpu["offline_real_time_factor"]["median"],
            candidate_cpu["memory"]["process_peak_rss_bytes"]["median"],
        )
        if all(
            all(left <= right for left, right in zip(candidate_values, other_values))
            for other in architectures
            if other["architecture"] != quality_winner
            for other_values in [
                (
                    other["parameters"],
                    other["benchmarks"]["cpu"]["blocks"][block_size]["p95_ms"][
                        "median"
                    ],
                    other["benchmarks"]["cpu"]["offline_real_time_factor"]["median"],
                    other["benchmarks"]["cpu"]["memory"]["process_peak_rss_bytes"][
                        "median"
                    ],
                )
            ]
        ):
            pareto_winner = quality_winner
    return {
        "status": "complete",
        "material_differences": material_differences,
        "pareto_winner": pareto_winner,
        "statement": (
            f"{pareto_winner} is Pareto-dominant under the configured rule"
            if pareto_winner is not None
            else "no Pareto-dominant model under the configured rule"
        ),
    }


def build_architecture_report(
    results: list[dict[str, Any]],
    benchmark_results: list[dict[str, Any]] | None = None,
    *,
    policy: ArchitectureComparisonPolicy,
) -> tuple[dict[str, Any], str]:
    """Aggregate final seeds and apply an explicit comparison policy."""
    benchmark_results = benchmark_results or []
    by_architecture: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_architecture.setdefault(result["model"]["type"], []).append(result)
    benchmarks_by_architecture: dict[str, list[dict[str, Any]]] = {}
    for result in benchmark_results:
        benchmarks_by_architecture.setdefault(result["model"]["type"], []).append(
            result
        )

    architectures = []
    required_seed_set = set(policy.required_seeds)
    for architecture in policy.required_architectures:
        runs = by_architecture.get(architecture, [])
        seed_runs = {int(run["training"]["seed"]): run for run in runs}
        seeds = set(seed_runs)
        finite = all(
            math.isfinite(float(run["metrics"][metric]))
            for run in runs
            for metric in QUALITY_METRICS
        )
        parameter_counts = {run["model"]["trainable_parameters"] for run in runs}
        complete = (
            seeds == required_seed_set
            and len(runs) == len(policy.required_seeds)
            and len(parameter_counts) == 1
            and finite
            and all(run["run_kind"] == "final" for run in runs)
        )
        incomplete_reasons = []
        if seeds != required_seed_set or len(runs) != len(policy.required_seeds):
            incomplete_reasons.append("required_seed_set_not_complete")
        if len(parameter_counts) != 1:
            incomplete_reasons.append("parameter_count_not_consistent")
        if not finite:
            incomplete_reasons.append("quality_metric_not_finite")
        if any(run["run_kind"] != "final" for run in runs):
            incomplete_reasons.append("non_final_run_present")
        metrics = (
            {
                metric: _numeric_summary(
                    {
                        str(seed): float(seed_runs[seed]["metrics"][metric])
                        for seed in sorted(seed_runs)
                    }
                )
                for metric in QUALITY_METRICS
            }
            if runs
            else {}
        )
        representative = None
        representative_checkpoint = None
        if complete:
            representative_run = sorted(runs, key=lambda run: run["metrics"]["esr"])[
                len(runs) // 2
            ]
            representative_checkpoint = representative_run["sources"]["checkpoint"]
            representative = {
                "seed": int(representative_run["training"]["seed"]),
                "experiment_id": representative_run["experiment_id"],
                "selection_rule": "median_esr_seed",
                "sources": representative_run["sources"],
                "listening_samples": representative_run.get("artifacts", {}),
            }
        architectures.append(
            {
                "architecture": architecture,
                "status": "complete" if complete else "incomplete",
                "incomplete_reasons": incomplete_reasons,
                "seeds": sorted(seeds),
                "missing_seeds": sorted(required_seed_set - seeds),
                "parameters": next(iter(parameter_counts))
                if len(parameter_counts) == 1
                else None,
                "metrics": metrics,
                "runs": [
                    {
                        "seed": int(run["training"]["seed"]),
                        "experiment_id": run["experiment_id"],
                        "metrics": run["metrics"],
                        "training": run["training"],
                        "sources": run["sources"],
                        "artifacts": run.get("artifacts", {}),
                    }
                    for run in sorted(
                        runs, key=lambda run: int(run["training"]["seed"])
                    )
                ],
                "representative": representative,
                "benchmarks": _aggregate_benchmarks(
                    benchmarks_by_architecture.get(architecture, []),
                    representative_checkpoint,
                    policy,
                ),
            }
        )

    size_inputs = [
        {
            "experiment_id": item["architecture"],
            "model": {"trainable_parameters": item["parameters"]},
        }
        for item in architectures
        if item["parameters"] is not None
    ]
    size_groups = (
        _size_matched_groups(size_inputs, policy.size_tolerance_ratio)
        if size_inputs
        else []
    )
    all_architectures_size_matched = len(size_groups) == 1 and len(
        size_groups[0]["experiments"]
    ) == len(policy.required_architectures)
    quality = _quality_conclusion(
        architectures,
        policy,
        size_matched=all_architectures_size_matched,
    )
    performance = _performance_conclusion(
        architectures,
        quality.get("winner"),
        policy,
    )
    report = {
        "schema_version": ARCHITECTURE_REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_rule": policy.decision_rule_id,
        "required_seeds": list(policy.required_seeds),
        "required_architectures": list(policy.required_architectures),
        "size_tolerance_ratio": policy.size_tolerance_ratio,
        "size_groups": size_groups,
        "all_architectures_size_matched": all_architectures_size_matched,
        "rules": {
            "quality": {
                "median_esr_relative_improvement": policy.median_esr_relative_improvement,
                "requires_lower_esr_for_all_matched_seeds": policy.require_lower_esr_for_all_matched_seeds,
                "requires_esr_standard_deviation_below_median_gap": policy.require_esr_standard_deviation_below_median_gap,
                "maximum_median_mse_regression": policy.maximum_median_mse_regression,
                "maximum_median_mr_stft_regression": policy.maximum_median_mr_stft_regression,
                "maximum_median_correlation_regression": policy.maximum_median_correlation_regression,
            },
            "performance": {
                "minimum_median_relative_improvement": policy.minimum_performance_relative_improvement,
                "requires_non_overlapping_ranges": policy.require_non_overlapping_performance_ranges,
                "real_time_block_size": policy.real_time_block_size,
                "real_time_p95_deadline_fraction": policy.real_time_p95_deadline_fraction,
                "benchmark_runs_per_device": policy.benchmark_runs_per_device,
            },
            "pareto": {
                "requires_quality_winner": True,
                "dimensions": [
                    "parameters",
                    f"cpu_{policy.real_time_block_size}_sample_p95",
                    "cpu_offline_real_time_factor",
                    "cpu_peak_rss",
                ],
            },
        },
        "architectures": architectures,
        "conclusion": {
            "quality": quality,
            "performance": performance,
        },
    }

    lines = [
        "# Architecture-level model comparison",
        "",
        f"> {quality['statement']}.",
        "",
        "| Architecture | Seeds | Parameters | ESR median | ESR std | MSE median | MR-STFT median | Correlation median | Representative |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in architectures:
        if item["status"] == "complete":
            representative = item["representative"]
            lines.append(
                "| {architecture} | {seeds} | {parameters:,} | {esr:.6f} | "
                "{esr_std:.6f} | {mse:.6f} | {stft:.6f} | {correlation:.4f} | "
                "seed {representative_seed} ({experiment_id}) |".format(
                    architecture=item["architecture"],
                    seeds=", ".join(str(seed) for seed in item["seeds"]),
                    parameters=item["parameters"],
                    esr=item["metrics"]["esr"]["median"],
                    esr_std=item["metrics"]["esr"]["standard_deviation"],
                    mse=item["metrics"]["mse"]["median"],
                    stft=item["metrics"]["multi_resolution_stft_distance"]["median"],
                    correlation=item["metrics"]["correlation"]["median"],
                    representative_seed=representative["seed"],
                    experiment_id=representative["experiment_id"],
                )
            )
        else:
            lines.append(
                f"| {item['architecture']} | {', '.join(map(str, item['seeds'])) or '—'} "
                f"| {item['parameters'] or '—'} | — | — | — | — | — | incomplete |"
            )
    lines.extend(
        [
            "",
            "## Policy conclusions",
            "",
            f"- Quality: {quality['statement']}.",
            f"- Performance: {performance['statement']}.",
            "- Listening representatives are the median-ESR seed for each architecture; listening cannot override the metric conclusion.",
        ]
    )
    return report, "\n".join(lines) + "\n"


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
    report = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "workflow_validation_only"
            if any(result["run_kind"] == "smoke" for result in results)
            else "final_experiment"
        ),
        "size_tolerance_ratio": size_tolerance,
        "size_groups": groups,
        "results": rows,
    }

    header = (
        "| Experiment | Size group | Kind | Type | Parameters | ESR | MSE | "
        "MR-STFT | Correlation | Offline RTF | 128 p95 ms | Inputs | Samples |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
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
            "| {experiment_id} | {size_group} | {run_kind} | {model_type} | "
            "{parameters:,} | {esr:.6f} | {mse:.6f} | {stft:.6f} | "
            "{correlation:.4f} | {rtf} | {p95} | {sources} | {samples} |".format(
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
    prefix = "# Model comparison\n\n" + (
        "> These results include smoke runs. They validate the workflow and must not be used as a final quality ranking.\n\n"
        if report["interpretation"] == "workflow_validation_only"
        else "> Final experiment results. Interpret them with the recorded dataset, seeds, and hardware.\n\n"
    )
    return report, prefix + "\n".join(lines) + "\n"
