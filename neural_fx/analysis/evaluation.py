"""Controlled checkpoint evaluation and cross-model comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import yaml

from ..losses.audio_losses import ESR, MultiResolutionSTFTLoss
from ..preprocessing.latency import LatencyCalibrator
from .benchmarking import load_benchmark_result, load_model_for_evaluation

EVALUATION_SCHEMA_VERSION = "1.0"
COMPARISON_SCHEMA_VERSION = "1.0"


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
    for key in ("config", "checkpoint"):
        if key not in data["model"]:
            raise ValueError(f"Experiment model is missing '{key}'")
        data["model"][key] = str(_resolve_path(data["model"][key], path))
    for key in ("input_audio", "target_audio", "split"):
        if key not in data["dataset"]:
            raise ValueError(f"Experiment dataset is missing '{key}'")
    for key in ("input_audio", "target_audio"):
        data["dataset"][key] = str(_resolve_path(data["dataset"][key], path))
    if "seed" not in data["training"]:
        raise ValueError("Experiment training section is missing 'seed'")
    benchmark_path = data["model"].get("benchmark_result")
    if benchmark_path is not None:
        data["model"]["benchmark_result"] = str(
            _resolve_path(benchmark_path, path)
        )
    data["manifest_path"] = str(path)
    return data


def _load_mono_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    audio, source_rate = torchaudio.load(str(path))
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio.to(dtype=torch.float32)


def _prepare_evaluation_audio(
    manifest: dict[str, Any], sample_rate: int
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = manifest["dataset"]
    input_audio = _load_mono_audio(dataset["input_audio"], sample_rate)
    target_audio = _load_mono_audio(dataset["target_audio"], sample_rate)
    delay_samples = int(dataset.get("latency_samples", 0))
    input_audio, target_audio = LatencyCalibrator().apply_delay(
        input_audio, target_audio, delay_samples
    )

    start = int(dataset.get("start_sample", 0))
    available = min(input_audio.shape[-1], target_audio.shape[-1])
    num_samples = int(dataset.get("num_samples", available - start))
    if start < 0 or num_samples <= 0 or start + num_samples > available:
        raise ValueError(
            f"Requested samples [{start}, {start + num_samples}) exceed aligned audio length {available}"
        )
    input_audio = input_audio[..., start : start + num_samples]
    target_audio = target_audio[..., start : start + num_samples]

    normalization = dataset.get("normalization", "paired_peak")
    if normalization == "paired_peak":
        peak = torch.maximum(input_audio.abs().max(), target_audio.abs().max())
        if peak > 0:
            input_audio = input_audio / peak
            target_audio = target_audio / peak
    elif normalization != "none":
        raise ValueError("normalization must be 'paired_peak' or 'none'")
    return input_audio, target_audio


def _safe_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = prediction.detach().cpu().numpy().ravel()
    target_np = target.detach().cpu().numpy().ravel()
    if np.std(pred_np) == 0 or np.std(target_np) == 0:
        return 0.0
    return float(np.corrcoef(pred_np, target_np)[0, 1])


def evaluate_experiment(
    manifest: dict[str, Any],
    output_dir: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate one checkpoint and write aligned listening samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = manifest["model"]
    model, config = load_model_for_evaluation(
        model_spec["config"], model_spec["checkpoint"], device=device
    )
    input_audio, target_audio = _prepare_evaluation_audio(
        manifest, config.sample_rate
    )

    input_batch = input_audio.unsqueeze(0).to(device)
    with torch.inference_mode():
        model.reset_state()
        prediction = model(input_batch).detach().cpu()
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

    stft_distance = MultiResolutionSTFTLoss()(prediction, target_batch).item()
    metrics = {
        "esr": ESR(prediction, target_batch).item(),
        "mse": torch.mean((prediction - target_batch) ** 2).item(),
        "correlation": _safe_correlation(prediction, target_batch),
        "multi_resolution_stft_distance": stft_distance,
    }

    artifacts = {}
    for name, audio in (
        ("input", input_batch),
        ("target", target_batch),
        ("prediction", prediction),
    ):
        path = output_dir / f"{name}.wav"
        torchaudio.save(str(path), audio.squeeze(0).clamp(-1.0, 1.0), config.sample_rate)
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
            "config": str(Path(model_spec["config"]).resolve()),
            "checkpoint": str(Path(model_spec["checkpoint"]).resolve()),
        },
        "model": {
            "name": config.name,
            "type": config.model.type,
            "trainable_parameters": trainable_parameters,
        },
        "training": manifest["training"],
        "dataset": {
            **manifest["dataset"],
            "evaluated_samples": common_length,
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
    ordered = sorted(results, key=lambda result: result["model"]["trainable_parameters"])
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
    )
    expected_dataset = {
        key: results[0].get("dataset", {}).get(key) for key in dataset_keys
    }
    for result in results[1:]:
        candidate = {
            key: result.get("dataset", {}).get(key) for key in dataset_keys
        }
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
        source_links = (
            f"[manifest]({sources['manifest']}), [config]({sources['config']}), "
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
    prefix = (
        "# Model comparison\n\n"
        + (
            "> These results include smoke runs. They validate the workflow and must not be used as a final quality ranking.\n\n"
            if report["interpretation"] == "workflow_validation_only"
            else "> Final experiment results. Interpret them with the recorded dataset, seeds, and hardware.\n\n"
        )
    )
    return report, prefix + "\n".join(lines) + "\n"
