"""Reproducible preparation of aligned audio pairs for fixed experiments."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .latency import LatencyCalibrator

MANIFEST_NAME = "dataset-manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SplitSpec:
    """A half-open region in the prepared, target-rate timeline."""

    name: str
    start_sample: int
    num_samples: int
    write_audio: bool = True

    @property
    def end_sample(self) -> int:
        """Return the exclusive end sample."""
        return self.start_sample + self.num_samples


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_float_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    """Write an uncompressed little-endian IEEE float32 WAV."""
    if audio.ndim != 2:
        raise ValueError("WAV audio must have shape (channels, samples)")
    channels = int(audio.shape[0])
    interleaved = (
        audio.detach()
        .to(device="cpu", dtype=torch.float32)
        .transpose(0, 1)
        .contiguous()
    )
    payload = interleaved.numpy().tobytes()
    block_align = channels * 4
    byte_rate = sample_rate * block_align
    header = b"".join(
        (
            b"RIFF",
            struct.pack("<I", 36 + len(payload)),
            b"WAVEfmt ",
            struct.pack(
                "<IHHIIHH", 16, 3, channels, sample_rate, byte_rate, block_align, 32
            ),
            b"data",
            struct.pack("<I", len(payload)),
        )
    )
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(payload)


def _validate_splits(splits: tuple[SplitSpec, ...]) -> None:
    if not splits:
        raise ValueError("At least one split is required")

    previous_end = 0
    names: set[str] = set()
    for split in splits:
        if split.name in names:
            raise ValueError(f"Duplicate split name: {split.name}")
        if split.start_sample < previous_end:
            raise ValueError(f"Split {split.name} overlaps the preceding split")
        if split.start_sample < 0 or split.num_samples <= 0:
            raise ValueError(
                f"Split {split.name} must have non-negative start and positive length"
            )
        names.add(split.name)
        previous_end = split.end_sample


def _signal_stats(audio: torch.Tensor) -> dict[str, float]:
    peak = float(audio.abs().max()) if audio.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(audio.square()))) if audio.numel() else 0.0
    return {"peak": peak, "rms": rms}


def _relative_gain(
    input_audio: torch.Tensor, target_audio: torch.Tensor
) -> dict[str, float | None]:
    input_stats = _signal_stats(input_audio)
    target_stats = _signal_stats(target_audio)
    peak_ratio = (
        target_stats["peak"] / input_stats["peak"] if input_stats["peak"] else None
    )
    rms_ratio = target_stats["rms"] / input_stats["rms"] if input_stats["rms"] else None
    return {
        "input_peak": input_stats["peak"],
        "target_peak": target_stats["peak"],
        "target_to_input_peak_ratio": peak_ratio,
        "input_rms": input_stats["rms"],
        "target_rms": target_stats["rms"],
        "target_to_input_rms_ratio": rms_ratio,
    }


def _aggregate_zero_lag_correlation(
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
) -> dict[str, Any]:
    """Measure fixed-window normalized correlation after alignment."""
    window_samples = 10 * sample_rate
    interval_samples = 30 * sample_rate
    first_sample = 30 * sample_rate
    starts = list(
        range(
            first_sample, input_audio.shape[-1] - window_samples + 1, interval_samples
        )
    )
    if not starts:
        starts = [0]
        window_samples = min(window_samples, input_audio.shape[-1])

    correlations: list[float] = []
    for start in starts:
        input_window = input_audio[..., start : start + window_samples]
        target_window = target_audio[..., start : start + window_samples]
        denominator = torch.linalg.vector_norm(input_window) * torch.linalg.vector_norm(
            target_window
        )
        correlation = (
            float(torch.sum(input_window * target_window) / denominator)
            if denominator
            else 0.0
        )
        correlations.append(correlation)
    return {
        "lag_samples": 0,
        "window_samples": window_samples,
        "window_start_samples": starts,
        "signed_mean_normalized_correlation": sum(correlations) / len(correlations),
        "absolute_mean_normalized_correlation": sum(
            abs(value) for value in correlations
        )
        / len(correlations),
    }


def _artifact_metadata(
    path: Path, root: Path, audio: torch.Tensor, sample_rate: int
) -> dict[str, Any]:
    stats = _signal_stats(audio)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "sample_rate": sample_rate,
        "channels": int(audio.shape[0]),
        "samples": int(audio.shape[-1]),
        "duration_seconds": audio.shape[-1] / sample_rate,
        "encoding": "IEEE_FLOAT",
        "bits_per_sample": 32,
        **stats,
    }


def _preparation_spec(
    input_hash: str,
    target_hash: str,
    target_sample_rate: int,
    target_delay_source_samples: int,
    splits: tuple[SplitSpec, ...],
    experiment_id: str | None,
) -> dict[str, Any]:
    return {
        "input_sha256": input_hash,
        "target_sha256": target_hash,
        "target_sample_rate": target_sample_rate,
        "target_delay_source_samples": target_delay_source_samples,
        "splits": [asdict(split) for split in splits],
        "normalization": "none",
        "experiment_id": experiment_id,
    }


def _reuse_existing(output_dir: Path, desired_spec: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileExistsError(f"Output exists without {MANIFEST_NAME}: {output_dir}")

    with manifest_path.open() as handle:
        manifest = json.load(handle)

    if manifest.get("preparation_spec") != desired_spec:
        raise FileExistsError(
            f"Output was prepared with a different specification: {output_dir}"
        )

    expected_artifacts = {
        f"{split['name']}_{role}"
        for split in desired_spec["splits"]
        if split["write_audio"]
        for role in ("input", "target")
    }
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != expected_artifacts:
        raise ValueError("Prepared manifest does not list the expected split artifacts")
    for artifact in artifacts.values():
        artifact_path = output_dir / artifact["path"]
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != artifact["sha256"]
        ):
            raise ValueError(
                f"Prepared artifact is missing or changed: {artifact_path}"
            )
    return manifest


def prepare_aligned_audio(
    input_path: Path,
    target_path: Path,
    output_dir: Path,
    *,
    splits: tuple[SplitSpec, ...],
    target_sample_rate: int = 48_000,
    target_delay_source_samples: int = 0,
    expected_input_sha256: str | None = None,
    expected_target_sha256: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Prepare one immutable aligned, resampled, and split audio-pair dataset.

    A positive delay means that the source target lags the source input. The
    target is advanced by that many source-rate samples before both signals are
    resampled exactly once. Existing output is reused only when its complete
    specification and every artifact hash still match.
    """
    input_path = input_path.expanduser().resolve()
    target_path = target_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {input_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"Target audio does not exist: {target_path}")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")
    _validate_splits(splits)

    input_hash = sha256_file(input_path)
    target_hash = sha256_file(target_path)
    if expected_input_sha256 is not None and input_hash != expected_input_sha256:
        raise ValueError("Input source hash does not match the expected recording")
    if expected_target_sha256 is not None and target_hash != expected_target_sha256:
        raise ValueError("Target source hash does not match the expected recording")
    desired_spec = _preparation_spec(
        input_hash,
        target_hash,
        target_sample_rate,
        target_delay_source_samples,
        splits,
        experiment_id,
    )
    if output_dir.exists():
        return _reuse_existing(output_dir, desired_spec)

    input_audio, input_rate = torchaudio.load(input_path)
    target_audio, target_rate = torchaudio.load(target_path)
    if input_audio.shape[0] != 1 or target_audio.shape[0] != 1:
        raise ValueError("Audio preparation requires mono input and target audio")
    if input_rate != target_rate:
        raise ValueError(
            f"Source sample rates differ: input={input_rate}, target={target_rate}"
        )
    if input_audio.shape[-1] != target_audio.shape[-1]:
        raise ValueError(
            f"Source sample counts differ: input={input_audio.shape[-1]}, target={target_audio.shape[-1]}"
        )
    if not torch.isfinite(input_audio).all() or not torch.isfinite(target_audio).all():
        raise ValueError("Source audio contains NaN or infinite samples")
    if abs(target_delay_source_samples) >= input_audio.shape[-1]:
        raise ValueError("Alignment delay consumes the entire source recording")

    source_samples = int(input_audio.shape[-1])
    input_source_stats = _signal_stats(input_audio)
    target_source_stats = _signal_stats(target_audio)
    if target_delay_source_samples:
        input_audio, target_audio = LatencyCalibrator().apply_delay(
            input_audio,
            target_audio,
            target_delay_source_samples,
        )

    if input_rate != target_sample_rate:
        input_audio = torchaudio.functional.resample(
            input_audio, input_rate, target_sample_rate
        )
        target_audio = torchaudio.functional.resample(
            target_audio, target_rate, target_sample_rate
        )
    if input_audio.shape[-1] != target_audio.shape[-1]:
        raise RuntimeError("Resampling produced unequal input and target lengths")

    required_samples = max(split.end_sample for split in splits)
    if input_audio.shape[-1] < required_samples:
        raise ValueError(
            f"Prepared audio has {input_audio.shape[-1]} samples; split plan needs {required_samples}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "payload"
        staging.mkdir()
        artifacts: dict[str, dict[str, Any]] = {}
        split_manifest: dict[str, dict[str, Any]] = {}

        for split in splits:
            split_input = input_audio[..., split.start_sample : split.end_sample]
            split_target = target_audio[..., split.start_sample : split.end_sample]
            split_record: dict[str, Any] = {
                "start_sample": split.start_sample,
                "end_sample": split.end_sample,
                "num_samples": split.num_samples,
                "duration_seconds": split.num_samples / target_sample_rate,
                "write_audio": split.write_audio,
                "relative_gain": _relative_gain(split_input, split_target),
            }
            if split.write_audio:
                split_dir = staging / "data" / split.name
                split_dir.mkdir(parents=True)
                for role, audio in (("input", split_input), ("target", split_target)):
                    path = split_dir / f"{role}.wav"
                    _write_float_wav(path, audio, target_sample_rate)
                    key = f"{split.name}_{role}"
                    artifacts[key] = _artifact_metadata(
                        path, staging, audio, target_sample_rate
                    )
                    split_record[f"{role}_artifact"] = key
            split_manifest[split.name] = split_record

        aligned_samples = int(input_audio.shape[-1])
        last_end = max(split.end_sample for split in splits)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "preparation_spec": desired_spec,
            "sources": {
                "input": {
                    "path": str(input_path),
                    "sha256": input_hash,
                    "sample_rate": input_rate,
                    "channels": 1,
                    "samples": source_samples,
                    "duration_seconds": source_samples / input_rate,
                    **input_source_stats,
                },
                "target": {
                    "path": str(target_path),
                    "sha256": target_hash,
                    "sample_rate": target_rate,
                    "channels": 1,
                    "samples": source_samples,
                    "duration_seconds": source_samples / target_rate,
                    **target_source_stats,
                },
            },
            "alignment": {
                "method": "fixed_aggregate_cross_correlation",
                "target_delay_source_samples": target_delay_source_samples,
                "operation": (
                    "trim input tail; trim target head"
                    if target_delay_source_samples > 0
                    else "trim input head; trim target tail"
                    if target_delay_source_samples < 0
                    else "none"
                ),
                "polarity": "preserved",
            },
            "resampling": {
                "implementation": "torchaudio.functional.resample",
                "torchaudio_version": torchaudio.__version__,
                "source_sample_rate": input_rate,
                "target_sample_rate": target_sample_rate,
                "passes_per_signal": 0 if input_rate == target_sample_rate else 1,
            },
            "prepared_timeline": {
                "samples": aligned_samples,
                "duration_seconds": aligned_samples / target_sample_rate,
                "unused_tail_start_sample": last_end,
                "unused_tail_samples": aligned_samples - last_end,
            },
            "normalization": "none",
            "relative_gain": _relative_gain(input_audio, target_audio),
            "alignment_sanity": _aggregate_zero_lag_correlation(
                input_audio, target_audio, target_sample_rate
            ),
            "splits": split_manifest,
            "artifacts": artifacts,
        }
        manifest_path = staging / MANIFEST_NAME
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.rename(staging, output_dir)

    return manifest
