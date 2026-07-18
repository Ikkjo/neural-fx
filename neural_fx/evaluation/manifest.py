"""Versioned experiment manifests for fair model comparisons."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "1.0"


class ManifestValidationError(ValueError):
    """Raised when an experiment manifest is incomplete or inconsistent."""


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field_name} must be a mapping")
    return value


def _require_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManifestValidationError(f"{field_name} must be a list")
    return value


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class AudioPair:
    """One aligned input/target audio pair."""

    input: str
    target: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], field_name: str) -> AudioPair:
        return cls(
            input=_required_text(data.get("input"), f"{field_name}.input"),
            target=_required_text(data.get("target"), f"{field_name}.target"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"input": self.input, "target": self.target}


@dataclass(frozen=True)
class DatasetSplits:
    """Explicit, non-overlapping train, validation, and test data."""

    train: tuple[AudioPair, ...]
    validation: tuple[AudioPair, ...]
    test: tuple[AudioPair, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetSplits:
        def load_split(name: str) -> tuple[AudioPair, ...]:
            raw = _require_sequence(data.get(name), f"dataset.{name}")
            if not raw:
                raise ManifestValidationError(f"dataset.{name} must not be empty")
            return tuple(
                AudioPair.from_dict(
                    _require_mapping(item, f"dataset.{name}[{index}]"),
                    f"dataset.{name}[{index}]",
                )
                for index, item in enumerate(raw)
            )

        return cls(
            train=load_split("train"),
            validation=load_split("validation"),
            test=load_split("test"),
        )

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {
            "train": [pair.to_dict() for pair in self.train],
            "validation": [pair.to_dict() for pair in self.validation],
            "test": [pair.to_dict() for pair in self.test],
        }


@dataclass(frozen=True)
class ParameterBudget:
    """Parameter-count target and allowed fractional deviation."""

    target: int = 60_000
    tolerance: float = 0.10

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParameterBudget:
        target = data.get("target", 60_000)
        tolerance = data.get("tolerance", 0.10)
        if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
            raise ManifestValidationError("parameter_budget.target must be positive")
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not math.isfinite(tolerance)
            or not 0 <= tolerance < 1
        ):
            raise ManifestValidationError(
                "parameter_budget.tolerance must be in the range [0, 1)"
            )
        return cls(target=target, tolerance=float(tolerance))

    @property
    def minimum(self) -> int:
        return math.ceil(self.target * (1 - self.tolerance))

    @property
    def maximum(self) -> int:
        return math.floor(self.target * (1 + self.tolerance))

    def contains(self, parameter_count: int) -> bool:
        return self.minimum <= parameter_count <= self.maximum

    def to_dict(self) -> dict[str, int | float]:
        return {"target": self.target, "tolerance": self.tolerance}


@dataclass(frozen=True)
class ModelRun:
    """One architecture/seed run participating in the comparison."""

    id: str
    architecture: str
    seed: int
    config: str
    checkpoint: str
    size_label: str
    parameter_count: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], index: int) -> ModelRun:
        prefix = f"models[{index}]"
        seed = data.get("seed")
        parameter_count = data.get("parameter_count")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ManifestValidationError(
                f"{prefix}.seed must be a non-negative integer"
            )
        if (
            not isinstance(parameter_count, int)
            or isinstance(parameter_count, bool)
            or parameter_count <= 0
        ):
            raise ManifestValidationError(
                f"{prefix}.parameter_count must be a positive integer"
            )
        return cls(
            id=_required_text(data.get("id"), f"{prefix}.id"),
            architecture=_required_text(
                data.get("architecture"), f"{prefix}.architecture"
            ),
            seed=seed,
            config=_required_text(data.get("config"), f"{prefix}.config"),
            checkpoint=_required_text(
                data.get("checkpoint"), f"{prefix}.checkpoint"
            ),
            size_label=_required_text(
                data.get("size_label"), f"{prefix}.size_label"
            ),
            parameter_count=parameter_count,
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "id": self.id,
            "architecture": self.architecture,
            "seed": self.seed,
            "config": self.config,
            "checkpoint": self.checkpoint,
            "size_label": self.size_label,
            "parameter_count": self.parameter_count,
        }


@dataclass(frozen=True)
class ExperimentManifest:
    """Complete declaration of a reproducible comparison experiment."""

    name: str
    dataset: DatasetSplits
    models: tuple[ModelRun, ...]
    seeds: tuple[int, ...]
    parameter_budget: ParameterBudget = field(default_factory=ParameterBudget)
    shared_training: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentManifest:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ManifestValidationError(
                f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}"
            )

        raw_models = _require_sequence(data.get("models"), "models")
        if not raw_models:
            raise ManifestValidationError("models must not be empty")
        models = tuple(
            ModelRun.from_dict(_require_mapping(item, f"models[{index}]"), index)
            for index, item in enumerate(raw_models)
        )

        raw_seeds = _require_sequence(data.get("seeds"), "seeds")
        seeds: list[int] = []
        for index, seed in enumerate(raw_seeds):
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ManifestValidationError(
                    f"seeds[{index}] must be a non-negative integer"
                )
            seeds.append(seed)
        if not seeds:
            raise ManifestValidationError("seeds must not be empty")

        manifest = cls(
            name=_required_text(data.get("name"), "name"),
            dataset=DatasetSplits.from_dict(
                _require_mapping(data.get("dataset"), "dataset")
            ),
            models=models,
            seeds=tuple(seeds),
            parameter_budget=ParameterBudget.from_dict(
                _require_mapping(
                    data.get("parameter_budget", {}), "parameter_budget"
                )
            ),
            shared_training=dict(
                _require_mapping(data.get("shared_training", {}), "shared_training")
            ),
            schema_version=version,
        )
        manifest.validate()
        return manifest

    def validate(
        self, base_dir: Path | None = None, require_files: bool = False
    ) -> None:
        """Validate split isolation, run coverage, and the parameter budget."""
        if len(set(self.seeds)) != len(self.seeds):
            raise ManifestValidationError("seeds must not contain duplicates")

        seen_paths: dict[str, str] = {}
        for split_name in ("train", "validation", "test"):
            pairs = getattr(self.dataset, split_name)
            for index, pair in enumerate(pairs):
                for role, raw_path in (("input", pair.input), ("target", pair.target)):
                    path_key = _normalized_path(raw_path, base_dir)
                    location = f"dataset.{split_name}[{index}].{role}"
                    previous = seen_paths.get(path_key)
                    if previous is not None:
                        raise ManifestValidationError(
                            f"Audio path {raw_path!r} is reused by {previous} "
                            f"and {location}"
                        )
                    seen_paths[path_key] = location
                    if (
                        require_files
                        and not _resolve_path(raw_path, base_dir).is_file()
                    ):
                        raise ManifestValidationError(
                            f"Required audio file does not exist: {raw_path}"
                        )

        run_ids: set[str] = set()
        run_keys: set[tuple[str, str, int]] = set()
        architecture_seeds: dict[tuple[str, str], set[int]] = {}
        expected_seeds = set(self.seeds)
        for run in self.models:
            if run.id in run_ids:
                raise ManifestValidationError(f"Duplicate model run id: {run.id}")
            run_ids.add(run.id)
            run_key = (run.architecture, run.size_label, run.seed)
            if run_key in run_keys:
                key_text = "/".join(map(str, run_key))
                raise ManifestValidationError(
                    f"Duplicate architecture/size/seed run: {key_text}"
                )
            run_keys.add(run_key)
            architecture_seeds.setdefault(
                (run.architecture, run.size_label), set()
            ).add(run.seed)
            if run.seed not in expected_seeds:
                raise ManifestValidationError(
                    f"Run {run.id!r} uses undeclared seed {run.seed}"
                )
            if not self.parameter_budget.contains(run.parameter_count):
                raise ManifestValidationError(
                    f"Run {run.id!r} has {run.parameter_count} parameters; expected "
                    f"{self.parameter_budget.minimum}..{self.parameter_budget.maximum}"
                )
            if require_files:
                for field_name, raw_path in (
                    ("config", run.config),
                    ("checkpoint", run.checkpoint),
                ):
                    if not _resolve_path(raw_path, base_dir).is_file():
                        raise ManifestValidationError(
                            f"Run {run.id!r} {field_name} file does not exist: "
                            f"{raw_path}"
                        )

        for (architecture, size_label), actual_seeds in architecture_seeds.items():
            if actual_seeds != expected_seeds:
                missing = sorted(expected_seeds - actual_seeds)
                extra = sorted(actual_seeds - expected_seeds)
                raise ManifestValidationError(
                    f"Architecture {architecture!r} size {size_label!r} does not cover "
                    f"the declared seeds (missing={missing}, extra={extra})"
                )
            parameter_counts = {
                run.parameter_count
                for run in self.models
                if run.architecture == architecture and run.size_label == size_label
            }
            if len(parameter_counts) != 1:
                raise ManifestValidationError(
                    f"Architecture {architecture!r} size {size_label!r} has "
                    f"inconsistent parameter counts: {sorted(parameter_counts)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "dataset": self.dataset.to_dict(),
            "parameter_budget": self.parameter_budget.to_dict(),
            "seeds": list(self.seeds),
            "shared_training": dict(self.shared_training),
            "models": [run.to_dict() for run in self.models],
        }


def _resolve_path(raw_path: str, base_dir: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve(strict=False)


def _normalized_path(raw_path: str, base_dir: Path | None) -> str:
    return str(_resolve_path(raw_path, base_dir)).casefold()


def load_manifest(
    path: str | Path, require_files: bool = False
) -> ExperimentManifest:
    """Load a JSON or YAML manifest and resolve validation relative to its folder."""
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        if manifest_path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            raw = yaml.safe_load(handle)
    manifest = ExperimentManifest.from_dict(_require_mapping(raw, "manifest"))
    manifest.validate(base_dir=manifest_path.parent, require_files=require_files)
    return manifest
