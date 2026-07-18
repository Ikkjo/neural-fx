"""Tests for comparison manifest validation, aggregation, and reports."""

from __future__ import annotations

import csv
import json
import math

import pytest
import yaml

from neural_fx.evaluation import (
    ExperimentManifest,
    ManifestValidationError,
    RunResult,
    aggregate_results,
    load_manifest,
    load_run_results,
    write_reports,
)


def synthetic_manifest_dict() -> dict:
    """Return a deliberately synthetic two-model, three-seed experiment."""
    models = []
    parameter_counts = {"lstm": 59_500, "gru": 60_500}
    for architecture in ("lstm", "gru"):
        for seed in (42, 43, 44):
            models.append(
                {
                    "id": f"{architecture}-medium-{seed}",
                    "architecture": architecture,
                    "seed": seed,
                    "config": f"configs/{architecture}.yaml",
                    "checkpoint": f"checkpoints/{architecture}-{seed}.ckpt",
                    "size_label": "medium",
                    "parameter_count": parameter_counts[architecture],
                }
            )
    return {
        "schema_version": "1.0",
        "name": "Synthetic comparison fixture",
        "dataset": {
            "train": [
                {"input": "audio/train-in.wav", "target": "audio/train-out.wav"}
            ],
            "validation": [
                {
                    "input": "audio/validation-in.wav",
                    "target": "audio/validation-out.wav",
                }
            ],
            "test": [{"input": "audio/test-in.wav", "target": "audio/test-out.wav"}],
        },
        "parameter_budget": {"target": 60_000, "tolerance": 0.10},
        "seeds": [42, 43, 44],
        "shared_training": {
            "sample_rate": 48_000,
            "segment_length": 8192,
            "optimizer": "adam",
            "epochs": 100,
        },
        "models": models,
    }


def synthetic_records() -> tuple[RunResult, ...]:
    records = []
    for architecture, offset in (("lstm", 0.0), ("gru", 0.1)):
        for seed, esr in zip((42, 43, 44), (0.1, 0.2, 0.3)):
            records.append(
                RunResult(
                    run_id=f"{architecture}-medium-{seed}",
                    metrics={"esr": esr + offset, "mse": (esr + offset) / 2},
                )
            )
    return tuple(records)


def test_manifest_accepts_complete_synthetic_experiment():
    manifest = ExperimentManifest.from_dict(synthetic_manifest_dict())

    assert manifest.name == "Synthetic comparison fixture"
    assert manifest.seeds == (42, 43, 44)
    assert len(manifest.models) == 6
    assert manifest.parameter_budget.minimum == 54_000
    assert manifest.parameter_budget.maximum == 66_000


@pytest.mark.parametrize(
    ("parameter_count", "is_valid"),
    [(53_999, False), (54_000, True), (66_000, True), (66_001, False)],
)
def test_parameter_budget_boundaries(parameter_count, is_valid):
    data = synthetic_manifest_dict()
    data["models"][0]["parameter_count"] = parameter_count

    if is_valid:
        ExperimentManifest.from_dict(data)
    else:
        with pytest.raises(ManifestValidationError, match="parameters; expected"):
            ExperimentManifest.from_dict(data)


def test_manifest_rejects_data_leakage_between_splits():
    data = synthetic_manifest_dict()
    data["dataset"]["test"][0]["input"] = data["dataset"]["train"][0]["input"]

    with pytest.raises(ManifestValidationError, match="is reused"):
        ExperimentManifest.from_dict(data)


def test_manifest_rejects_missing_seed_for_architecture():
    data = synthetic_manifest_dict()
    data["models"] = [
        model for model in data["models"] if model["id"] != "gru-medium-44"
    ]

    with pytest.raises(ManifestValidationError, match=r"missing=\[44\]"):
        ExperimentManifest.from_dict(data)


def test_manifest_rejects_duplicate_run_id():
    data = synthetic_manifest_dict()
    data["models"][1]["id"] = data["models"][0]["id"]

    with pytest.raises(ManifestValidationError, match="Duplicate model run id"):
        ExperimentManifest.from_dict(data)


def test_manifest_rejects_parameter_count_that_changes_between_seeds():
    data = synthetic_manifest_dict()
    data["models"][1]["parameter_count"] = 60_000

    with pytest.raises(ManifestValidationError, match="inconsistent parameter counts"):
        ExperimentManifest.from_dict(data)


def test_load_manifest_can_require_files(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(synthetic_manifest_dict()), encoding="utf-8"
    )

    load_manifest(manifest_path)
    with pytest.raises(ManifestValidationError, match="audio file does not exist"):
        load_manifest(manifest_path, require_files=True)


def test_load_results_validates_schema_and_finite_metrics(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "records": [
                    {"run_id": "lstm-medium-42", "metrics": {"esr": math.nan}}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="finite number"):
        load_run_results(results_path)


def test_aggregation_computes_mean_and_sample_std():
    manifest = ExperimentManifest.from_dict(synthetic_manifest_dict())

    aggregates = aggregate_results(manifest, synthetic_records())

    assert [result.architecture for result in aggregates] == ["gru", "lstm"]
    lstm = aggregates[1]
    assert lstm.seeds == (42, 43, 44)
    assert lstm.parameter_count_mean == 59_500
    assert lstm.metrics["esr"].mean == pytest.approx(0.2)
    assert lstm.metrics["esr"].std == pytest.approx(0.1)


def test_aggregation_requires_exactly_one_result_per_run():
    manifest = ExperimentManifest.from_dict(synthetic_manifest_dict())

    with pytest.raises(ManifestValidationError, match="Missing results"):
        aggregate_results(manifest, synthetic_records()[:-1])
    with pytest.raises(ManifestValidationError, match="Duplicate result"):
        aggregate_results(
            manifest, (*synthetic_records(), synthetic_records()[0])
        )


def test_aggregation_rejects_inconsistent_metrics():
    manifest = ExperimentManifest.from_dict(synthetic_manifest_dict())
    records = list(synthetic_records())
    records[1] = RunResult(records[1].run_id, {"esr": 0.2})

    with pytest.raises(ManifestValidationError, match="inconsistent metric names"):
        aggregate_results(manifest, records)


def test_all_report_formats_are_written(tmp_path):
    manifest = ExperimentManifest.from_dict(synthetic_manifest_dict())
    aggregates = aggregate_results(manifest, synthetic_records())

    paths = write_reports(manifest, aggregates, tmp_path)

    assert set(paths) == {"json", "csv", "markdown", "html"}
    assert all(path.is_file() for path in paths.values())

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["aggregation"]["std"].startswith("sample")
    assert len(payload["results"]) == 2

    with paths["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["architecture"] == "gru"
    assert "esr_mean" in rows[0]
    assert "esr_std" in rows[0]

    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "Synthetic fixtures exercise this harness" in markdown
    assert "0.2 ± 0.1" in markdown

    html = paths["html"].read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "does not imply real project evaluation results" in html
