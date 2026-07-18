"""JSON, CSV, Markdown, and HTML comparison report writers."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Sequence

from .manifest import SCHEMA_VERSION, ExperimentManifest
from .results import AggregateResult


def _metric_names(results: Sequence[AggregateResult]) -> list[str]:
    names: set[str] = set()
    for result in results:
        names.update(result.metrics)
    return sorted(names)


def _report_payload(
    manifest: ExperimentManifest, results: Sequence[AggregateResult]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": manifest.name,
        "parameter_budget": manifest.parameter_budget.to_dict(),
        "aggregation": {
            "unit": "seed",
            "mean": "arithmetic",
            "std": "sample (n-1); zero for one seed",
        },
        "results": [result.to_dict() for result in results],
    }


def write_json_report(
    manifest: ExperimentManifest,
    results: Sequence[AggregateResult],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_report_payload(manifest, results), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def write_csv_report(
    manifest: ExperimentManifest,
    results: Sequence[AggregateResult],
    output_path: str | Path,
) -> Path:
    del manifest  # Kept in the common writer signature for callers.
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = _metric_names(results)
    fieldnames = ["architecture", "size_label", "parameter_count_mean", "seeds"]
    for metric in metrics:
        fieldnames.extend((f"{metric}_mean", f"{metric}_std"))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: dict[str, Any] = {
                "architecture": result.architecture,
                "size_label": result.size_label,
                "parameter_count_mean": result.parameter_count_mean,
                "seeds": ";".join(map(str, result.seeds)),
            }
            for metric in metrics:
                summary = result.metrics.get(metric)
                row[f"{metric}_mean"] = "" if summary is None else summary.mean
                row[f"{metric}_std"] = "" if summary is None else summary.std
            writer.writerow(row)
    return output


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def write_markdown_report(
    manifest: ExperimentManifest,
    results: Sequence[AggregateResult],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = _metric_names(results)
    headings = ["Architecture", "Size", "Parameters", "Seeds", *metrics]
    lines = [
        f"# {manifest.name}",
        "",
        "Synthetic fixtures exercise this harness; this report does not imply "
        "real project evaluation results.",
        "",
        " | ".join(f"{heading}" for heading in headings),
        " | ".join("---" for _ in headings),
    ]
    for result in results:
        cells = [
            result.architecture,
            result.size_label,
            _format_number(result.parameter_count_mean),
            ", ".join(map(str, result.seeds)),
        ]
        for metric in metrics:
            summary = result.metrics.get(metric)
            cells.append(
                "—"
                if summary is None
                else f"{_format_number(summary.mean)} ± {_format_number(summary.std)}"
            )
        lines.append(" | ".join(cells))
    lines.extend(
        (
            "",
            "Values are arithmetic mean ± sample standard deviation across seeds.",
            "",
        )
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_html_report(
    manifest: ExperimentManifest,
    results: Sequence[AggregateResult],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = _metric_names(results)
    headings = ["Architecture", "Size", "Parameters", "Seeds", *metrics]
    rows: list[str] = []
    for result in results:
        cells = [
            result.architecture,
            result.size_label,
            _format_number(result.parameter_count_mean),
            ", ".join(map(str, result.seeds)),
        ]
        for metric in metrics:
            summary = result.metrics.get(metric)
            cells.append(
                "—"
                if summary is None
                else f"{_format_number(summary.mean)} ± {_format_number(summary.std)}"
            )
        rows.append(
            "      <tr>"
            + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells)
            + "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(manifest.name)}</title>
  <style>
    body {{
      font: 16px/1.5 system-ui, sans-serif;
      margin: 2rem auto;
      max-width: 72rem;
      padding: 0 1rem;
    }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #bbb; padding: .5rem; text-align: right; }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    caption {{ font-weight: 600; margin-bottom: .75rem; text-align: left; }}
  </style>
</head>
<body>
  <h1>{html.escape(manifest.name)}</h1>
  <p>
    Synthetic fixtures exercise this harness; this page does not imply real
    project evaluation results.
  </p>
  <table>
    <caption>Arithmetic mean ± sample standard deviation across seeds</caption>
    <thead>
      <tr>{''.join(f'<th>{html.escape(item)}</th>' for item in headings)}</tr>
    </thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return output


def write_reports(
    manifest: ExperimentManifest,
    results: Sequence[AggregateResult],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write every supported report format to one output directory."""
    destination = Path(output_dir)
    return {
        "json": write_json_report(manifest, results, destination / "comparison.json"),
        "csv": write_csv_report(manifest, results, destination / "comparison.csv"),
        "markdown": write_markdown_report(
            manifest, results, destination / "comparison.md"
        ),
        "html": write_html_report(manifest, results, destination / "comparison.html"),
    }
