"""Serialize monitoring reports."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from .schema import MonitoringReport


def _rows(report: MonitoringReport) -> list[dict[str, Any]]:
    common = {
        "schema_version": report.schema_version,
        "suite_id": report.suite["id"],
        "suite_fingerprint": report.suite["fingerprint"],
        "artifact_type": report.artifact["type"],
        "artifact_sha256": report.artifact["sha256"],
        "inference_category": report.artifact["inference_category"],
        "effective_inference_chunk_size": report.artifact[
            "effective_inference_chunk_size"
        ],
        "model_type": report.artifact["model_type"],
        "device_class": report.runtime["device_class"],
        "device_name": report.runtime["device_name"],
        "silence_policy_id": report.workload["silence_policy"]["id"],
    }
    counts = report.aggregate.get("relative_score_counts", {})
    rows = [
        {
            **common,
            "scope": "aggregate",
            "case_id": "",
            "relative_score_status": "aggregate",
            "digital_silence": "",
            "silent_case_count": report.aggregate.get("silent_case_count", 0),
            "total_case_count": len(report.cases),
            **report.aggregate["metrics"],
            **{
                f"{metric}_eligible_count": values["eligible"]
                for metric, values in counts.items()
            },
            **{
                f"{metric}_excluded_count": values["excluded"]
                for metric, values in counts.items()
            },
        }
    ]
    for case in report.cases:
        full_latency = case.latency["full"]
        rows.append(
            {
                **common,
                "scope": "case",
                "case_id": case.case_id,
                **case.metrics,
                **case.diagnostics,
                "p50_latency_ms": full_latency["p50_latency_ms"],
                "p95_latency_ms": full_latency["p95_latency_ms"],
                "real_time_factor": full_latency["real_time_factor"],
                "peak_memory_bytes": report.aggregate["metrics"][
                    "peak_memory_bytes"
                ],
                "artifact_size_bytes": report.aggregate["metrics"][
                    "artifact_size_bytes"
                ],
                "silent_case_count": "",
                "total_case_count": "",
            }
        )
    return rows


def _write_csv(report: MonitoringReport, path: Path) -> None:
    rows = _rows(report)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(report: MonitoringReport, path: Path) -> None:
    def display(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.8g}"
        return str(value)

    metric_rows = "".join(
        f"<tr><th>{html.escape(name)}</th>"
        f"<td>{html.escape(display(value))}</td></tr>"
        for name, value in report.aggregate["metrics"].items()
    )
    count_rows = "".join(
        f"<tr><th>{html.escape(name)}</th>"
        f"<td>{html.escape(str(values['eligible']))}</td>"
        f"<td>{html.escape(str(values['excluded']))}</td></tr>"
        for name, values in report.aggregate.get("relative_score_counts", {}).items()
    )
    case_rows = "".join(
        f"<tr><td>{html.escape(case.case_id)}</td>"
        f"<td>{html.escape(case.diagnostics['relative_score_status'])}</td>"
        f"<td>{html.escape(str(case.diagnostics['digital_silence']))}</td>"
        f"<td>{html.escape('; '.join(f'{name}={display(value)}' for name, value in {**case.diagnostics, **case.metrics}.items()))}</td></tr>"
        for case in report.cases
    )
    policy = html.escape(json.dumps(report.workload["silence_policy"], sort_keys=True))
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>neural-fx monitoring</title></head>
<body>
<h1>neural-fx offline monitoring</h1>
<p>Suite <code>{html.escape(report.suite['id'])}</code></p>
<p>Artifact <code>{html.escape(report.artifact['sha256'])}</code></p>
<h2>Silence policy</h2><pre>{policy}</pre>
<h2>Aggregate metrics</h2><table>{metric_rows}</table>
<h2>Relative-score counts</h2><table><tr><th>Metric</th><th>Eligible</th><th>Excluded</th></tr>{count_rows}</table>
<h2>Cases</h2><table><tr><th>Case</th><th>Status</th><th>Digital silence</th><th>Diagnostics and metrics</th></tr>{case_rows}</table>
</body>
</html>
"""
    )


def write_monitoring_outputs(
    report: MonitoringReport,
    output_dir: str | Path,
    *,
    include_html: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write JSON, CSV, and optional HTML from one monitoring report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "monitoring.json",
        "csv": output_dir / "monitoring.csv",
    }
    if include_html:
        paths["html"] = output_dir / "monitoring.html"
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Monitoring output already exists: {existing[0]}")
    paths["json"].write_text(
        json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n"
    )
    _write_csv(report, paths["csv"])
    if include_html:
        _write_html(report, paths["html"])
    return paths
