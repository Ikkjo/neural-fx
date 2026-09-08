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
        "model_type": report.artifact["model_type"],
        "device_class": report.runtime["device_class"],
        "device_name": report.runtime["device_name"],
    }
    rows = [
        {
            **common,
            "scope": "aggregate",
            "case_id": "",
            **report.aggregate["metrics"],
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
                "p50_latency_ms": full_latency["p50_latency_ms"],
                "p95_latency_ms": full_latency["p95_latency_ms"],
                "real_time_factor": full_latency["real_time_factor"],
                "peak_memory_bytes": report.aggregate["metrics"][
                    "peak_memory_bytes"
                ],
                "artifact_size_bytes": report.aggregate["metrics"][
                    "artifact_size_bytes"
                ],
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
    metric_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{value:.8g}</td></tr>"
        for name, value in report.aggregate["metrics"].items()
    )
    case_rows = "".join(
        f"<tr><td>{html.escape(case.case_id)}</td>"
        f"<td>{html.escape(json.dumps(case.metrics))}</td></tr>"
        for case in report.cases
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>neural-fx monitoring</title></head>
<body>
<h1>neural-fx offline monitoring</h1>
<p>Suite <code>{html.escape(report.suite['id'])}</code></p>
<p>Artifact <code>{html.escape(report.artifact['sha256'])}</code></p>
<h2>Aggregate metrics</h2><table>{metric_rows}</table>
<h2>Cases</h2><table>{case_rows}</table>
</body>
</html>
"""
    )


def write_monitoring_outputs(
    report: MonitoringReport,
    output_dir: str | Path,
    *,
    include_html: bool = False,
) -> dict[str, Path]:
    """Write JSON, CSV, and optional HTML from one monitoring report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "monitoring.json",
        "csv": output_dir / "monitoring.csv",
    }
    paths["json"].write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    _write_csv(report, paths["csv"])
    if include_html:
        paths["html"] = output_dir / "monitoring.html"
        _write_html(report, paths["html"])
    return paths
