#!/usr/bin/env python3
"""Compare two offline monitoring reports against a fixed policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_fx.monitoring import MonitoringError, compare_monitoring_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    paths = [args.output_dir / "comparison.json", args.output_dir / "comparison.md"]
    if not args.overwrite and any(path.exists() for path in paths):
        print(
            f"Comparison output already exists: {next(path for path in paths if path.exists())}",
            file=sys.stderr,
        )
        return 2
    try:
        report, markdown = compare_monitoring_reports(
            args.baseline_report,
            args.candidate_report,
            args.policy,
            scenario=args.scenario,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths[0].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        paths[1].write_text(markdown)
    except (MonitoringError, OSError, TypeError, ValueError) as exc:
        print(f"Monitoring comparison failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote comparison report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
