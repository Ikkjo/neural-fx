#!/usr/bin/env python3
"""Combine checkpoint quality and benchmark performance results."""

import argparse
import json
from pathlib import Path

from neural_fx.analysis.evaluation import (
    build_comparison_report,
    load_evaluation_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size-tolerance", type=float, default=1.35)
    args = parser.parse_args()

    report, markdown = build_comparison_report(
        [load_evaluation_result(path) for path in args.results],
        size_tolerance=args.size_tolerance,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "comparison.md").write_text(markdown)
    print(f"Wrote comparison report to {args.output_dir}")


if __name__ == "__main__":
    main()
