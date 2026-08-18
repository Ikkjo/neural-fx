#!/usr/bin/env python3
"""Combine checkpoint quality and benchmark performance results."""

import argparse
import json
from pathlib import Path

from neural_fx.analysis.benchmarking import load_benchmark_result
from neural_fx.analysis.evaluation import (
    build_architecture_report,
    build_comparison_report,
    load_evaluation_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        type=Path,
        default=[],
        help="Fresh benchmark results for representative checkpoints",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size-tolerance", type=float, default=1.01)
    args = parser.parse_args()

    evaluation_results = [load_evaluation_result(path) for path in args.results]
    benchmark_results = []
    for path in args.benchmarks:
        benchmark = load_benchmark_result(path)
        benchmark["_result_path"] = str(path.resolve())
        benchmark_results.append(benchmark)

    report, markdown = build_architecture_report(
        evaluation_results,
        benchmark_results,
        size_tolerance=args.size_tolerance,
    )
    seed_report, seed_markdown = build_comparison_report(
        evaluation_results,
        size_tolerance=args.size_tolerance,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "comparison.md").write_text(markdown)
    (args.output_dir / "seed-comparison.json").write_text(
        json.dumps(seed_report, indent=2) + "\n"
    )
    (args.output_dir / "seed-comparison.md").write_text(seed_markdown)
    print(f"Wrote comparison report to {args.output_dir}")


if __name__ == "__main__":
    main()
