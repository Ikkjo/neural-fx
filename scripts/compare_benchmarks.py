#!/usr/bin/env python3
"""Build a Markdown comparison table from benchmark JSON files."""

import argparse
from pathlib import Path

from neural_fx.analysis.benchmarking import (
    format_benchmark_table,
    load_benchmark_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    table = format_benchmark_table(
        [load_benchmark_result(path) for path in args.results]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"Wrote benchmark comparison to {args.output}")


if __name__ == "__main__":
    main()
