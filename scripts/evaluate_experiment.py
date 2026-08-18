#!/usr/bin/env python3
"""Evaluate one trained checkpoint from a controlled experiment manifest."""

import argparse
from pathlib import Path

from neural_fx.analysis.evaluation import (
    evaluate_experiment,
    load_experiment_manifest,
    write_evaluation_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = load_experiment_manifest(args.manifest)
    result = evaluate_experiment(manifest, args.output_dir, args.device)
    result_path = args.output_dir / "evaluation.json"
    write_evaluation_result(result, result_path)
    print(f"Wrote evaluation result to {result_path}")


if __name__ == "__main__":
    main()
