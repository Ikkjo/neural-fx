#!/usr/bin/env python3
"""Prepare the preregistered final dataset for GitHub issue 15."""

import argparse
import json
from pathlib import Path

from neural_fx.preprocessing.experiment_data import prepare_issue15_dataset


def main() -> None:
    """Run the fixed issue-15 data preparation contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source DI WAV")
    parser.add_argument(
        "--target", type=Path, required=True, help="Source amplifier WAV"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New immutable output directory"
    )
    args = parser.parse_args()

    manifest = prepare_issue15_dataset(args.input, args.target, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifacts": len(manifest["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
