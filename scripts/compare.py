"""Validate and aggregate a Neural-FX comparison experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permit direct invocation from a source checkout.
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_fx.evaluation import (
    ManifestValidationError,
    aggregate_results,
    load_manifest,
    load_run_results,
    write_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, aggregate, and report a comparison experiment"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Versioned JSON containing one metrics record per manifest run",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("comparison"))
    parser.add_argument(
        "--validate-files",
        action="store_true",
        help="Require every audio, config, and checkpoint path to exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest, require_files=args.validate_files)
        records = load_run_results(args.results)
        aggregates = aggregate_results(manifest, records)
        reports = write_reports(manifest, aggregates, args.output_dir)
    except (ManifestValidationError, OSError, ValueError) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 2

    print(f"Validated {len(records)} runs for {manifest.name!r}.")
    for format_name, path in reports.items():
        print(f"{format_name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
