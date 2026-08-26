#!/usr/bin/env python3
"""Monitor a checkpoint or TorchScript artifact against a fixed audio suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neural_fx.monitoring import (
    MonitoringError,
    monitor_artifact,
    write_monitoring_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--artifact-type", choices=("checkpoint", "torchscript"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--html", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = monitor_artifact(
            args.manifest,
            args.artifact,
            artifact_type=args.artifact_type,
            config_path=args.config,
            device=args.device,
        )
        paths = write_monitoring_outputs(
            report,
            args.output_dir,
            include_html=args.html,
        )
    except (MonitoringError, OSError, TypeError, ValueError) as exc:
        print(f"Monitoring failed: {exc}", file=sys.stderr)
        return 2
    for output_type, path in paths.items():
        print(f"Wrote {output_type} report to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
