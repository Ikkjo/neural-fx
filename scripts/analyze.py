#!/usr/bin/env python3
"""Generate the legacy analysis artifacts for one trained checkpoint."""

import argparse
import json
from pathlib import Path

from neural_fx.analysis.plotting import generate_analysis_report, write_html_report
from neural_fx.artifacts import load_model
from neural_fx.data.dataset import AudioDataset


def load_checkpoint(checkpoint_path: str, config_path: str | None = None):
    loaded = load_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    return loaded.model, loaded.config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Config YAML for legacy checkpoints without embedded configuration",
    )
    parser.add_argument("--input", type=Path, help="Override input audio")
    parser.add_argument("--target", type=Path, help="Override target audio")
    parser.add_argument("--output_dir", type=Path, default=Path("analysis"))
    parser.add_argument("--num_samples", type=int, default=48_000)
    parser.add_argument("--generate_html", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config = load_checkpoint(str(args.checkpoint), str(args.config) if args.config else None)
    input_path = args.input or config.data.train.input
    target_path = args.target or config.data.train.target
    dataset = AudioDataset(
        input_path=input_path,
        target_path=target_path,
        segment_length=config.training.segment_length,
        sample_rate=config.sample_rate,
        normalize=config.data.normalize,
        random_segments=False,
    )
    report = generate_analysis_report(
        model,
        dataset,
        args.output_dir,
        args.num_samples,
        config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    if args.generate_html:
        write_html_report(report, args.output_dir / "report.html")

    print(f"ESR: {report['esr']:.6f} - {report['esr_comment']}")
    print(f"MSE: {report['mse']:.6f}")
    print(f"Correlation: {report['correlation']:.4f}")
    print(f"Model Parameters: {report['num_params']:,}")
    print(f"JSON report saved to: {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
