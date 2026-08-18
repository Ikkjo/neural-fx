#!/usr/bin/env python3
"""Generate issue-15 pilot or final training and evaluation files."""

import argparse
from pathlib import Path

from neural_fx.experiments.issue15 import generate_issue15_run_files


def main() -> None:
    """Parse the run phase and write its immutable files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "final"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    for model_type in ("lstm", "gru", "wavenet", "s4"):
        parser.add_argument(f"--{model_type}-lr", type=float)
    args = parser.parse_args()

    selected_rates = None
    if args.phase == "final":
        selected_rates = {
            "lstm": args.lstm_lr,
            "gru": args.gru_lr,
            "wavenet": args.wavenet_lr,
            "s4": args.s4_lr,
        }
        if any(rate is None for rate in selected_rates.values()):
            parser.error(
                "final phase requires --lstm-lr, --gru-lr, --wavenet-lr, and --s4-lr"
            )

    written = generate_issue15_run_files(
        args.dataset_root,
        args.output,
        args.checkpoint_root,
        selected_learning_rates=selected_rates,
    )
    print(f"Wrote {len(written)} run files under {args.output.resolve()}")


if __name__ == "__main__":
    main()
