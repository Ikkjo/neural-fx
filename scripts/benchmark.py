#!/usr/bin/env python3
"""Benchmark one neural-fx model in a fresh process."""

import argparse
from pathlib import Path

import torch

from neural_fx.analysis.benchmarking import (
    benchmark_model,
    load_model_for_evaluation,
    write_benchmark_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Config YAML; optional when the checkpoint embeds neural_fx_config",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--num-samples", type=int, default=48_000)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measurement-runs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()

    if args.config is None and args.checkpoint is None:
        parser.error("one of --config or --checkpoint is required")

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    model, config = load_model_for_evaluation(
        args.config, args.checkpoint, device=device
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = benchmark_model(
        model,
        model_name=config.name,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        block_sizes=args.block_sizes,
        num_samples=args.num_samples,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    write_benchmark_result(result, args.output)
    print(f"Wrote benchmark result to {args.output}")


if __name__ == "__main__":
    main()
