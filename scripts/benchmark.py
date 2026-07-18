"""Benchmark Neural-FX model inference latency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_fx.benchmarking import (  # noqa: E402
    BenchmarkModel,
    BenchmarkResult,
    benchmark_model,
    collect_environment,
    write_csv,
    write_json,
    write_markdown,
)
from neural_fx.config import load_config  # noqa: E402
from neural_fx.models import create_model_from_config  # noqa: E402


def _parse_block_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "block sizes must be comma-separated integers"
        ) from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("block sizes must be positive")
    return sizes


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _load_checkpoint(model, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    model.load_state_dict(state_dict)


def _parameter_bytes(model) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="Model YAML config paths")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        type=Path,
        default=None,
        help="Optional checkpoints in the same order as configs",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--block-sizes",
        type=_parse_block_sizes,
        default=[1, 64, 128, 256, 512, 1024],
        help="Comma-separated block sizes",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--name", default="benchmark", help="Artifact filename stem")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    if args.checkpoints is not None and len(args.checkpoints) != len(args.configs):
        raise SystemExit("--checkpoints must contain one path per config")

    device = _resolve_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    repo_root = Path(__file__).resolve().parent.parent
    results: list[BenchmarkResult] = []
    checkpoints = args.checkpoints or [None] * len(args.configs)

    for config_path, checkpoint_path in zip(args.configs, checkpoints, strict=True):
        config = load_config(config_path)
        model = create_model_from_config(config.model)
        if checkpoint_path is not None:
            _load_checkpoint(model, checkpoint_path)

        measurements = benchmark_model(
            model,
            block_sizes=args.block_sizes,
            warmup_iterations=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        environment = collect_environment(
            device=device, seed=args.seed, repo_root=repo_root
        )
        results.append(
            BenchmarkResult(
                environment=environment,
                model=BenchmarkModel(
                    name=config.name,
                    type=config.model.type,
                    sample_rate=config.sample_rate,
                    input_size=config.model.input_size,
                    output_size=config.model.output_size,
                    parameter_count=model.num_params,
                    parameter_bytes=_parameter_bytes(model),
                    config_path=str(config_path),
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                ),
                measurements=measurements,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(results, args.output_dir / f"{args.name}.json")
    write_csv(results, args.output_dir / f"{args.name}.csv")
    write_markdown(results, args.output_dir / f"{args.name}.md")


if __name__ == "__main__":
    main()
