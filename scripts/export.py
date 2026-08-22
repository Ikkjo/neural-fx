#!/usr/bin/env python3
"""Export script for trained neural audio effects models."""

import argparse
from pathlib import Path

from neural_fx.artifacts import load_model


def main():
    parser = argparse.ArgumentParser(
        description="Export trained neural audio effects model"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./exports", help="Output directory"
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="onnx,torchscript,rtneural",
        help="Comma-separated export formats",
    )
    args = parser.parse_args()

    loaded = load_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )
    model = loaded.model
    config = loaded.config

    output_dir = Path(args.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = [f.strip().lower() for f in args.formats.split(",")]

    for fmt in formats:
        if fmt not in model.supported_export_formats:
            supported = ", ".join(model.supported_export_formats)
            print(
                f"Skipping unsupported {fmt} export for {config.model.type}; "
                f"supported formats: {supported or 'none'}"
            )
            continue

        if fmt == "onnx":
            output_path = output_dir / f"{config.name}.onnx"
            model.export_onnx(output_path)
            print(f"Exported ONNX model to {output_path}")

        elif fmt == "torchscript":
            output_path = output_dir / f"{config.name}.pt"
            model.export_torchscript(output_path)
            print(f"Exported TorchScript model to {output_path}")

        elif fmt == "rtneural":
            output_path = output_dir / f"{config.name}.json"
            model.export_rtneural(output_path)
            print(f"Exported RTNeural model to {output_path}")

        else:
            print(f"Unknown format: {fmt}")

    print("Export complete.")


if __name__ == "__main__":
    main()
