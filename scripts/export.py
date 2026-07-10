#!/usr/bin/env python3
"""Export script for trained neural audio effects models."""

import argparse
from pathlib import Path

import torch

from neural_fx.config import load_config
from neural_fx.models.recurrent import RecurrentNeuralFXModel


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

    config = load_config(args.config)
    model = RecurrentNeuralFXModel.from_config(config.model)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Remove "model." prefix if present (from Lightning checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    output_dir = Path(args.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = [f.strip().lower() for f in args.formats.split(",")]

    for fmt in formats:
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
