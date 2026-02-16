#!/usr/bin/env python3
"""Training script for neural audio effects models."""

import argparse
from pathlib import Path

import lightning as L
import torch

from neural_fx.config import load_config
from neural_fx.models.recurrent import RecurrentNeuralFXModel
from neural_fx.training.lightning_module import NeuralFXModule


def main():
    parser = argparse.ArgumentParser(description="Train neural audio effects model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="./lightning_logs", help="Checkpoint directory")
    args = parser.parse_args()

    config = load_config(args.config)

    L.seed_everything(config.training.seed, workers=True)

    model = RecurrentNeuralFXModel.from_config(config.model)

    module = NeuralFXModule(model, config)

    epochs = args.max_epochs if args.max_epochs else config.training.epochs

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir) / config.name,
        filename="{epoch:02d}-{train_loss:.4f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if args.gpus > 0 and torch.cuda.is_available() else "cpu",
        devices=args.gpus if args.gpus > 0 and torch.cuda.is_available() else 1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    trainer.fit(module)

    print(f"Training complete. Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
