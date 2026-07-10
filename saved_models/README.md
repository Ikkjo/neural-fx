# Saved Models Directory

This directory is for storing versioned, exported model files that are meant to be shared or deployed.

## Purpose

- Store exported models in formats like ONNX, TorchScript, or RTNeural
- Keep versioned checkpoints for production use
- Share trained models between team members

## Note

Training artifacts (checkpoints, logs) are stored in `lightning_logs/` and are git-ignored.
Only manually exported models should be placed here.

## File Types

- `.onnx` - ONNX format for cross-platform inference
- `.pt`, `.pth` - PyTorch TorchScript or state dict files
- `.json` - RTNeural format configuration and weights
