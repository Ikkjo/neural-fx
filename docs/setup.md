# Setup

Run commands from the repository root. Use Python 3.10 through 3.13.

## System audio dependency

neural-fx uses TorchCodec through torchaudio. TorchCodec needs FFmpeg shared libraries.

On Ubuntu, install FFmpeg before the Python package:

```bash
sudo apt-get update
sudo apt-get install --yes ffmpeg
```

Use the FFmpeg package from your operating system on other platforms. Confirm that the `ffmpeg` command is available:

```bash
ffmpeg -version
```

## Core installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

The core installation includes PyTorch, torchaudio, TorchCodec, Lightning, TensorBoard, NumPy, SciPy, Matplotlib, tqdm, and PyYAML.

## CUDA installation

The core command installs the PyTorch build selected by pip for your platform. For a specific CUDA build, install PyTorch first with the command from the [PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install neural-fx:

```bash
python -m pip install -e .
```

Check device access:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Training uses one CUDA device when `torch.cuda.is_available()` is true. Use `--cpu` when you need an explicit CPU run.

## Development installation

Install pytest, coverage support, and Ruff:

```bash
python -m pip install -e ".[dev]"
```

Run the same core checks as CI:

```bash
python -m pytest tests
ruff check neural_fx scripts tests setup.py
```

CI tests Python 3.10 and 3.12 with PyTorch 2.11 on CPU.

## ONNX installation

Install the optional ONNX exporter and runtime dependencies:

```bash
python -m pip install -e ".[onnx]"
```

Install both optional groups for development:

```bash
python -m pip install -e ".[dev,onnx]"
```

Run the optional ONNX Runtime tests:

```bash
python -m pytest --run-onnx tests/test_export.py tests/test_wavenet.py
```

## Installation check

Check the installed dependency set and script imports:

```bash
python -m pip check
python scripts/train.py --help
python scripts/monitor.py --help
```
