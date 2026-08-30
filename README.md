# neural-fx

neural-fx trains neural networks to model guitar amplifiers and effects from paired audio. The package supports LSTM, GRU, WaveNet, and S4D models.

The main workflows run through scripts. The `neural_fx` package provides shared model, training, inference, evaluation, and monitoring code.

## Requirements

- Python 3.10 through 3.13
- FFmpeg shared libraries for audio loading
- A dry input WAV and its time-aligned processed target WAV
- A CUDA GPU for practical training times, or a CPU for small runs

## Quick start

Clone the repository and install the core package:

```bash
git clone https://github.com/Ikkjo/neural-fx.git
cd neural-fx
python -m pip install -e .
```

Put a paired recording at these paths:

```text
data/DI.wav
data/effect.wav
```

The files must contain the same performance. `DI.wav` is the dry signal. `effect.wav` is the recorded amplifier or effect output.

Train the small LSTM for one epoch:

```bash
python scripts/train.py \
  --config configs/models/lstm/lstm_small.yaml \
  --max_epochs 1
```

The command uses one CUDA device when PyTorch can access it. Add `--cpu` to force CPU training:

```bash
python scripts/train.py \
  --config configs/models/lstm/lstm_small.yaml \
  --max_epochs 1 \
  --cpu
```

Training writes the terminal checkpoint to `lightning_logs/lstm_small/last.ckpt`. Process a WAV file with that checkpoint:

```python
from neural_fx.artifacts import load_model
from neural_fx.inference import process_audio

loaded = load_model(checkpoint_path="lightning_logs/lstm_small/last.ckpt")
process_audio(loaded.model, "data/DI.wav", "outputs/lstm_small.wav")
```

One epoch checks the workflow. It does not produce a useful amplifier or effect model. Train on separate training and validation recordings for real experiments.

## Models

| Model | Config name | Training | Streaming inference | Export summary |
| --- | --- | --- | --- | --- |
| LSTM | `lstm` | Eager by default | Sample and block | ONNX, TorchScript, and restricted RTNeural |
| GRU | `gru` | Eager by default | Sample and block | ONNX, TorchScript, and restricted RTNeural |
| WaveNet | `wavenet` | Compiled in shipped configs | Sample and block | ONNX and TorchScript |
| S4D | `s4` | Eager by default | Sample and block | TorchScript |

The shipped WaveNet configs enable `torch.compile`. This default improved warmed training time by about 20% on the measured RTX 3050 workload. Results can differ on other hardware.

## Documentation

- [Setup](docs/setup.md): development tools, ONNX dependencies, CUDA, and environment checks
- [Training](docs/training.md): audio requirements, configs, checkpoints, resume, logs, and training options
- [Inference and export](docs/inference-and-export.md): file and streaming inference, artifact loading, and export support
- [Evaluation](docs/evaluation.md): controlled quality evaluation and inference benchmarks
- [Offline monitoring](docs/monitoring.md): repeatable checks for checkpoints and TorchScript artifacts
- [S4D architecture decision](docs/decisions/ssm-architecture.md): the accepted state-space model design

## Development checks

```bash
python -m pytest tests
ruff check neural_fx scripts tests setup.py
```
