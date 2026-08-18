# neural-fx

Real-time guitar effect and amp modelling using neural networks (LSTM, GRU, WaveNet, SSM/Mamba/S4).

## Features

- **Model Architectures**: LSTM, GRU, and causal WaveNet with configurable sizes
- **Audio Processing**: 48kHz sample rate, chunked processing for memory efficiency
- **Training**: PyTorch Lightning with TBPTT (Truncated Backpropagation Through Time) and burn-in support
- **Live metrics**: CSV and TensorBoard logs written to the same training run directory
- **Conditioning**: Support for gain knob and other control parameters
- **Export**: ONNX, TorchScript, and RTNeural JSON formats for deployment
- **Inference**: Real-time streaming processor for single-sample and block processing

## Quick Start

### Installation

```bash
git clone <repository-url>
cd neural-fx
pip install -e .
```

### Training a Model

```bash
# Train an LSTM model
python scripts/train.py --config configs/models/lstm/lstm_medium.yaml

# Train a GRU model
python scripts/train.py --config configs/models/gru/gru_medium.yaml

# Train a causal WaveNet
python scripts/train.py --config configs/models/wavenet/wavenet_small.yaml

# Train with custom epochs
python scripts/train.py --config configs/models/lstm/lstm_small.yaml --max_epochs 50

# Follow a run while it trains
tensorboard --logdir lightning_logs
```

For short experiments, pass `--log_every_n_steps 1` to update both loggers on
every training batch. The default interval is 50 steps.

### Exporting a Trained Model

```bash
# Export to all formats (ONNX, TorchScript, RTNeural)
python scripts/export.py \
    --config configs/models/lstm/lstm_medium.yaml \
    --checkpoint lightning_logs/lstm_medium/epoch=10.ckpt \
    --output_dir ./exports

# Export to specific formats only
python scripts/export.py \
    --config configs/models/lstm/lstm_medium.yaml \
    --checkpoint lightning_logs/lstm_medium/epoch=10.ckpt \
    --formats onnx,torchscript
```

## Model Configurations

### LSTM Models

| Model | Hidden Size | Layers | Conv Filters | Stride | Params |
|-------|-------------|--------|--------------|--------|--------|
| nano | 20 | 1 | 16 | 4 | ~3K |
| small | 36 | 2 | 16 | 12 | ~11K |
| medium | 64 | 2 | 36 | 4 | ~41K |
| large | 96 | 2 | 36 | 3 | ~81K |
| xl | 128 | 2 | 36 | 3 | ~137K |

### GRU Models

| Model | Hidden Size | Layers | Conv Filters | Stride | Params |
|-------|-------------|--------|--------------|--------|--------|
| nano | 20 | 1 | 16 | 4 | ~2K |
| small | 36 | 2 | 16 | 12 | ~8K |
| medium | 64 | 2 | 36 | 4 | ~31K |
| large | 96 | 2 | 36 | 3 | ~61K |
| xl | 128 | 2 | 36 | 3 | ~103K |

### WaveNet Models

WaveNet configurations use repeated dilation cycles. Its exact receptive field is
`1 + (kernel_size - 1) * stacks * (2**layers - 1)` samples. Full-sequence,
chunked, and cached sample inference are causal. TorchScript and ONNX export are
supported; the current RTNeural JSON format cannot represent the WaveNet graph.

## Using NAM-Style Test Signals

The system supports both real audio recordings and NAM-style test tone signals for training:

### NAM Test Tones

[NAM (Neural Amp Modeler)](https://github.com/sdatkinson/neural-amp-modeler) uses specialized test signals with **blips** (impulse spikes) for precise latency calibration. These signals provide:
- Clean impulse responses for accurate delay measurement
- Replicability checks that pass cleanly (low ESR)
- Standardized input files for consistent results

### Training with NAM Inputs

To use NAM-style inputs, set the latency method to `blip` for automatic detection:

```bash
python scripts/train.py \
    --config configs/models/lstm/lstm_nano_nam.yaml \
    --latency_method blip
```

### NAM Configuration Example

```yaml
# NAM-specific configuration
latency:
  method: "blip"  # Use blip detection for NAM inputs
  manual_delay: null
  max_delay: 10000
  calibration_duration_seconds: 5.0

data:
  train:
    input: "data/nam_input_v3.wav"  # NAM v3 standard input
    target: "data/nam_output.wav"
```

The replicability check will typically show low ESR values (< 0.01) with NAM inputs, indicating the model can accurately replicate the target signal.

## Configuration

Models are configured via YAML files. Example:

```yaml
version: "1.0"
name: "lstm_medium"

model:
  type: "lstm"  # or "gru"
  input_size: 1
  output_size: 1
  sample_rate: 48000
  params:
    hidden_size: 64
    num_layers: 2
    conv1d:
      filters: 36
      kernel_size: 3
      stride: 4
    skip_connection: false
    dropout: 0.0
    conditioning_size: 0  # Set >0 for gain knob support

training:
  batch_size: 32
  epochs: 100
  segment_length: 8192
  tbptt:
    enabled: true
    burn_in: 4096  # Samples to warm up hidden state
  seed: 42

optimizer:
  type: "adam"
  lr: 0.01

lr_scheduler:
  type: "exponential"
  gamma: 0.995

loss:
  type: "mse"
  weights:
    esr: 0.0  # Error-to-signal ratio weight
    mse: 1.0  # MSE weight
  pre_emphasis:
    enabled: true
    coef: 0.85
  mask_first: 4096  # Exclude from loss calculation

data:
  train:
    input: "data/DI.wav"    # Dry guitar input
    target: "data/effect.wav"  # Processed target
```

`model.sample_rate` is the single source of truth for model construction, data
resampling, latency calibration, inference, analysis, and saved audio.
Set `latency.calibration_duration_seconds` to `0` to disable latency
calibration; positive values select how much audio is used.

## Project Structure

```
neural-fx/
├── neural_fx/              # Main package
│   ├── config.py           # Configuration dataclasses
│   ├── data/               # Data loading & transforms
│   │   ├── dataset.py      # AudioDataset for training
│   │   └── transforms.py   # Audio preprocessing
│   ├── models/             # Model implementations
│   │   ├── base.py         # Base model interface
│   │   ├── recurrent.py    # LSTM/GRU implementations
│   │   ├── wavenet.py      # WaveNet (future)
│   │   └── ssm.py          # SSM/Mamba/S4 (future)
│   ├── training/           # Training infrastructure
│   │   └── lightning_module.py  # LightningModule with TBPTT
│   ├── losses/             # Loss functions
│   │   └── audio_losses.py # ESR, MSE with pre-emphasis
│   └── inference/          # Inference utilities
│       └── streaming.py    # Real-time processing
├── configs/models/         # Model configurations
│   ├── lstm/               # LSTM configs (nano/small/medium/large/xl)
│   └── gru/                # GRU configs (nano/small/medium/large/xl)
├── scripts/                # Entry-point scripts
│   ├── train.py            # Training script
│   └── export.py           # Model export script
├── tests/                  # Test suite
│   └── test_recurrent.py   # LSTM/GRU tests
└── notebooks/              # Jupyter notebooks (training, analysis)
```

## Testing

This project uses `pytest` for testing.

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_recurrent.py -v

# Run with coverage
pytest --cov=neural_fx
```

See [TESTING.md](TESTING.md) for detailed testing guidelines.

## Linting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and code style enforcement.

```bash
# Run ruff linter
ruff check .

# Run ruff formatter
ruff format .

# Fix auto-fixable issues
ruff check --fix .
```

## Usage Examples

### Loading and Using a Model

```python
from neural_fx.config import load_config
from neural_fx.models.recurrent import RecurrentNeuralFXModel

# Load config and create model
config = load_config('configs/models/lstm/lstm_medium.yaml')
model = RecurrentNeuralFXModel.from_config(config.model)

# Process audio
import torch
audio = torch.randn(1, 1, 48000)  # [batch, channels, samples]
output = model(audio)
```

### Real-time Inference

```python
from neural_fx.inference import StreamingProcessor

# Create streaming processor
processor = StreamingProcessor(model)  # Uses model.sample_rate

# Process single sample (for real-time audio callbacks)
output_sample = processor.process_sample(input_sample)

# Or process a block of samples
output_block = processor.process_block(input_block)
```

### Evaluation

```python
from neural_fx.inference import evaluate_model

metrics = evaluate_model(
    model,
    input_path='data/test_DI.wav',
    target_path='data/test_effect.wav',
    burn_in=4096  # Exclude burn-in from metrics
)
print(f"MSE: {metrics['mse']:.6f}, ESR: {metrics['esr']:.6f}")
```

## Training Details

### TBPTT (Truncated Backpropagation Through Time)

The training uses a sliding window approach for TBPTT:
- Each training segment is processed in chunks
- Hidden state is detached between chunks to truncate gradient flow
- Burn-in samples warm up the hidden state but are excluded from loss

### Burn-in

The first `burn_in` samples of each training segment:
- Are processed through the model to warm up hidden state
- Are excluded from loss calculation entirely
- Help the model reach a stable state before learning

## Model Export Formats

### ONNX
- Compatible with ONNX Runtime
- Dynamic batch and time dimensions
- Suitable for cross-platform deployment

### TorchScript
- Stateless wrapper for traceability
- Can be loaded in C++ with libtorch
- Optimized for inference

### RTNeural JSON
- Compatible with [RTNeural](https://github.com/jatinchowdhury18/RTNeural) C++ library
- Lightweight format for real-time audio plugins
- Supports LSTM, GRU, Conv1D, and Dense layers

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines and code style.

## License

[Add your license here]
