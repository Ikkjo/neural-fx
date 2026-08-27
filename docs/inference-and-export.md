# Inference and export

Use a checkpoint for Python inference. Export a model when another runtime needs the graph and weights.

## Load a checkpoint

New Lightning checkpoints contain the complete neural-fx config:

```python
from neural_fx.artifacts import load_model

loaded = load_model(checkpoint_path="lightning_logs/lstm_small/last.ckpt")
model = loaded.model
```

Pass `config_path` for a legacy checkpoint without an embedded config or `.meta.json` sidecar:

```python
loaded = load_model(
    checkpoint_path="checkpoints/model.ckpt",
    config_path="configs/models/lstm/lstm_small.yaml",
)
```

The loader accepts Lightning checkpoints and raw model state dictionaries. It loads weights strictly.

## Process an audio file

`process_audio` loads a file, resamples it to the model rate, mixes it to mono, carries model state across chunks, and writes the result:

```python
from neural_fx.artifacts import load_model
from neural_fx.inference import process_audio

loaded = load_model(checkpoint_path="lightning_logs/lstm_small/last.ckpt")
process_audio(
    loaded.model,
    "audio/dry.wav",
    "outputs/processed.wav",
    chunk_size=8192,
)
```

Each call resets state once before the first chunk.

## Process a stream

Use `InferenceSession` for a persistent stream:

```python
from neural_fx.inference import InferenceSession

session = InferenceSession(model)
output_block = session.process_block(input_block)
output_sample = session.process_sample(input_sample)
session.reset()
```

`process_block` expects a tensor with batch, channel, and sample dimensions. `process_sample` accepts one floating-point sample and returns one floating-point sample.

Keep one session per independent audio stream. Call `reset()` between unrelated streams.

## Export a checkpoint

The export script requires a config and checkpoint:

```bash
python scripts/export.py \
  --config configs/models/lstm/lstm_small.yaml \
  --checkpoint lightning_logs/lstm_small/last.ckpt \
  --output_dir exports
```

Select formats with a comma-separated list:

```bash
python scripts/export.py \
  --config configs/models/lstm/lstm_small.yaml \
  --checkpoint lightning_logs/lstm_small/last.ckpt \
  --output_dir exports \
  --formats onnx,torchscript
```

The script writes artifacts under `OUTPUT_DIR/CONFIG_NAME/`. It skips unsupported requested formats and prints the supported formats.

## Compatibility matrix

| Model graph | Streaming | TorchScript | ONNX | RTNeural | Conditioning |
| --- | --- | --- | --- | --- | --- |
| LSTM | Yes | Yes | Yes | Only plain recurrent and dense graphs | Python, TorchScript, and ONNX |
| GRU | Yes | Yes | Yes | Only plain recurrent and dense graphs | Python, TorchScript, and ONNX |
| WaveNet | Yes | Yes | Yes | No | No |
| S4D | Yes | Yes, with explicit state | No | No | No |

The shipped LSTM and GRU configs use strided convolution. Their graphs need transposed-convolution upsampling, so RTNeural export is not supported for those configs.

RTNeural also rejects conditioned recurrent models and recurrent skip connections. The exporter raises an actionable error instead of dropping those graph features.

ONNX exports have dynamic batch and time axes for LSTM, GRU, and WaveNet. S4D uses FFT operations during full-sequence execution and exports only its real-valued recurrent TorchScript cell.
