# Training

Training uses a YAML model config and a paired dry and processed recording.

## Prepare audio

The input and target must contain the same performance. Record the processed signal by sending the dry signal through the target device.

Training applies these steps:

1. Load each file.
2. Resample each file to `model.sample_rate`.
3. Mix multichannel audio to mono.
4. Normalize the pair by their shared peak when `data.normalize` is true.
5. Apply the measured or configured latency correction.
6. Truncate unequal files to the shorter aligned length.

Keep training and validation recordings separate. Point `data.train` and `data.val` to the correct pairs.

## Select a config

Shipped configs live under `configs/models/`:

```text
configs/models/lstm/
configs/models/gru/
configs/models/wavenet/
configs/models/s4/
```

Copy the closest config when you need different paths or hyperparameters. Do not edit several model variants to describe one experiment.

`model.sample_rate` controls model construction, audio loading, latency calibration, inference, and saved audio. The default is 48 kHz.

The training section controls the batch size, epochs, segment length, random segments, seed, workers, augmentation, and TBPTT settings. The loss section controls ESR, MSE, MR-STFT, pre-emphasis, and initial-sample masking.

## Start training

This command uses one CUDA device when PyTorch can access it:

```bash
python scripts/train.py --config configs/models/lstm/lstm_small.yaml
```

Force a CPU run when no suitable GPU is available:

```bash
python scripts/train.py \
  --config configs/models/lstm/lstm_small.yaml \
  --cpu
```

Override the epoch count without changing the config:

```bash
python scripts/train.py \
  --config configs/models/lstm/lstm_small.yaml \
  --max_epochs 10
```

## Latency and validation

Training validates the audio before it constructs the model. Validation checks file access, finite values, clipping, DC offset, sample rates, channels, and compatible lengths.

Latency calibration uses cross-correlation by default. Set a known delay with the manual mode:

```bash
python scripts/train.py \
  --config configs/models/lstm/lstm_small.yaml \
  --latency_method manual \
  --latency_manual 128
```

Set `latency.calibration_duration_seconds: 0` in the YAML file to disable calibration.

`--ignore_checks` continues after failed data checks. Use it only after you inspect the reported failures.

## Checkpoints and resume

The default checkpoint root is `lightning_logs/`. A run for `lstm_small` writes files under `lightning_logs/lstm_small/`.

- `best.ckpt` is a copy of the best monitored checkpoint when one is available.
- `last.ckpt` contains the terminal trainer state.
- Each checkpoint has a `.meta.json` sidecar with the config, data paths, hardware, Git commit, and training metrics.

Resume the terminal trainer state:

```bash
python scripts/train.py \
  --config configs/models/lstm/lstm_small.yaml \
  --resume lightning_logs/lstm_small/last.ckpt
```

Use `--checkpoint_dir PATH` to select another artifact root.

## Logs and reports

Training writes CSV and TensorBoard metrics to the same numbered run directory:

```bash
tensorboard --logdir lightning_logs
```

The default logging interval is 50 training steps. Use `--log_every_n_steps 1` for short smoke runs.

Add `--plot` to generate a post-training analysis report. A report failure does not invalidate a completed training run.

## Compiled WaveNet training

The shipped WaveNet configs set `training.compile: true`. The issue #40 RTX 3050 measurements showed about 20% faster warmed training for that workload.

The shipped LSTM, GRU, and S4D configs remain eager. Compilation results depend on the workload and hardware.

Use `--no-compile` to disable the WaveNet default:

```bash
python scripts/train.py \
  --config configs/models/wavenet/wavenet_small.yaml \
  --no-compile
```

Compiled training supports CPU or one CUDA device. It rejects enabled TBPTT and requests for more than one GPU. Compiler failures stop the run instead of retrying in eager mode.
