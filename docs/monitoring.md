# Offline monitoring

Offline monitoring checks one checkpoint or TorchScript artifact against a fixed audio suite. It does not monitor a live service.

Use the same suite for each new artifact. The suite fingerprint identifies the workload and complete audio contents.

## Create a suite

Copy the tracked example:

```bash
cp configs/monitoring/offline-suite.example.yaml \
  configs/monitoring/my-suite.yaml
```

Edit the case paths and workload settings. Relative paths start from the manifest directory.

Each file must match the declared sample rate and channel count. Monitoring does not resample, mix, normalize, or align audio.

The manifest controls the segment length, burn-in, inference chunks, latency blocks, warm-up runs, measured runs, quality metrics, amplitude limits, and ordered cases.

## Monitor a checkpoint

```bash
python scripts/monitor.py \
  --manifest configs/monitoring/my-suite.yaml \
  --artifact lightning_logs/lstm_small/last.ckpt \
  --output-dir monitoring/lstm-small
```

Pass `--config` when the checkpoint has no embedded config or valid sidecar.

## Monitor a TorchScript artifact

```bash
python scripts/monitor.py \
  --manifest configs/monitoring/my-suite.yaml \
  --artifact exports/lstm_small/lstm_small.pt \
  --artifact-type torchscript \
  --config configs/models/lstm/lstm_small.yaml \
  --output-dir monitoring/lstm-small-torchscript \
  --html
```

TorchScript monitoring needs the neural-fx model config. Version 1 supports unconditioned LSTM, GRU, WaveNet, and S4D artifacts.

## Outputs

A successful run writes:

- `monitoring.json`
- `monitoring.csv`
- `monitoring.html` when `--html` is present

The version 1.0 report records:

- Suite and audio fingerprints
- Artifact and config hashes
- Runtime and device identity
- Preflight results and warnings
- Per-case ESR, MSE, MR-STFT, latency, and real-time factor
- Aggregate quality, latency, artifact size, parameter count, and supported memory measurements

The suite fingerprint covers workload settings, ordered case slices, and complete audio hashes. Moving the same files does not change it.

The command returns 0 after success. It returns 2 for expected monitoring errors and 1 for unexpected failures.

Monitoring reports describe each artifact. They do not apply a baseline regression policy or select a preferred model.
