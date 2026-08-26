# Offline monitoring

Use `scripts/monitor.py` to evaluate one checkpoint or TorchScript artifact against a fixed audio suite. This command does not monitor a live service.

Copy `configs/monitoring/offline-suite.example.yaml`, then edit the input and target paths. The command resolves relative paths from the manifest directory.

Each audio file must use the sample rate and channel count declared in the manifest. The command does not resample, mix, normalize, or align audio.

## Checkpoint

```bash
python scripts/monitor.py \
  --manifest configs/monitoring/my-suite.yaml \
  --artifact checkpoints/model.ckpt \
  --output-dir monitoring/checkpoint
```

Use `--config` if the checkpoint has no embedded configuration or sidecar configuration file.

## TorchScript

```bash
python scripts/monitor.py \
  --manifest configs/monitoring/my-suite.yaml \
  --artifact exports/model.pt \
  --artifact-type torchscript \
  --config configs/models/lstm/lstm_nano.yaml \
  --output-dir monitoring/torchscript \
  --html
```

A TorchScript artifact requires a neural-fx configuration file. Version 1 supports unconditioned LSTM, GRU, WaveNet, and S4 exports.

## Report

A successful run writes `monitoring.json` and `monitoring.csv`. Add `--html` to write `monitoring.html`.

The version `1.0` JSON report contains:

- suite and audio fingerprints
- artifact and effective configuration hashes
- runtime and device identity
- preflight checks and warnings
- per-case ESR, MSE, MR-STFT, latency, and real-time factor
- aggregate quality, latency, artifact size, parameter count, and supported memory measurements

The suite fingerprint covers the workload settings, ordered case slices, and complete audio file hashes. File locations do not affect the fingerprint.

The command returns exit code `0` after a successful run. It returns `2` for expected monitoring errors and `1` for unexpected errors.
