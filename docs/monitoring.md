# Offline monitoring

`scripts/monitor.py` evaluates one checkpoint or TorchScript artifact against a fixed audio suite. It is a development-time check, not live-service telemetry.

Copy `configs/monitoring/offline-suite.example.yaml` and set the input and target paths. Relative paths resolve from the manifest directory. Monitoring requires the native sample rate and channel count declared by the manifest. It does not resample, mix, normalize, or align the files.

## Checkpoint

```bash
python scripts/monitor.py \
  --manifest configs/monitoring/my-suite.yaml \
  --artifact checkpoints/model.ckpt \
  --output-dir monitoring/checkpoint
```

Pass `--config` for a legacy checkpoint without embedded or sidecar configuration.

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

TorchScript requires an explicit neural-fx configuration. Version 1 supports unconditioned LSTM, GRU, WaveNet, and S4 exports.

## Report

Every successful run writes `monitoring.json` and `monitoring.csv`. `--html` also writes `monitoring.html`. The version `1.0` JSON report contains:

- suite and audio fingerprints
- artifact and effective configuration hashes
- runtime and device identity
- preflight checks and warnings
- per-case ESR, MSE, MR-STFT, latency, and real-time factor
- aggregate quality, latency, artifact size, parameter count, and supported memory measurements

The suite fingerprint covers normalized workload settings, ordered case slices, and complete input and target file hashes. Moving unchanged files does not change it.

Exit code `0` means success. Expected manifest, validation, artifact, execution, or output failures return `2`. Unexpected failures return `1`.
