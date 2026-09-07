# Offline monitoring

Offline monitoring checks one checkpoint or TorchScript artifact against a fixed audio suite.

Use the same suite for each new artifact. The suite fingerprint identifies the workload and complete audio contents.

## Create a suite

Copy the tracked example:

```bash
cp configs/monitoring/offline-suite.example.yaml \
  configs/monitoring/my-suite.yaml
```

Edit the case paths and workload settings. Relative paths start from the manifest directory.

Each file must match the declared sample rate and channel count. Monitoring does not resample, mix, normalize, or align audio.

The manifest controls the segment length, burn-in, inference chunks, latency blocks, warm-up runs, measured runs, quality metrics, amplitude limits, and ordered cases. Set `esr_mode` and `esr_pre_emphasis` explicitly because both settings change the meaning of the ESR value.

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

Use a different output directory for each artifact version. The command refuses to replace an existing report unless `--overwrite` is passed.

## Outputs

A successful run writes:

- `monitoring.json`
- `monitoring.csv`
- `monitoring.html` when `--html` is present

The version 1.1 report records:

- Suite and audio fingerprints
- Artifact and config hashes
- Runtime, device identity, and PyTorch thread count
- Preflight results and warnings
- Per-case ESR, MSE, MR-STFT, latency, and real-time factor
- Digital-silence status, absolute MSE, prediction RMS/peak, and eligible/excluded relative-score counts
- Prediction peak-amplitude and clipping warnings
- Aggregate quality, latency, artifact size, parameter count, and supported memory measurements

The suite fingerprint covers workload settings, ordered case slices, and complete audio hashes. Moving the same files does not change it.

### Digital silence policy

The fixed `digital_silence_v1` policy classifies a post-burn-in case as silent only when every target sample is exactly zero. ESR and MR-STFT are stored as JSON `null` for silent cases and excluded from their aggregate means; if no case is eligible, the aggregate is also `null`. MSE, prediction RMS, and prediction absolute peak remain finite diagnostics. CSV leaves unavailable relative values blank and includes status/count fields; HTML renders them as `N/A`.

Monitoring and controlled evaluation use the same target-only policy but different workloads: monitoring averages complete post-burn-in cases, while evaluation averages its existing fixed three-second MR-STFT windows and uses one complete segment for ESR/MSE. Training losses and checkpoints are unchanged.

The command returns 0 after success. It returns 2 for expected monitoring errors and 1 for unexpected failures.

## Compare reports

Run monitoring for each new checkpoint or export, then compare its JSON or CSV report with the chosen baseline. A comparison is valid only when:

- `suite.fingerprint` matches;
- `workload.esr_mode`, `workload.esr_pre_emphasis`, burn-in, segment length, and selected quality metrics match;
- latency results use the same device, dtype, inference category, and effective chunk size.

Checkpoint inference uses stateful chunks and records the configured `inference_chunk_size` as its effective chunk size. Current TorchScript recurrent and WaveNet exports process the complete sequence in one call because their exported interfaces do not expose streaming state. Their effective chunk size is therefore `null`. Do not compare latency between these two execution methods as if they represented the same deployment workload.

Lower ESR, MSE, MR-STFT distance, latency, real-time factor, memory use, and artifact size are better when every other comparison condition is fixed. Treat a candidate as a regression only when it crosses a threshold chosen before inspecting that candidate. Record both the absolute change and percentage change. Use validation cases while choosing models and reserve fixed test cases for final checks to avoid adapting repeatedly to the test set.

Monitoring reports describe each artifact. They do not apply a baseline regression policy or select a preferred model.

## Compare artifact versions

TAMU's offline version-monitoring example compares two reports from the same fixed validation suite against a declared policy:

```bash
python scripts/monitor.py \
  --manifest configs/monitoring/tamu-ds1-gain-75-validation.yaml \
  --artifact local/gear_comparison_44100/checkpoints/gear_comparison_44100_ds1_gain_75_lstm_7k_seed42/best.ckpt \
  --output-dir local/tamu_monitoring/baseline

python scripts/monitor.py \
  --manifest configs/monitoring/tamu-ds1-gain-75-validation.yaml \
  --artifact local/issue4/checkpoints/issue4_ds1_gain_75_lstm_nano_seed42/best.ckpt \
  --output-dir local/tamu_monitoring/candidate

python scripts/compare_monitoring.py \
  --baseline-report local/tamu_monitoring/baseline/monitoring.json \
  --candidate-report local/tamu_monitoring/candidate/monitoring.json \
  --policy configs/monitoring/tamu-artifact-policy.yaml \
  --scenario controlled_rollback_failure \
  --output-dir local/tamu_monitoring/comparison
```

The policy rejects invalid or incomparable reports, output-contract failures, and ESR or MR-STFT increases above 10%. It investigates MSE or artifact-size increases above 10%, and p95 latency, real-time factor, or process peak RSS increases above 20%. Equality does not cross a threshold. Results retain raw values, absolute deltas, and relative deltas; the relative delta is unavailable when its baseline is zero or unavailable.

The example is a controlled rollback failure: it compares the accepted DS-1 LSTM-40 checkpoint with a preserved older LSTM-nano artifact. They are both LSTM-family artifacts, but differ in capacity and training/preparation recipe. Do not attribute a regression to a single difference. On rejection, retain the accepted baseline, do not promote the candidate, inspect the recorded version/config differences, correct the candidate, and rerun the same suite.

This is offline fixed-suite artifact monitoring, not production drift detection. Process RSS is whole-process high-water memory, checkpoint size can include training state, and latency is comparable only for matching runtime identity, inference category, chunk size, and workload.

## Published evidence

The final four-target reports are under [`monitoring/gear_comparison_44100/reports`](../monitoring/gear_comparison_44100/reports/). The selected checkpoints, result files, and the controlled rollback reports are in the [final evidence package](reproducibility.md). The rollback comparison remains a controlled failure example, not a deployment history.
