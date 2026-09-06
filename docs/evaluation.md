# Evaluation and inference benchmarks

Use controlled evaluation to measure model quality on one held-out audio segment. Use inference benchmarks to measure runtime behavior.

Do not use an untrained benchmark model as quality evidence.

## Controlled quality evaluation

Copy the [evaluation manifest example](examples/evaluation-manifest.yaml). Resolve its paths relative to the manifest file.

Each final comparison must use the same aligned input, target, segment, latency correction, and metric window. Give smoke runs `run_kind: smoke`. Use `run_kind: final` only for complete controlled experiments.

Evaluate one checkpoint:

```bash
python scripts/evaluate_experiment.py \
  --manifest docs/examples/evaluation-manifest.yaml \
  --output-dir results/lstm-small \
  --device cpu
```

The command resets model state once and carries it across inference chunks. Use `--chunk-size` to override the manifest value.

The result directory contains:

- `evaluation.json`
- `input.wav`
- `target.wav`
- `prediction.wav`

Quality metrics include ESR, MSE, correlation, and multi-resolution STFT distance. ESR, MSE, and correlation cover the complete post-mask segment.

MR-STFT uses up to ten fixed, uniformly placed, non-overlapping three-second windows. The report records each window start and value.

### Digital silence policy

Evaluation results use `digital_silence_v1`. A scoring segment is digital silence only when every target sample is exactly zero; quiet nonzero targets remain eligible. ESR and MR-STFT are unavailable for silent targets, so JSON stores `null`, Markdown renders `N/A`, and averages use eligible windows only. MSE, prediction RMS, and prediction absolute peak remain recorded for every segment. The complete post-burn-in segment is used for ESR and MSE; the fixed MR-STFT windows are unchanged and each window records its status, absolute diagnostics, and scored/excluded counts.

The evaluation result schema is `1.1`; input manifests remain schema `1.0`. Comparisons reject missing or mismatched silence/STFT recipes, and unavailable ESR values receive no rank.

The metric mask uses this precedence:

1. Manifest `burn_in_samples`
2. Dataset `metric_mask_first`
3. Checkpoint loss `mask_first`

Set `burn_in_samples` explicitly when several models must use one metric window. Listening files retain the complete aligned segment.

## NM course comparison

The four-device NM comparison uses `lstm_7k` (LSTM-40), a single-layer LSTM with 40 hidden units, as the reference model for `ds1_gain_75`, `tsmini_gain_75`, `pearl_clean_sm57`, and `full_rig`. The [gear comparison experiment](../configs/experiments/gear_comparison_44100/experiment.yaml) defines the shared data, training, evaluation, and benchmark settings.

| Model and config | Architecture | Parameters | Role |
| --- | --- | ---: | --- |
| LSTM-40, [`lstm_7k.yaml`](../configs/models/lstm/lstm_7k.yaml) | One LSTM layer with 40 hidden units, scalar input and linear scalar output, an input-to-output skip connection, and no convolution | 6,921 | NM reference for all four targets |
| GRU-46, [`gru_7k.yaml`](../configs/models/gru/gru_7k.yaml) | One GRU layer with 46 hidden units, scalar input and linear scalar output, an input-to-output skip connection, and no convolution | 6,809 | Approximately parameter-matched alternative |
| WaveNet, [`wavenet_12k.yaml`](../configs/models/wavenet/wavenet_12k.yaml) | Causal dilated convolutional network | 12,129 | Larger convolutional alternative |

The 40-unit LSTM size follows the architecture size documented by [GuitarML Proteus](https://github.com/GuitarML/Proteus/blob/main/README.md). That source supports the hidden-size choice only. This project uses its own implementation, data, training procedure, and trained checkpoints. GRU-46 matches the LSTM parameter count approximately, while WaveNet has a larger parameter budget. The comparison does not require either alternative to outperform the reference model.

The saved comparison files are ignored local artifacts. They exist in this working tree and are absent from a public clone.

| Target | Local-only comparison |
| --- | --- |
| `ds1_gain_75` | [`local/gear_comparison_44100/results/comparisons/ds1_gain_75/comparison.md`](../local/gear_comparison_44100/results/comparisons/ds1_gain_75/comparison.md) |
| `tsmini_gain_75` | [`local/gear_comparison_44100/results/comparisons/tsmini_gain_75/comparison.md`](../local/gear_comparison_44100/results/comparisons/tsmini_gain_75/comparison.md) |
| `pearl_clean_sm57` | [`local/gear_comparison_44100/results/comparisons/pearl_clean_sm57/comparison.md`](../local/gear_comparison_44100/results/comparisons/pearl_clean_sm57/comparison.md) |
| `full_rig` | [`local/gear_comparison_44100/results/comparisons/full_rig/comparison.md`](../local/gear_comparison_44100/results/comparisons/full_rig/comparison.md) |

The saved rows retain their measured ESR ranking. Naming LSTM-40 as the reference does not change the ranking or any reported result.

Use this caption for the four-device comparison:

> Four-device model comparison using LSTM-40 as the reference architecture. GRU-46 has approximately the same parameter count; WaveNet is larger. Rows retain their measured ESR ranking.

The NM reference identifies the architecture used as the comparison point for the four-device quality and runtime experiment. The TAMU baseline identifies a previously accepted artifact in a same-device model-version monitoring comparison. These roles are separate. The TAMU baseline does not replace the NM reference architecture.

## Compare quality results

Compare results from the same dataset segment:

```bash
python scripts/compare_evaluations.py \
  results/lstm/evaluation.json \
  results/gru/evaluation.json \
  --output-dir results/comparison \
  --size-tolerance 1.35
```

The command writes `comparison.json` and `comparison.md`. It preserves metrics, model sizes, sources, listening samples, and linked benchmark measurements.

The size tolerance groups models by measured parameter count. The report does not select a winner or apply a regression policy.

## Benchmark inference

Run one model per process so peak CPU memory remains comparable:

```bash
python scripts/benchmark.py \
  --config configs/models/lstm/lstm_small.yaml \
  --checkpoint lightning_logs/lstm_small/last.ckpt \
  --output results/lstm-small-benchmark.json \
  --device cpu \
  --threads 2
```

New checkpoints embed their config, so `--config` is optional for them. A config without a checkpoint benchmarks an initialized model.

The benchmark excludes warm-up runs and synchronizes CUDA around timed regions. It measures whole-buffer and stateful block inference.

The JSON result records the workload, runtime, latency distribution, real-time factor, deadline misses, model-state size, and process peak memory.

Create a Markdown table from several benchmark results:

```bash
python scripts/compare_benchmarks.py \
  results/lstm-small-benchmark.json \
  results/gru-small-benchmark.json \
  --output results/benchmark-comparison.md
```

Pass a benchmark result path in each evaluation manifest when the quality comparison must include runtime measurements.
