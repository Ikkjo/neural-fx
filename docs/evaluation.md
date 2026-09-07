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

The final comparison files and their associated listening WAVs are published in the repository evidence package. Retrieve the WAVs with Git LFS as described in the [reproducibility guide](reproducibility.md).

| Target | Comparison | Listening WAVs |
| --- | --- | --- |
| `ds1_gain_75` | [`comparison`](../artifacts/gear_comparison_44100/comparisons/ds1_gain_75/comparison.md) | [`evaluations`](../artifacts/gear_comparison_44100/evaluations/) |
| `tsmini_gain_75` | [`comparison`](../artifacts/gear_comparison_44100/comparisons/tsmini_gain_75/comparison.md) | [`evaluations`](../artifacts/gear_comparison_44100/evaluations/) |
| `pearl_clean_sm57` | [`comparison`](../artifacts/gear_comparison_44100/comparisons/pearl_clean_sm57/comparison.md) | [`evaluations`](../artifacts/gear_comparison_44100/evaluations/) |
| `full_rig` | [`comparison`](../artifacts/gear_comparison_44100/comparisons/full_rig/comparison.md) | [`evaluations`](../artifacts/gear_comparison_44100/evaluations/) |

The saved rows retain their measured ESR ranking. Naming LSTM-40 as the reference does not change the ranking or any reported result.

Use this caption for the four-device comparison:

> Four-device model comparison using LSTM-40 as the reference architecture. GRU-46 has approximately the same parameter count; WaveNet is larger. Rows retain their measured ESR ranking.

The NM reference identifies the architecture used as the comparison point for the four-device quality and runtime experiment. The TAMU baseline identifies a previously accepted artifact in a same-device model-version monitoring comparison. These roles are separate. The TAMU baseline does not replace the NM reference architecture.

### Dataset and experimental protocol

The experiment uses a privately curated, 14.76-minute mono guitar dataset stored as 44.1 kHz, 32-bit floating-point WAV. REAPER played the assembled DI signal through Focusrite Scarlett 2i4 output 1, the target chain, and Scarlett input 1. Each target was recorded during one continuous playback pass. The paired files received no gain adjustment, fades, EQ, noise reduction, compression, limiting, or normalization; disabling normalization preserves the recorded input-to-target gain relationship.

| Target | Capture chain | Capture detail |
| --- | --- | --- |
| `ds1_gain_75` | Scarlett output 1 → Boss DS-1 → Scarlett input 1 | Direct pedal output; instrument input mode is recalled, not verified. |
| `tsmini_gain_75` | Scarlett output 1 → Ibanez Tube Screamer Mini → Scarlett input 1 | Direct pedal output; instrument input mode is recalled, not verified. |
| `pearl_clean_sm57` | Scarlett output 1 → Pearl PFT 101 Dual Reverb → speaker → Shure SM57 → Scarlett input 1 | Microphone capture; line input mode is recalled, not verified. |
| `full_rig` | Scarlett output 1 → DS-1 → Pearl PFT 101 Dual Reverb → speaker → SM57 → Scarlett input 1 | DS-1 gain was near maximum; microphone capture and recalled line input mode. |

Only pedal gain/drive was swept; its percentage is encoded in the filename. Other pedal controls stayed fixed. The reported 75% setting is a representative middle setting between 50% and 100%, not a score-based choice. The Pearl settings were held fixed across amplifier captures. Capture photographs indicate approximate Pearl settings of Volume 5, Treble 7, Middle 4, Bass 5, Reverb 0, and Speed 0. The SM57 was recalled as on-axis, about one inch or less from the grille, near the dust-cap/cone boundary. These are approximate setup details, not calibrated measurements.

Preparation applies a common -41-sample alignment without normalization, producing 39,051,208 aligned samples. Contiguous train, validation, and final-evaluation segments contain 31,162,368, 3,891,200, and 3,907,584 samples, separated by two 44,100-sample guard gaps; 1,856 tail samples are unused. Target-specific cross-correlation estimates differed. Subsequent ESR comparisons on the final-evaluation segment selected -41 empirically across the captures; it was not the cross-correlation result for every target.

Within each run, `best.ckpt` is selected by validation loss, configured as un-pre-emphasized NAM ESR. Equivalent deterministic checkpoints were reused where available rather than retrained. The wider loss recipe, capacities, and alignment were developed iteratively after inspecting final-segment ESR.

The dataset is one privately curated performance collection, not independent recording sessions. Guard gaps reduce adjacent-window leakage without making performances independent. Results use one seed and fixed device/control settings; WaveNet has more parameters than the recurrent models. Test-guided protocol development, including alignment selection, means the final results describe this fixed dataset and protocol rather than an untouched estimate of generalization.

## Compare quality results

Compare results from the same dataset segment:

```bash
python scripts/compare_evaluations.py \
  results/lstm/evaluation.json \
  results/gru/evaluation.json \
  --reference-experiment-id lstm-run-id \
  --output-dir results/comparison \
  --size-tolerance 1.35
```

The command writes schema-`1.2` `comparison.json` and `comparison.md`. `--reference-experiment-id` is required and must identify exactly one of the supplied results. It preserves raw metrics, model sizes, sources, listening samples, and linked benchmark measurements.

The report contains separate quality/model and CPU tables. ESR, MSE, corrected MR-STFT, parameter count, offline RTF, and block p95 time include the raw value plus `candidate / LSTM-40` and `(candidate / LSTM-40 - 1) * 100`. For these lower-is-better measures, a ratio below `1.0x` is an improvement. A missing or zero reference produces `null` in JSON and `N/A` in Markdown. Correlation and model-state bytes remain raw; process peak RSS is not a model-memory comparison. Comparisons reject mismatched benchmark environments or workloads and do not average ratios across targets.

The CPU table retains 64, 128, 256, and 512-sample p95 compute time, deadline, and misses. The 128-sample case is the main discussion point. Block measurements cover stateful model-forward compute only; they exclude audio-interface latency, buffering, operating-system scheduling, and device round-trip. A zero-miss row met the measured compute deadline, which is not a hard real-time guarantee.

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
