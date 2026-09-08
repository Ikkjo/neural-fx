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

The metric mask uses this precedence:

1. Manifest `burn_in_samples`
2. Dataset `metric_mask_first`
3. Checkpoint loss `mask_first`

Set `burn_in_samples` explicitly when several models must use one metric window. Listening files retain the complete aligned segment.

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
