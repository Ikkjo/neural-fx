# Benchmarking and model comparison

Run one model per process so the CPU peak-RSS value is comparable between models:

```bash
python scripts/benchmark.py \
  --config configs/models/lstm/lstm_nano.yaml \
  --checkpoint path/to/model.ckpt \
  --output results/lstm_nano.json \
  --device cpu \
  --threads 2
```

The checkpoint is optional for performance-only measurements. The script excludes
warm-up runs, synchronizes CUDA around timed regions, and measures both whole-file
and stateful block inference. Each JSON result includes the model input, workload,
runtime environment, latency distribution, real-time factor, deadline misses,
model-state size, and process peak memory.

New training checkpoints embed the complete neural-fx configuration. For those
checkpoints, `--config` can be omitted. Keep passing `--config` for an untrained
performance run or a legacy checkpoint.

Create a comparison table directly from result files:

```bash
python scripts/compare_benchmarks.py results/*.json --output results/comparison.md
```

Do not use performance results from an untrained model as evidence of model
quality. Performance generally depends on architecture and size, but quality
metrics require a trained checkpoint and the controlled evaluation workflow.

## Controlled quality evaluation

Record each experiment in a manifest. Paths can be absolute or relative to the
manifest:

```yaml
schema_version: "1.0"
experiment_id: lstm-nano-seed-42
run_kind: final  # use smoke for partial workflow-validation runs
# Reset model state once, then retain it across sequential chunks.
inference_chunk_size: 65536
# Use the same post-reset burn-in for every architecture.
burn_in_samples: 4096
model:
  # Optional for new checkpoints that embed neural_fx_config.
  config: ../../configs/models/lstm/lstm_nano.yaml
  checkpoint: checkpoints/lstm-nano-seed-42.ckpt
  benchmark_result: results/lstm_nano.json
dataset:
  input_audio: data/DI.wav
  target_audio: data/pearl_clean_sm57.wav
  split: test
  start_sample: 480000
  num_samples: 144000
  latency_samples: 0
  normalization: paired_peak
  # Optional shared comparison window; defaults to checkpoint loss.mask_first.
  metric_mask_first: 4096
training:
  seed: 42
  epochs: 100
  early_stopping: val_loss
notes: Held-out temporal test split.
```

Evaluate the checkpoint and create aligned input, target, and prediction WAVs:

```bash
python scripts/evaluate_experiment.py \
  --manifest experiments/lstm-nano-seed-42.yaml \
  --output-dir results/lstm-nano-seed-42 \
  --chunk-size 65536
```

Evaluation resets model state once and carries it between chunks. ESR, MSE, and
correlation cover the complete post-burn segment. MR-STFT is the mean from up to
ten fixed, uniformly placed, non-overlapping three-second windows; the result
records every window start and value.

Combine the three seed results for every architecture with three fresh benchmark
processes per representative checkpoint. The architecture report retains all raw
seed values and reports mean, sample standard deviation, median, minimum, and
maximum. The median-ESR seed supplies the listening samples and benchmark target.

```bash
python scripts/compare_evaluations.py results/*/evaluation.json \
  --benchmarks benchmarks/*.json \
  --output-dir results/comparison \
  --size-tolerance 1.01
```

`comparison.json` and `comparison.md` contain the architecture aggregation and
the preregistered conclusion-rule output. `seed-comparison.json` and
`seed-comparison.md` retain the per-run view. Missing seeds or benchmarks produce
an explicit incomplete conclusion. A complete result that fails any quality rule
states `no clear quality winner under this budget`.

Only complete, controlled runs should use `run_kind: final`. Smoke runs validate
the workflow but do not support a final model ranking.

Evaluation uses the same paired normalization and latency-compensation behavior as
training. By default, quality metrics exclude `burn_in_samples`, then
`dataset.metric_mask_first`, then `loss.mask_first` in that precedence order, and
use the configured ESR pre-emphasis setting. Listening files retain the full
aligned segment. Set `burn_in_samples` in every manifest when models must be
compared over one identical metric window.
