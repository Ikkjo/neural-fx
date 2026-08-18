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
  --output-dir results/lstm-nano-seed-42
```

Combine experiment result files into JSON and Markdown reports. Models whose
measured parameter counts are within the requested ratio are placed in the same
size-matched group.

```bash
python scripts/compare_evaluations.py results/*/evaluation.json \
  --output-dir results/comparison \
  --size-tolerance 1.35
```

The report labels any comparison containing a smoke run as workflow validation.
Only complete, controlled runs should use `run_kind: final`.

Evaluation uses the same paired normalization and latency-compensation behavior as
training. By default, quality metrics exclude `loss.mask_first` samples and use the
configured ESR pre-emphasis setting; listening files retain the full aligned segment.
Set `dataset.metric_mask_first` in every manifest when models have different
training masks but must be compared over one identical metric window.
