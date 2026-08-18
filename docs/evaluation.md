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

Create a comparison table directly from result files:

```bash
python scripts/compare_benchmarks.py results/*.json --output results/comparison.md
```

Do not use performance results from an untrained model as evidence of model
quality. Performance generally depends on architecture and size, but quality
metrics require a trained checkpoint and the controlled evaluation workflow.
