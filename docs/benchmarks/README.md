# Neural-FX benchmarks

The benchmark CLI measures stateful streaming inference latency for one or
more model configurations. Randomly initialized models are valid for latency
testing; checkpoints are optional because weight values do not change model
structure or operation count.

```bash
python scripts/benchmark.py \
  configs/models/lstm/lstm_nano.yaml \
  configs/models/gru/gru_nano.yaml \
  --device cpu \
  --threads 1 \
  --warmup 20 \
  --iterations 100 \
  --block-sizes 1,64,128,256,512,1024 \
  --output-dir benchmarks \
  --name recurrent-nano
```

To benchmark trained weights, pass one checkpoint per config in the same
order:

```bash
python scripts/benchmark.py configs/models/lstm/lstm_nano.yaml \
  --checkpoints checkpoints/lstm.ckpt
```

The command emits versioned JSON, flat CSV, and a Markdown comparison table.
Each result records the git revision, operating system, CPU/GPU, Python and
PyTorch versions, thread count, seed, model identity, parameter size, and
timing settings. The reported real-time factor is median processing time
divided by the corresponding audio-block duration; values below `1.0` are
faster than real time.

For comparable CPU results, keep `--threads 1`, close other compute-heavy
applications, and run all compared models on the same machine in one command.
