# Final experiment evidence

The published `gear_comparison_44100` evidence is under [`artifacts/gear_comparison_44100`](../artifacts/gear_comparison_44100/). It includes selected checkpoints, final benchmark/evaluation/comparison results, the TAMU controlled rollback example, and listening WAVs. Raw recordings and prepared train, validation, and test splits are not distributed with the repository.

## Retrieve and verify the evidence

Install Git LFS before cloning, or retrieve the files after cloning:

```bash
git lfs install
git lfs pull
cd artifacts/gear_comparison_44100
sha256sum -c SHA256SUMS
```

The 36 `input.wav`, `target.wav`, and `prediction.wav` files under `evaluations/` are listening material for the saved evaluation segments. They support informal inspection; they are not a listening study.

## Inspect checkpoints

The 12 selected checkpoint files are under `checkpoints/`. Each contains its model configuration and can be loaded without the private dataset:

```bash
python - <<'PY'
from neural_fx.artifacts import load_model

load_model(checkpoint_path="artifacts/gear_comparison_44100/checkpoints/gear_comparison_44100_ds1_gain_75_lstm_7k_seed42/best.ckpt")
PY
```

`environment.txt` records the resolved Python packages used for the final monitoring reports. Use the supplied benchmark, evaluation, comparison, and monitoring JSON files to inspect the published measurements without rerunning them.

## Rerun measurements

Rerunning evaluation or monitoring requires the separately supplied private curated dataset at the paths referenced by the tracked suite and experiment configuration files. It does not require retraining. Recreate a monitoring report with:

```bash
python scripts/monitor.py \
  --manifest monitoring/gear_comparison_44100/suites/ds1_gain_75.yaml \
  --artifact artifacts/gear_comparison_44100/checkpoints/gear_comparison_44100_ds1_gain_75_lstm_7k_seed42/best.ckpt \
  --output-dir results/monitoring-ds1-lstm \
  --html
```

The published monitoring reports measure fixed test cases. Use validation cases for tuning and reserve the test cases for final checks.
