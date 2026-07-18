# Comparison harness

The comparison core validates that architecture runs use the same explicit data
splits, declared seeds, and parameter budget. It then aggregates externally
produced numeric metrics and writes JSON, CSV, Markdown, and HTML reports.

The checked-in manifest and results are **synthetic schema fixtures**, not
project evaluation results. Real conclusions require trained checkpoints and a
held-out dataset.

## Manifest contract

`schema_version` is currently `"1.0"`. A manifest declares:

- Non-empty `train`, `validation`, and `test` lists of input/target audio paths.
- A target parameter count and fractional tolerance. The default is 60,000 ±10%.
- The complete seed set expected for each architecture/size combination.
- Shared training settings recorded for reproducibility.
- One model entry per architecture, size, and seed, including its config,
  checkpoint, and measured parameter count.

Validation rejects reused audio paths, duplicate runs, undeclared or missing
seeds, and parameter counts outside the budget. Paths are resolved relative to
the manifest. They are allowed to point to future artifacts unless
`--validate-files` is supplied.

## Result contract

Results use the same schema version and contain one record per manifest run:

```json
{
  "schema_version": "1.0",
  "records": [
    {"run_id": "lstm-medium-42", "metrics": {"esr": 0.12, "mse": 0.006}}
  ]
}
```

Every metric must be a finite number and every run in the manifest must appear
exactly once. Each architecture must expose the same metric names across its
seeds. Aggregates use arithmetic means and sample standard deviation (`n - 1`);
a one-seed experiment reports a standard deviation of zero.

## Generate synthetic reports

From the repository root:

```powershell
python scripts/compare.py `
  --manifest docs/comparison/synthetic_manifest.yaml `
  --results docs/comparison/synthetic_results.json `
  --output-dir comparison-output
```

Do not pass `--validate-files` for the checked-in synthetic fixture because its
audio, config, and checkpoint paths are intentionally placeholders. A real
experiment should use `--validate-files` so missing inputs fail before reporting.
