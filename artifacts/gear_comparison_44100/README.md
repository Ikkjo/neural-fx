# Published artifact paths

Paths beginning with `artifacts/` or `configs/` in the JSON and Markdown files
are repository-root-relative references to files published with this evidence
package or tracked by the repository. They are the paths to use when inspecting
the saved results.

The `private-curated-dataset/` paths in evaluation metadata identify the source
segments used to produce the results. The private source files are not
published, so those paths are provenance only and are expected not to resolve in
this checkout.

The checkpoints retain the training configuration embedded at run time. Its
`local/...` dataset paths and historical experiment names are provenance from
the training workspace. Loading a published checkpoint uses its embedded model
configuration and weights; it does not need those training files. The
`issue4_ds1_gain_75_lstm_nano_seed42` checkpoint is the retained TAMU candidate
for the documented rollback comparison, not a promoted final model.

Verify the published files from this directory with:

```bash
sha256sum -c SHA256SUMS
```
