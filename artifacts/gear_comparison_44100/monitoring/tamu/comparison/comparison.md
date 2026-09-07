# TAMU artifact comparison

Scenario: `controlled_rollback_failure`. This is a controlled rollback failure, not a claim that the historical artifact was deployed after the baseline.

Suite `tamu-ds1-gain-75-validation` (`65e1f2a131e0a259018d6c785ca0718ee397dce2145f8eebd2e80efe9333b8ea`). Both reports used `native_stateful` inference, chunk `8192`, dtype `float32`, on `x86_64`.

Baseline `3f9a8875913e12fc059de1ed28489a426c998f965aa244400b465ef89b3f0f37`: `artifacts/gear_comparison_44100/checkpoints/gear_comparison_44100_ds1_gain_75_lstm_7k_seed42/best.ckpt`. Candidate `706c11e9ea96f4a1f4939be541f53480abebeb3f65fc0d0308515d126b96bcc5`: `artifacts/gear_comparison_44100/checkpoints/issue4_ds1_gain_75_lstm_nano_seed42/best.ckpt`.

| Metric | Baseline | Candidate | Delta | Change |
| --- | ---: | ---: | ---: | ---: |
| esr | 1.74994 | 4.05054 | 2.3006 | 131.468% |
| mse | 6.04225e-06 | 0.000728045 | 0.000722003 | 11949.2% |
| multi_resolution_stft_distance | 5.45507 | 6.27544 | 0.820374 | 15.0387% |
| p95_latency_ms | 46.3764 | 8.70809 | -37.6683 | -81.223% |
| real_time_factor | 0.032852 | 0.00779889 | -0.0250531 | -76.2605% |
| peak_memory_bytes | 8.55888e+08 | 8.5846e+08 | 2.57229e+06 | 0.30054% |
| artifact_size_bytes | 96283 | 55400 | -40883 | -42.4613% |

| Severity | Metric | Observed | Threshold | Crossed |
| --- | --- | ---: | ---: | --- |
| reject | esr | 131.468% | 10% | yes |
| reject | multi_resolution_stft_distance | 15.0387% | 10% | yes |
| investigate | mse | 11949.2% | 10% | yes |
| investigate | p95_latency_ms | -81.223% | 20% | no |
| investigate | real_time_factor | -76.2605% | 20% | no |
| investigate | peak_memory_bytes | 0.30054% | 20% | no |
| investigate | artifact_size_bytes | -42.4613% | 10% | no |

Decision: **reject**. Retain the accepted baseline, do not promote the candidate, inspect version/config differences, correct the candidate, and rerun the same validation suite before reconsideration.
