# Model comparison

> ESR is the primary ranking metric. Lower ESR is better. MSE, correlation, and MR-STFT are secondary metrics. Ratios use candidate / LSTM-40; for ratio metrics, lower is better.

> Final experiment results. Interpret them with the recorded dataset, seeds, and hardware.

> Environment: 13th Gen Intel(R) Core(TM) i5-13400F; Linux-6.8.0-138-generic-x86_64-with-glibc2.39; PyTorch 2.11.0+cu130; float32; 1 thread(s); 3 warm-ups; 20 measurements.

> Block measurements cover stateful model-forward compute only. They exclude audio-interface latency, buffering, operating-system scheduling, and device round-trip. `met` means zero recorded deadline misses; it is not a hard real-time guarantee.

## Quality and model

| ESR rank | Experiment | Type | Parameters (relative) | State (bytes) | ESR (relative) | MSE (relative) | MR-STFT (relative) | STFT scored/excluded | Correlation |
| ---: | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |
| 1 | gear_comparison_44100_pearl_clean_sm57_wavenet_12k_seed42 | wavenet | 12,129 (1.75x (+75%)) | 48516 | 0.019712 (0.41x (-59%)) | 0.000154 (0.41x (-59%)) | 1.572783 (0.93x (-7%)) | 10/0 | 0.9904 |
| 2 | gear_comparison_44100_pearl_clean_sm57_gru_7k_seed42 | gru | 6,809 (0.98x (-2%)) | 27236 | 0.041467 (0.86x (-14%)) | 0.000325 (0.86x (-14%)) | 1.830929 (1.08x (+8%)) | 10/0 | 0.9791 |
| 3 | gear_comparison_44100_pearl_clean_sm57_lstm_7k_seed42 | lstm | 6,921 (1.00x (+0%)) | 27684 | 0.048418 (1.00x (+0%)) | 0.000379 (1.00x (+0%)) | 1.693819 (1.00x (+0%)) | 10/0 | 0.9761 |

## CPU

| Experiment | Offline RTF (relative) | 64 p95/deadline/misses (relative) | 128 p95/deadline/misses (relative) | 256 p95/deadline/misses (relative) | 512 p95/deadline/misses (relative) |
| --- | --- | --- | --- | --- | --- |
| gear_comparison_44100_pearl_clean_sm57_wavenet_12k_seed42 | 0.2727 (6.89x (+589%)) | 2.588/1.451/20 (23.43x (+2243%); missed) | 2.919/2.902/6 (20.00x (+1900%); missed) | 4.172/5.805/0 (19.05x (+1805%); met) | 6.034/11.610/0 (15.68x (+1468%); met) |
| gear_comparison_44100_pearl_clean_sm57_gru_7k_seed42 | 0.3471 (8.77x (+777%)) | 0.612/1.451/0 (5.54x (+454%); met) | 1.091/2.902/0 (7.47x (+647%); met) | 2.229/5.805/0 (10.18x (+918%); met) | 4.201/11.610/0 (10.91x (+991%); met) |
| gear_comparison_44100_pearl_clean_sm57_lstm_7k_seed42 | 0.0396 (1.00x (+0%)) | 0.110/1.451/0 (1.00x (+0%); met) | 0.146/2.902/0 (1.00x (+0%); met) | 0.219/5.805/0 (1.00x (+0%); met) | 0.385/11.610/0 (1.00x (+0%); met) |
