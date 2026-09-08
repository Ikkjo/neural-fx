# Model comparison

> ESR is the primary ranking metric. Lower ESR is better. MSE, correlation, and MR-STFT are secondary metrics. Ratios use candidate / LSTM-40; for ratio metrics, lower is better.

> Final experiment results. Interpret them with the recorded dataset, seeds, and hardware.

> Environment: 13th Gen Intel(R) Core(TM) i5-13400F; Linux-6.8.0-138-generic-x86_64-with-glibc2.39; PyTorch 2.11.0+cu130; float32; 1 thread(s); 3 warm-ups; 20 measurements.

> Block measurements cover stateful model-forward compute only. They exclude audio-interface latency, buffering, operating-system scheduling, and device round-trip. `met` means zero recorded deadline misses; it is not a hard real-time guarantee.

## Quality and model

| ESR rank | Experiment | Type | Parameters (relative) | State (bytes) | ESR (relative) | MSE (relative) | MR-STFT (relative) | STFT scored/excluded | Correlation |
| ---: | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |
| 1 | gear_comparison_44100_ds1_gain_75_wavenet_12k_seed42 | wavenet | 12,129 (1.75x (+75%)) | 48516 | 0.002556 (0.52x (-48%)) | 0.000050 (0.52x (-48%)) | 0.439029 (0.77x (-23%)) | 9/1 | 0.9988 |
| 2 | gear_comparison_44100_ds1_gain_75_gru_7k_seed42 | gru | 6,809 (0.98x (-2%)) | 27236 | 0.003409 (0.70x (-30%)) | 0.000066 (0.70x (-30%)) | 0.578990 (1.01x (+1%)) | 9/1 | 0.9983 |
| 3 | gear_comparison_44100_ds1_gain_75_lstm_7k_seed42 | lstm | 6,921 (1.00x (+0%)) | 27684 | 0.004901 (1.00x (+0%)) | 0.000095 (1.00x (+0%)) | 0.571802 (1.00x (+0%)) | 9/1 | 0.9977 |

## CPU

| Experiment | Offline RTF (relative) | 64 p95/deadline/misses (relative) | 128 p95/deadline/misses (relative) | 256 p95/deadline/misses (relative) | 512 p95/deadline/misses (relative) |
| --- | --- | --- | --- | --- | --- |
| gear_comparison_44100_ds1_gain_75_wavenet_12k_seed42 | 0.3798 (8.80x (+780%)) | 2.970/1.451/20 (28.65x (+2765%); missed) | 3.481/2.902/20 (24.17x (+2317%); missed) | 4.511/5.805/0 (20.37x (+1937%); met) | 6.590/11.610/0 (17.29x (+1629%); met) |
| gear_comparison_44100_ds1_gain_75_gru_7k_seed42 | 0.3470 (8.04x (+704%)) | 0.608/1.451/0 (5.87x (+487%); met) | 1.158/2.902/0 (8.04x (+704%); met) | 2.225/5.805/0 (10.05x (+905%); met) | 4.063/11.610/0 (10.66x (+966%); met) |
| gear_comparison_44100_ds1_gain_75_lstm_7k_seed42 | 0.0431 (1.00x (+0%)) | 0.104/1.451/0 (1.00x (+0%); met) | 0.144/2.902/0 (1.00x (+0%); met) | 0.221/5.805/0 (1.00x (+0%); met) | 0.381/11.610/0 (1.00x (+0%); met) |
