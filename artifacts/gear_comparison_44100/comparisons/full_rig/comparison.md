# Model comparison

> ESR is the primary ranking metric. Lower ESR is better. MSE, correlation, and MR-STFT are secondary metrics. Ratios use candidate / LSTM-40; for ratio metrics, lower is better.

> Final experiment results. Interpret them with the recorded dataset, seeds, and hardware.

> Environment: 13th Gen Intel(R) Core(TM) i5-13400F; Linux-6.8.0-138-generic-x86_64-with-glibc2.39; PyTorch 2.11.0+cu130; float32; 1 thread(s); 3 warm-ups; 20 measurements.

> Block measurements cover stateful model-forward compute only. They exclude audio-interface latency, buffering, operating-system scheduling, and device round-trip. `met` means zero recorded deadline misses; it is not a hard real-time guarantee.

## Quality and model

| ESR rank | Experiment | Type | Parameters (relative) | State (bytes) | ESR (relative) | MSE (relative) | MR-STFT (relative) | STFT scored/excluded | Correlation |
| ---: | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |
| 1 | gear_comparison_44100_full_rig_wavenet_12k_seed42 | wavenet | 12,129 (1.75x (+75%)) | 48516 | 0.016105 (0.05x (-95%)) | 0.000106 (0.05x (-95%)) | 0.714613 (0.45x (-55%)) | 9/1 | 0.9920 |
| 2 | gear_comparison_44100_full_rig_gru_7k_seed42 | gru | 6,809 (0.98x (-2%)) | 27236 | 0.292360 (0.95x (-5%)) | 0.001916 (0.95x (-5%)) | 1.425079 (0.90x (-10%)) | 9/1 | 0.8427 |
| 3 | gear_comparison_44100_full_rig_lstm_7k_seed42 | lstm | 6,921 (1.00x (+0%)) | 27684 | 0.306729 (1.00x (+0%)) | 0.002010 (1.00x (+0%)) | 1.579493 (1.00x (+0%)) | 9/1 | 0.8346 |

## CPU

| Experiment | Offline RTF (relative) | 64 p95/deadline/misses (relative) | 128 p95/deadline/misses (relative) | 256 p95/deadline/misses (relative) | 512 p95/deadline/misses (relative) |
| --- | --- | --- | --- | --- | --- |
| gear_comparison_44100_full_rig_wavenet_12k_seed42 | 0.3632 (8.43x (+743%)) | 2.839/1.451/20 (24.39x (+2339%); missed) | 3.439/2.902/20 (21.21x (+2021%); missed) | 4.490/5.805/0 (18.45x (+1745%); met) | 6.499/11.610/0 (14.80x (+1380%); met) |
| gear_comparison_44100_full_rig_gru_7k_seed42 | 0.3627 (8.42x (+742%)) | 0.595/1.451/0 (5.11x (+411%); met) | 1.168/2.902/0 (7.20x (+620%); met) | 2.252/5.805/0 (9.25x (+825%); met) | 4.398/11.610/0 (10.01x (+901%); met) |
| gear_comparison_44100_full_rig_lstm_7k_seed42 | 0.0431 (1.00x (+0%)) | 0.116/1.451/0 (1.00x (+0%); met) | 0.162/2.902/0 (1.00x (+0%); met) | 0.243/5.805/0 (1.00x (+0%); met) | 0.439/11.610/0 (1.00x (+0%); met) |
