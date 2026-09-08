# Model comparison

> ESR is the primary ranking metric. Lower ESR is better. MSE, correlation, and MR-STFT are secondary metrics. Ratios use candidate / LSTM-40; for ratio metrics, lower is better.

> Final experiment results. Interpret them with the recorded dataset, seeds, and hardware.

> Environment: 13th Gen Intel(R) Core(TM) i5-13400F; Linux-6.8.0-138-generic-x86_64-with-glibc2.39; PyTorch 2.11.0+cu130; float32; 1 thread(s); 3 warm-ups; 20 measurements.

> Block measurements cover stateful model-forward compute only. They exclude audio-interface latency, buffering, operating-system scheduling, and device round-trip. `met` means zero recorded deadline misses; it is not a hard real-time guarantee.

## Quality and model

| ESR rank | Experiment | Type | Parameters (relative) | State (bytes) | ESR (relative) | MSE (relative) | MR-STFT (relative) | STFT scored/excluded | Correlation |
| ---: | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |
| 1 | gear_comparison_44100_tsmini_gain_75_wavenet_12k_seed42 | wavenet | 12,129 (1.75x (+75%)) | 48516 | 0.000293 (0.30x (-70%)) | 0.000004 (0.30x (-70%)) | 0.545602 (0.83x (-17%)) | 9/1 | 0.9999 |
| 2 | gear_comparison_44100_tsmini_gain_75_gru_7k_seed42 | gru | 6,809 (0.98x (-2%)) | 27236 | 0.000585 (0.60x (-40%)) | 0.000007 (0.60x (-40%)) | 0.615631 (0.94x (-6%)) | 9/1 | 0.9997 |
| 3 | gear_comparison_44100_tsmini_gain_75_lstm_7k_seed42 | lstm | 6,921 (1.00x (+0%)) | 27684 | 0.000974 (1.00x (+0%)) | 0.000012 (1.00x (+0%)) | 0.656517 (1.00x (+0%)) | 9/1 | 0.9995 |

## CPU

| Experiment | Offline RTF (relative) | 64 p95/deadline/misses (relative) | 128 p95/deadline/misses (relative) | 256 p95/deadline/misses (relative) | 512 p95/deadline/misses (relative) |
| --- | --- | --- | --- | --- | --- |
| gear_comparison_44100_tsmini_gain_75_wavenet_12k_seed42 | 0.3252 (8.13x (+713%)) | 2.942/1.451/20 (27.64x (+2664%); missed) | 3.365/2.902/20 (20.60x (+1960%); missed) | 4.294/5.805/0 (18.89x (+1789%); met) | 6.149/11.610/0 (15.93x (+1493%); met) |
| gear_comparison_44100_tsmini_gain_75_gru_7k_seed42 | 0.3606 (9.01x (+801%)) | 0.613/1.451/0 (5.75x (+475%); met) | 1.154/2.902/0 (7.07x (+607%); met) | 2.247/5.805/0 (9.89x (+889%); met) | 4.418/11.610/0 (11.45x (+1045%); met) |
| gear_comparison_44100_tsmini_gain_75_lstm_7k_seed42 | 0.0400 (1.00x (+0%)) | 0.106/1.451/0 (1.00x (+0%); met) | 0.163/2.902/0 (1.00x (+0%); met) | 0.227/5.805/0 (1.00x (+0%); met) | 0.386/11.610/0 (1.00x (+0%); met) |
