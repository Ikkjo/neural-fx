# TODO.md

## GitHub Project Board - Neural-FX Issues

This document lists tasks derived from the GitHub issues for the `neural-fx` project.

### [Phase 0] Foundation & Refactoring
- [x] Create BaseNeuralFX abstract class
- [x] Reorganize package directory structure
- [x] Implement unified YAML config system with validation
- [x] Fix NeuralfxLSTM output dimension mismatch
- [x] Fix NeuralfxGRU super().init bug
- [x] Complete RNNBlock initialization

### [Phase 1] LSTM/GRU Models
- [ ] Implement NeuralfxLSTM with correct output shape
- [ ] Implement NeuralfxGRU
- [ ] Add residual/skip connections
- [ ] Add conditioning input support (e.g., gain knob)
- [ ] Implement `process_sample()` for real-time inference
- [ ] Implement TBPTT (Truncated BPTT)
- [ ] Add hidden state burn-in
- [ ] Create size variants (nano/small/medium/large)

### [Phase 2] WaveNet Model
- [ ] Implement CausalConv1d
- [ ] Implement gated activation unit
- [ ] Implement DilatedResidualBlock
- [ ] Implement WaveNetStack w/ dilation pattern
- [ ] Add skip connections to output
- [ ] Implement fast cached inference
- [ ] Implement receptive field calculator
- [ ] Create WaveNet config variants

### [Phase 3] State Space Models (SSM)
- [ ] Evaluate SSM libraries: mamba-ssm, s4, safari
- [ ] Choose one: Mamba, S4, S5
- [ ] Implement S4 layer
- [ ] Implement S4Block with normalization
- [ ] Implement Mamba wrapper (if CUDA)
- [ ] Create SSM config variants

### [Phase 4] Unified Training System
- [ ] Create `NeuralFXLightningModule` wrapper
- [ ] Implement unified `AudioPairDataset`
- [ ] Add random segment sampling
- [ ] Add audio augmentations (gain, noise)
- [ ] Implement ESR loss w/ pre-emphasis
- [ ] Implement multi-resolution STFT loss
- [ ] Implement weighted combination loss
- [ ] Add loss masking for burn-in samples
- [ ] Create CLI training script w/ config loading
- [ ] Add model registry by config `type` field
- [ ] Implement checkpointing & resume

### [Phase 5] Inference & Export
- [ ] Implement streaming inference in `streaming.py`
- [ ] Handle state persistence between calls
- [ ] Export to TorchScript (`.ts`)
- [ ] Export to ONNX (`.onnx`)
- [ ] Export to RTNeural JSON
- [ ] Latency benchmarks
- [ ] Memory benchmarks

### [Phase 6] Evaluation & Comparison
- [ ] Implement ESR evaluation on test set
- [ ] Implement STFT distance metric
- [ ] Generate audio samples for listening tests
- [ ] Train all architectures on same dataset
- [ ] Create size-matched comparison
- [ ] Document quality vs. speed vs. size tradeoffs