# Phase 1 Implementation Plan: LSTM/GRU Models

This document outlines the plan for implementing Phase 1 of the `neural-fx` project, focusing on Recurrent Neural Networks (LSTM and GRU) for audio effect modeling.

## Goal
To implement robust, efficient, and high-quality LSTM and GRU models capable of real-time audio effect emulation, including support for conditioning (control knobs) and stateful inference.

## Prerequisites
- `BaseNeuralFXModel` is defined in `neural-fx/models/base.py`.
- `LSTMParams` and `ModelConfig` are defined in `neural-fx/config.py`.

## Tasks

### 1. Core Recurrent Infrastructure (`neural-fx/models/recurrent.py`)

We need a flexible base class for recurrent models that handles common logic like input shaping, convolution (if used), and state management.

- **Action**: Create `RecurrentNeuralFXModel` class inheriting from `BaseNeuralFXModel`.
- **Responsibilities**:
    - Handle `from_config` instantiation.
    - Manage the optional input `Conv1d` layer (often used for learnable downsampling or feature extraction).
    - Manage the final `Linear` projection layer to the output.
    - Define abstract methods for the specific RNN core (LSTM vs GRU).

### 2. Implement `NeuralfxLSTM`

- **Action**: Implement `NeuralfxLSTM` class inheriting from `RecurrentNeuralFXModel`.
- **Key Features**:
    - Use `torch.nn.LSTM` as the core.
    - **Fix Output Dimension**: Ensure the output of the LSTM (batch, seq, hidden) is correctly projected back to audio channels (batch, channels, seq).
    - **Initialization**: Implement the "forget gate bias = 1" trick (or similar) to improve gradient flow at the start of training.

### 3. Implement `NeuralfxGRU`

- **Action**: Implement `NeuralfxGRU` class inheriting from `RecurrentNeuralFXModel`.
- **Key Features**:
    - Use `torch.nn.GRU` as the core.
    - Similar architecture to the LSTM version but often more efficient.

### 4. Enhancements

#### 4.1 Residual/Skip Connections
- **Requirement**: Allow the input signal to bypass the RNN and be added to the output. This is crucial for "wet/dry" mixes or modeling subtle effects where the original signal is largely preserved.
- **Implementation**: Add a `skip_connection` boolean to the config/init. If `True`, `output = predicted + input`.

#### 4.2 Conditioning Input Support
- **Requirement**: Support control knobs (e.g., Gain, Tone) as additional inputs.
- **Implementation**:
    - Modify `forward` to accept an optional `conditioning` tensor.
    - Concatenate conditioning data with the audio input (either at the raw audio level or after the initial convolution) before feeding it into the RNN.

### 5. Real-time Inference (`process_sample`)

- **Requirement**: The model must support processing one sample (or a small chunk) at a time, maintaining its hidden state between calls.
- **Implementation**:
    - Implement `process_sample(self, x: Tensor) -> Tensor`.
    - Maintain `self.hidden_state` as a persistent member.
    - Ensure shapes are handled correctly (removing/adding batch/sequence dimensions as needed for single-sample processing).

### 6. Training Features (TBPTT & Burn-in)

#### 6.1 Truncated Backpropagation Through Time (TBPTT)
- **Requirement**: Train on long sequences by processing them in chunks without backpropagating all the way to the beginning.
- **Implementation**:
    - Implement `detach_state()`: Detach the current hidden state from the computation graph (`h.detach()`).
    - The training loop (in Phase 4) will call this between batches.

#### 6.2 Hidden State Burn-in
- **Requirement**: The RNN's initial state (usually zeros) is incorrect. We need to run the model for a "warm-up" period before calculating loss to let the internal state settle.
- **Implementation**:
    - This is largely a loss function / training loop concern (masking the first N samples of the loss).
    - The model just needs to ensure it can run continuously.

### 7. Configuration Variants

Create standard YAML configurations to allow easy benchmarking of size/quality trade-offs.

- **Files**:
    - `configs/models/lstm/lstm_nano.yaml`: Very small, high speed.
    - `configs/models/lstm/lstm_small.yaml`
    - `configs/models/lstm/lstm_medium.yaml`
    - `configs/models/lstm/lstm_large.yaml`: High quality, slower.
    - Repeat for GRU.

## Next Steps
1.  Populate `neural-fx/models/recurrent.py` with the base class and implementations.
2.  Create the config files.
3.  Verify with a simple test script (instantiate model, run one forward pass).
