# TESTING.md - Testing Guidelines for neural-fx

This document outlines the testing infrastructure, guidelines, and best practices for the neural-fx project.

## Overview

The neural-fx testing suite uses **pytest** as the primary testing framework. Tests are organized under the `tests/` directory and follow a structured approach for validating model implementations, data processing, and training workflows.

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_recurrent.py

# Run specific test class
pytest tests/test_recurrent.py::TestRecurrentModels

# Run specific test method
pytest tests/test_recurrent.py::TestRecurrentModels::test_lstm_forward_shape

# Run with verbose output
pytest -v tests/test_recurrent.py

# Run with coverage report
pytest --cov=neural_fx

# Run with coverage and HTML report
pytest --cov=neural_fx --cov-report=html

# Run tests matching a pattern
pytest -k "lstm"

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Test Discovery

pytest automatically discovers tests in files matching:
- `test_*.py`
- `*_test.py`

## Testing Philosophy

### When to Write Tests

**DO write tests for:**
- Complex model architectures (LSTM, GRU, WaveNet, SSM)
- State management in recurrent models
- Data transformations and preprocessing
- Configuration loading and validation
- Export functionality (ONNX, TorchScript, RTNeural)
- Critical training logic

**AVOID writing tests for:**
- Trivial features already covered by PyTorch's own tests
- Simple getters/setters
- Boilerplate configuration code
- Duplicate coverage of the same functionality

### Testing Strategy

1. **Avoid redundancy**: Do not write tests if the feature is trivial or already covered by the underlying framework (e.g., PyTorch's own module tests).

2. **Verify complex logic**: If a test is the best way to verify a complex fix or feature, create a targeted test script or unit test.

3. **Manual verification**: Use simple scripts to instantiate classes or run a forward pass to catch basic errors during development.

4. **Integration over unit tests**: For neural networks, prefer integration tests that verify end-to-end behavior (forward pass shapes, state management) over mocking internal components.

## Test Structure

### Organization

```
tests/
├── test_recurrent.py      # Tests for LSTM/GRU models
├── test_wavenet.py        # Tests for WaveNet models (future)
├── test_ssm.py            # Tests for SSM models (future)
├── test_data.py           # Tests for data loading (future)
├── test_config.py         # Tests for configuration (future)
└── test_training.py       # Tests for training logic (future)
```

### Test Class Structure

Tests should be organized into classes grouped by functionality:

```python
import torch
import pytest
import sys
import os

# Ensure the package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural_fx.config import ModelConfig, LSTMParams, Conv1dConfig
from neural_fx.models.recurrent import NeuralfxLSTM, NeuralfxGRU


class TestRecurrentModels:
    """Test suite for LSTM and GRU models."""
    
    @pytest.fixture
    def lstm_config(self):
        """Provide a standard LSTM configuration for tests."""
        params = LSTMParams(
            hidden_size=20,
            num_layers=1,
            conv1d=Conv1dConfig(filters=16, kernel_size=3, stride=4),
            conditioning_size=2
        )
        return ModelConfig(
            type="lstm",
            params=params,
            input_size=1,
            output_size=1,
            sample_rate=48000
        )
    
    @pytest.fixture
    def gru_config(self):
        """Provide a standard GRU configuration for tests."""
        params = LSTMParams(
            hidden_size=20,
            num_layers=1,
            conv1d=None,
            conditioning_size=0
        )
        return ModelConfig(
            type="gru",
            params=params,
            input_size=1,
            output_size=1
        )
```

## Required Test Coverage

### 1. Model Architecture Tests

**Forward Pass Shape Validation**
- Verify output shape matches input shape
- Test with different batch sizes
- Test with and without convolution layers
- Test with and without conditioning

```python
def test_lstm_forward_shape(self, lstm_config):
    """Test LSTM forward pass output shape with convolution and conditioning."""
    model = NeuralfxLSTM(lstm_config)
    
    # Input: [Batch, Channels, Time]
    # Length must be divisible by stride if using convolution
    x = torch.randn(2, 1, 1024) 
    cond = torch.randn(2, 2)  # Conditioning [Batch, C_cond]
    
    y = model(x, conditioning=cond)
    
    assert y.shape == x.shape, f"Output shape mismatch! {y.shape} != {x.shape}"
```

### 2. State Management Tests

For recurrent models, test state handling:

```python
def test_state_management(self, lstm_config):
    """Test resetting and detaching state."""
    model = NeuralfxLSTM(lstm_config)
    x = torch.randn(1, 1, 100)
    
    # Run once to populate state
    model(x)
    assert model.hidden_state is not None
    
    # Detach state for TBPTT
    model.detach_state()
    if isinstance(model.hidden_state, tuple):
        assert not model.hidden_state[0].requires_grad
    else:
        assert not model.hidden_state.requires_grad
        
    # Reset state
    model.reset_state()
    assert model.hidden_state is None
```

### 3. Single-Sample Processing Tests

Test the `process_sample()` method used for real-time inference:

```python
def test_process_sample(self, lstm_config):
    """Test single sample processing."""
    model = NeuralfxLSTM(lstm_config)
    model.eval()
    
    # Single sample input [Channels]
    x = torch.randn(1)
    y = model.process_sample(x)
    
    assert y.ndim == 0 or y.ndim == 1
    assert model.hidden_state is not None
```

### 4. Configuration Tests

Test model instantiation from configurations:

```python
def test_model_from_config(self):
    """Test model creation from configuration dictionary."""
    config_dict = {
        "type": "lstm",
        "params": {
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.1
        },
        "input_size": 1,
        "output_size": 1,
        "sample_rate": 48000
    }
    
    model = RecurrentNeuralFXModel.from_config(config_dict)
    assert isinstance(model, NeuralfxLSTM)
    assert model.params.hidden_size == 32
```

## Testing Best Practices

### 1. Use Fixtures for Common Setup

```python
@pytest.fixture
def sample_audio_tensor():
    """Provide a sample audio tensor for testing."""
    return torch.randn(2, 1, 1024)  # [Batch, Channels, Time]

@pytest.fixture
def device():
    """Provide the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. Test Edge Cases

```python
def test_batch_size_one(self, lstm_config):
    """Test with batch size of 1."""
    model = NeuralfxLSTM(lstm_config)
    x = torch.randn(1, 1, 100)
    y = model(x)
    assert y.shape == x.shape

def test_no_conditioning(self, lstm_config):
    """Test model without providing conditioning."""
    model = NeuralfxLSTM(lstm_config)
    x = torch.randn(2, 1, 1024)
    # Should auto-generate zero conditioning
    y = model(x, conditioning=None)
    assert y.shape == x.shape
```

### 3. Test Error Handling

```python
def test_invalid_conditioning_shape(self, lstm_config):
    """Test that invalid conditioning shapes raise errors."""
    model = NeuralfxLSTM(lstm_config)
    x = torch.randn(2, 1, 1024)
    # Wrong conditioning size
    cond = torch.randn(2, 5)  # Config expects 2
    
    with pytest.raises(ValueError, match="Expected 2 conditioning channels"):
        model(x, conditioning=cond)
```

### 4. Use Meaningful Test Names

Test names should describe what is being tested:
- `test_lstm_forward_shape` - Good
- `test_lstm` - Too vague
- `test_state_management` - Good
- `test_1` - Bad

### 5. Add Docstrings to Tests

```python
def test_lstm_forget_gate_bias(self, lstm_config):
    """
    Test that LSTM forget gate bias is initialized to 1.0.
    
    This is important for gradient flow in long sequences.
    See: Gers et al. (2000) "Learning to Forget: Continual Prediction with LSTM"
    """
    model = NeuralfxLSTM(lstm_config)
    # Check bias values...
```

## Dependencies

Testing requires the following packages (install with pip):

```bash
pip install pytest pytest-cov

# Optional but recommended
pip install pytest-xdist    # Parallel test execution
pip install pytest-mock     # Mocking support
pip install hypothesis      # Property-based testing
```

## Continuous Integration

When setting up CI/CD, use:

```bash
# Install dependencies
pip install -e .
pip install pytest pytest-cov

# Run tests with coverage
pytest --cov=neural_fx --cov-report=xml --cov-fail-under=80

# Run linting before tests
ruff check neural_fx/
mypy neural_fx/
```

## Current Test Coverage

### Implemented Tests

**`tests/test_recurrent.py`** - LSTM and GRU model tests
- `test_lstm_forward_shape`: Validates output shape with conv + conditioning
- `test_gru_forward_shape`: Validates output shape without convolution
- `test_state_management`: Tests state reset, detach, and persistence
- `test_process_sample`: Tests single-sample inference mode

### Planned Test Coverage

Future test files to implement:
- `tests/test_wavenet.py`: WaveNet model architecture tests
- `tests/test_ssm.py`: State-space model tests (Mamba, S4)
- `tests/test_data.py`: AudioDataset and data transform tests
- `tests/test_config.py`: Configuration loading and validation tests
- `tests/test_export.py`: ONNX, TorchScript, RTNeural export tests
- `tests/test_training.py`: LightningModule and training loop tests

## Writing Tests for New Models

When adding a new model type:

1. **Create test file**: `tests/test_<model>.py`

2. **Required test coverage**:
   - Forward pass shape validation
   - State management (if applicable)
   - `process_sample()` single-sample mode
   - Configuration loading via `from_config()`
   - Skip connections (if applicable)
   - Conditioning support (if applicable)

3. **Example template**:

```python
import torch
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural_fx.config import ModelConfig, YourModelParams
from neural_fx.models.your_model import YourModel


class TestYourModel:
    @pytest.fixture
    def model_config(self):
        params = YourModelParams(...)
        return ModelConfig(type="your_model", params=params, ...)
    
    def test_forward_shape(self, model_config):
        model = YourModel(model_config)
        x = torch.randn(2, 1, 1000)
        y = model(x)
        assert y.shape == x.shape
    
    def test_from_config(self):
        # Test factory method
        pass
    
    def test_state_management(self, model_config):
        # If model has state
        pass
    
    def test_process_sample(self, model_config):
        # Test real-time inference mode
        pass
```

## References

- [pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing Best Practices](https://pytorch.org/docs/stable/testing.html)
- Project issues: See GitHub issue #10 "[Phase 1] LSTM/GRU Models" for related testing work
