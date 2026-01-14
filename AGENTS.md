# AGENTS.md - neural-fx Development Guide

## Project Overview
Neural-fx is a real-time guitar effect and amp modelling project using neural networks (LSTM, GRU, WaveNet, SSM/Mamba/S4). Built with PyTorch and PyTorch Lightning.

## Build & Development Commands

### Installation
```bash
pip install -e .
```

### Training Models
```bash
# From Jupyter notebooks
jupyter notebook notebooks/train_recurrent.ipynb

# Or run via Python directly
python scripts/train.py --config configs/models/lstm/config.yaml
```

### Exporting Models
```bash
python scripts/export.py --config configs/models/lstm/config.yaml --checkpoint lightning_logs/version_X/checkpoints/*.ckpt
```

### Running Tests
```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_models.py

# Run single test
pytest tests/test_models.py::test_lstm_forward

# Run with coverage
pytest --cov=neural_fx
```

### Linting & Type Checking
```bash
# Lint with ruff
ruff check neural_fx/

# Auto-fix issues
ruff check --fix neural_fx/

# Type check with mypy
mypy neural_fx/

# Format code
ruff format neural_fx/
```

---

## Code Style Guidelines

### Imports
Organize imports in three groups separated by blank lines:
1. Standard library imports (`from pathlib import Path`, `from typing import *`)
2. Third-party imports (`import torch`, `import lightning as L`)
3. Local imports (`from neural_fx.data import dataset`)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

import torch.nn as nn
from torch import Tensor

from neural_fx.config import ModelConfig
```

### Formatting
- Use 4 spaces for indentation (no tabs)
- Line length: 120 characters recommended
- Use trailing commas in multi-line dataclasses and function calls
- Blank lines: 2 between class definitions, 1 between function definitions within classes
- No blank line between type annotations and function body

### Type Hints
- Use type hints for all function signatures
- Prefer `from typing import *` for cleaner imports in smaller modules
- Use `| ` syntax for unions (Python 3.10+): `str | None`
- Define common type aliases at module level:
```python
StateType = Optional[Union[Tensor, Tuple[Tensor, ...], List[Tensor]]]
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `BaseNeuralFXModel`, `WaveNetParams` |
| Functions/Variables | snake_case | `load_config`, `sample_rate` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_DATASET_DIR` |
| Private methods | snake_case with leading underscore | `def _load_model_params()` |
| Type aliases | PascalCase ending in Type | `ModelParamsType` |

### Dataclasses for Configuration
Use `@dataclass` for all configuration objects:
```python
@dataclass
class Conv1dConfig:
    filters: int
    kernel_size: int = 3
    stride: int = 1
```

### Error Handling
- Use specific exceptions (`ValueError`, `FileNotFoundError`, `KeyError`)
- Provide descriptive error messages:
```python
if cls is None:
    raise ValueError(f"Unknown model type: {model_type}")
```

### Model Class Structure
All models must inherit from `BaseNeuralFXModel` and implement:
- `forward()` - batch training forward pass
- `reset_state()` - reset internal state
- `process_sample()` - single-sample validation
- `receptive_field` property
- `from_config()` class method factory
- Export methods: `export_onnx()`, `export_torchscript()`, `export_rtneural()`

```python
class BaseNeuralFXModel(nn.Module, ABC):
    model_type: str = "base"

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...
```

### Audio Processing Conventions
- Default sample rate: 48000 Hz
- Audio tensors: `(batch, channels, samples)` or `(batch, samples)`
- Use `torch.float32` for audio data
- Pre-emphasis filter coefficient: 0.85

### File Organization
```
neural-fx/
├── neural_fx/          # Main package
│   ├── config.py       # Configuration dataclasses
│   ├── data/           # Data loading & transforms
│   ├── models/         # Model implementations
│   ├── training/       # Lightning modules
│   ├── losses/         # Loss functions
│   └── inference/      # Inference utilities
├── configs/models/     # YAML configs per model type
├── scripts/            # Entry-point scripts
└── notebooks/          # Jupyter notebooks
```

### Key Patterns
1. **TBPTT**: Use truncated backpropagation through time with `burn_in` and `truncate` parameters
2. **Skip Connections**: Support optional skip connections in RNN blocks
3. **State Management**: Models with state implement `detach_state()` for gradient management
4. **Config Loaders**: Use factory pattern for model-specific params (see `config.py`)

### Documentation
- Use docstrings for all public classes and functions
- Include type hints in docstrings for complex signatures
- Add section comments for code organization:
```python
# =============================================================================
# MODEL-SPECIFIC PARAMS
# =============================================================================
```

### Common Pitfalls
- Remember to call `super().__init__()` in model `__init__`
- State in RNN models must be detached during TBPTT
- Audio files are loaded at native sample rate; resample explicitly
- Use `Path` objects for file operations, not raw strings

### Dependencies
Core stack: torch, torchaudio, numpy, scipy, tqdm, matplotlib, lightning, pyyaml
