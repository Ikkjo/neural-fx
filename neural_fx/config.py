from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union
import yaml

# =============================================================================
# MODEL-SPECIFIC PARAMS
# =============================================================================


@dataclass
class Conv1dConfig:
    filters: int
    kernel_size: int = 3
    stride: int = 1


@dataclass
class LSTMParams:
    """Parameters for LSTM and GRU models."""
    hidden_size: int
    num_layers: int = 2
    conv1d: Conv1dConfig | None = None
    skip_connection: bool = False
    dropout: float = 0.0
    conditioning_size: int = 0



@dataclass
class WaveNetParams:
    """Parameters for WaveNet models."""
    layers: int
    stacks: int = 3
    kernel_size: int = 3
    dilation_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32


@dataclass
class SSMParams:
    """
    Shared parameters for state-space models (e.g. Mamba, S4).

    Notes
    -----
    - ``d_state`` is the latent state dimension and is typically used by both
      Mamba- and S4-style implementations.
    - ``d_conv`` controls the size of any local convolutional kernel used in
      Mamba-style SSMs; S4 implementations in this codebase may ignore it.
    - ``expand`` is an expansion factor used by some Mamba variants to
      increase the internal channel width; S4 implementations may ignore it.
    Implementations for specific architectures are expected to read only the
    subset of fields that they support; unused fields are safe to ignore.
    """
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


ModelParamsType = Union[LSTMParams, WaveNetParams, SSMParams]

# =============================================================================
# SHARED CONFIG SECTIONS
# =============================================================================


@dataclass
class ModelConfig:
    type: Literal["lstm", "gru", "wavenet", "mamba", "s4"]
    params: ModelParamsType
    input_size: int = 1
    output_size: int = 1
    sample_rate: int = 48000


@dataclass
class TBPTTConfig:
    enabled: bool = True
    burn_in: int = 4096


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    segment_length: int = 8192
    tbptt: TBPTTConfig | None = None
    seed: int = 42


@dataclass
class OptimizerConfig:
    type: str = "adam"
    lr: float = 0.01


@dataclass
class LRSchedulerConfig:
    type: str = "exponential"
    gamma: float = 0.995


@dataclass
class PreEmphasisConfig:
    enabled: bool = True
    coef: float = 0.85


@dataclass
class LossWeights:
    esr: float = 0.0
    mse: float = 1.0


@dataclass
class LossConfig:
    type: str = "mse"
    weights: LossWeights | None = None
    pre_emphasis: PreEmphasisConfig | None = None
    mask_first: int = 4096


@dataclass
class DataPaths:
    input: str
    target: str


@dataclass
class DataConfig:
    train: DataPaths
    sample_rate: int = 48000

# =============================================================================
# ROOT CONFIG
# =============================================================================


@dataclass
class NeuralFXConfig:
    version: str
    name: str
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    loss: LossConfig
    data: DataConfig

# =============================================================================
# LOADER
# =============================================================================


# Factory mapping for model-specific params
_PARAM_CLASSES = {
    "lstm": LSTMParams,
    "gru": LSTMParams,
    "wavenet": WaveNetParams,
    "mamba": SSMParams,
    "s4": SSMParams,
}


def _load_model_params(model_type: str, params: dict) -> ModelParamsType:
    """Load model-specific params using the appropriate dataclass."""
    cls = _PARAM_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model type: {model_type}")

    # Handle nested conv1d for LSTM/GRU
    if cls == LSTMParams and "conv1d" in params:
        params = params.copy()
        params["conv1d"] = Conv1dConfig(**params["conv1d"])

    return cls(**params)


def load_config(path: Path | str) -> NeuralFXConfig:
    """Load and parse a YAML config file."""
    with open(path) as f:
        d = yaml.safe_load(f)

    model_type = d["model"]["type"]

    return NeuralFXConfig(
        version=version,
        name=name,
        model=ModelConfig(
            type=model_type,
            params=_load_model_params(model_type, model_params),
            input_size=model_cfg.get("input_size", 1),
            output_size=model_cfg.get("output_size", 1),
            sample_rate=model_cfg.get("sample_rate", 48000),
        ),
        training=TrainingConfig(
            batch_size=d["training"].get("batch_size", 32),
            epochs=d["training"].get("epochs", 100),
            segment_length=d["training"].get("segment_length", 8192),
            tbptt=TBPTTConfig(**d["training"]["tbptt"]
                              ) if "tbptt" in d["training"] else None,
            seed=d["training"].get("seed", 42),
        ),
        optimizer=OptimizerConfig(**optimizer_cfg),
        lr_scheduler=LRSchedulerConfig(**lr_scheduler_cfg),
        loss=LossConfig(
            type=d["loss"]["type"],
            weights=LossWeights(**d["loss"]["weights"]
                                ) if "weights" in d["loss"] else None,
            pre_emphasis=PreEmphasisConfig(
                **d["loss"]["pre_emphasis"]) if "pre_emphasis" in d["loss"] else None,
            mask_first=d["loss"].get("mask_first", 4096),
        ),
        data=DataConfig(
            train=DataPaths(**train_data_cfg),
            sample_rate=data_cfg.get("sample_rate", 48000),
        ),
    )
