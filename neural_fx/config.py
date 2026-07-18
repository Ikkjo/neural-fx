import warnings
from dataclasses import dataclass, field
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
    """Parameters for a causal WaveNet model.

    ``layers`` is the number of dilation layers in each stack. Every stack
    repeats the dilation sequence ``1, 2, 4, ..., 2**(layers - 1)``.
    """

    layers: int
    stacks: int = 3
    kernel_size: int = 3
    dilation_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32

    def __post_init__(self) -> None:
        values = {
            "layers": self.layers,
            "stacks": self.stacks,
            "kernel_size": self.kernel_size,
            "dilation_channels": self.dilation_channels,
            "residual_channels": self.residual_channels,
            "skip_channels": self.skip_channels,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid:
            raise ValueError(
                "WaveNet parameters must be positive: " + ", ".join(invalid)
            )


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

    d_model: int = 32
    d_state: int = 16
    num_layers: int = 4
    dropout: float = 0.0
    prenorm: bool = True
    d_conv: int = 4
    expand: int = 2

    def __post_init__(self) -> None:
        positive_values = {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "num_layers": self.num_layers,
            "d_conv": self.d_conv,
            "expand": self.expand,
        }
        invalid = [
            name for name, value in positive_values.items() if value <= 0
        ]
        if invalid:
            raise ValueError(
                "SSM parameters must be positive: " + ", ".join(invalid)
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("SSM dropout must be in the range [0, 1)")


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
class AugmentationConfig:
    """Configuration for audio augmentation during training."""

    enabled: bool = False
    random_gain: bool = False
    gain_min_db: float = -12.0
    gain_max_db: float = 0.0
    gain_p: float = 0.5
    add_noise: bool = False
    noise_min_snr_db: float = 20.0
    noise_max_snr_db: float = 40.0
    noise_p: float = 0.3


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    segment_length: int = 8192
    random_segments: bool = False  # Use random segment sampling
    tbptt: TBPTTConfig | None = None
    seed: int = 42
    augmentation: AugmentationConfig | None = None
    num_workers: int = 4


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
class STFTLossConfig:
    """Configuration for multi-resolution STFT loss."""

    enabled: bool = False
    fft_sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    hop_sizes: list[int] | None = None
    win_sizes: list[int] | None = None
    sc_weight: float = 1.0  # Spectral convergence weight
    mag_weight: float = 1.0  # Log magnitude weight


@dataclass
class LossWeights:
    esr: float = 0.0
    mse: float = 1.0
    stft: float = 0.0  # Weight for multi-resolution STFT loss


@dataclass
class LossConfig:
    type: str = "mse"
    weights: LossWeights | None = None
    pre_emphasis: PreEmphasisConfig | None = None
    mask_first: int = 4096
    stft: STFTLossConfig | None = None


@dataclass
class DataPaths:
    input: str
    target: str


@dataclass
class DataConfig:
    train: DataPaths
    val: DataPaths | None = None  # Optional validation data paths


@dataclass
class LatencyConfig:
    """Configuration for latency calibration between input and output audio."""

    enabled: bool = True
    method: str = "xcorr"  # xcorr, manual
    manual_delay: int | None = None
    max_delay: int = 10000
    calibration_duration_seconds: float = 5.0


@dataclass
class ValidationConfig:
    """Configuration for data validation before training."""

    enabled: bool = True
    check_clipping: bool = True
    check_dc_offset: bool = True
    check_replicability: bool | None = None
    ignore_warnings: bool = False


LoggerBackend = Literal["csv", "tensorboard"]


@dataclass
class LoggingConfig:
    """Configuration for training metrics and experiment logs."""

    backends: list[LoggerBackend] = field(
        default_factory=lambda: ["csv", "tensorboard"]
    )
    save_dir: str = "lightning_logs"
    log_every_n_steps: int = 50

    def __post_init__(self) -> None:
        supported = {"csv", "tensorboard"}
        unknown = set(self.backends) - supported
        if unknown:
            unknown_names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown logging backend(s): {unknown_names}")
        if not self.backends:
            raise ValueError("At least one logging backend must be configured")
        if len(self.backends) != len(set(self.backends)):
            raise ValueError("Logging backends must not contain duplicates")
        if self.log_every_n_steps <= 0:
            raise ValueError("logging.log_every_n_steps must be greater than zero")


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
    latency: LatencyConfig | None = None
    validation: ValidationConfig | None = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def sample_rate(self) -> int:
        """Authoritative sample rate for models, data, and audio processing."""
        return self.model.sample_rate


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


def _load_augmentation_config(aug_cfg: dict | None) -> AugmentationConfig | None:
    """Load augmentation configuration from dict."""
    if aug_cfg is None:
        return None
    return AugmentationConfig(**aug_cfg)


def _load_stft_loss_config(stft_cfg: dict | None) -> STFTLossConfig | None:
    """Load STFT loss configuration from dict."""
    if stft_cfg is None:
        return None

    # Create config with defaults
    config = STFTLossConfig(**stft_cfg)

    # Apply default hop_sizes and win_sizes if not provided
    if config.hop_sizes is None:
        config.hop_sizes = [fft // 4 for fft in config.fft_sizes]
    if config.win_sizes is None:
        config.win_sizes = config.fft_sizes

    return config


def _load_latency_config(lat_cfg: dict | None) -> LatencyConfig | None:
    """Load latency configuration from dict."""
    if lat_cfg is None:
        return LatencyConfig()  # Return default config
    return LatencyConfig(**lat_cfg)


def _load_validation_config(val_cfg: dict | None) -> ValidationConfig | None:
    """Load validation configuration from dict."""
    if val_cfg is None:
        return ValidationConfig()  # Return default config
    return ValidationConfig(**val_cfg)


def _load_logging_config(logging_cfg: dict | None) -> LoggingConfig:
    """Load logging configuration while preserving defaults for older YAMLs."""
    if logging_cfg is None:
        return LoggingConfig()
    return LoggingConfig(**logging_cfg)


def load_config(path: Path | str) -> NeuralFXConfig:
    """Load and parse a YAML config file."""
    with open(path) as f:
        d = yaml.safe_load(f)

    version = d["version"]
    name = d["name"]
    model_cfg = d["model"]
    model_type = model_cfg["type"]
    model_params = model_cfg.get("params", {})
    optimizer_cfg = d.get("optimizer", {"type": "adam", "lr": 0.01})
    lr_scheduler_cfg = d.get("lr_scheduler", {"type": "exponential", "gamma": 0.995})
    data_cfg = d["data"]
    train_data_cfg = data_cfg["train"]
    val_data_cfg = data_cfg.get("val")

    model_sample_rate = model_cfg.get("sample_rate")
    legacy_data_sample_rate = data_cfg.get("sample_rate")
    if (
        model_sample_rate is not None
        and legacy_data_sample_rate is not None
        and model_sample_rate != legacy_data_sample_rate
    ):
        raise ValueError(
            "Conflicting sample rates: model.sample_rate "
            f"({model_sample_rate}) does not match deprecated data.sample_rate "
            f"({legacy_data_sample_rate}). Remove data.sample_rate and keep the "
            "authoritative value under model.sample_rate."
        )
    if legacy_data_sample_rate is not None:
        warnings.warn(
            "data.sample_rate is deprecated; move the value to model.sample_rate.",
            DeprecationWarning,
            stacklevel=2,
        )
    if model_sample_rate is None:
        model_sample_rate = (
            legacy_data_sample_rate
            if legacy_data_sample_rate is not None
            else 48000
        )

    # Load augmentation config
    training_cfg = d["training"]
    aug_cfg = training_cfg.get("augmentation")
    augmentation = _load_augmentation_config(aug_cfg)

    # Load STFT loss config
    loss_cfg = d["loss"]
    stft_cfg = loss_cfg.get("stft")
    stft_loss_config = _load_stft_loss_config(stft_cfg)

    # Load latency and validation configs
    latency_cfg = _load_latency_config(d.get("latency"))
    validation_cfg = _load_validation_config(d.get("validation"))
    logging_cfg = _load_logging_config(d.get("logging"))

    return NeuralFXConfig(
        version=version,
        name=name,
        model=ModelConfig(
            type=model_type,
            params=_load_model_params(model_type, model_params),
            input_size=model_cfg.get("input_size", 1),
            output_size=model_cfg.get("output_size", 1),
            sample_rate=model_sample_rate,
        ),
        training=TrainingConfig(
            batch_size=training_cfg.get("batch_size", 32),
            epochs=training_cfg.get("epochs", 100),
            segment_length=training_cfg.get("segment_length", 8192),
            random_segments=training_cfg.get("random_segments", False),
            tbptt=TBPTTConfig(**training_cfg["tbptt"])
            if training_cfg.get("tbptt")
            else None,
            seed=training_cfg.get("seed", 42),
            augmentation=augmentation,
            num_workers=training_cfg.get("num_workers", 4),
        ),
        optimizer=OptimizerConfig(**optimizer_cfg),
        lr_scheduler=LRSchedulerConfig(**lr_scheduler_cfg),
        loss=LossConfig(
            type=loss_cfg["type"],
            weights=LossWeights(**loss_cfg["weights"])
            if "weights" in loss_cfg
            else None,
            pre_emphasis=PreEmphasisConfig(**loss_cfg["pre_emphasis"])
            if "pre_emphasis" in loss_cfg
            else None,
            mask_first=loss_cfg.get("mask_first", 4096),
            stft=stft_loss_config,
        ),
        data=DataConfig(
            train=DataPaths(**train_data_cfg),
            val=DataPaths(**val_data_cfg) if val_data_cfg else None,
        ),
        latency=latency_cfg,
        validation=validation_cfg,
        logging=logging_cfg,
    )
