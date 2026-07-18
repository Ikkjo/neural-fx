"""Optional CUDA-backed Mamba model for neural audio effects."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from torch import Tensor

from ..config import ModelConfig, SSMParams, _load_model_params
from .base import BaseNeuralFXModel
from .errors import OptionalDependencyError, UnsupportedExportFormatError


MAMBA_INSTALL_HINT = (
    "pip install -e .[mamba] on a Linux machine with a compatible NVIDIA GPU"
)
MambaState: TypeAlias = tuple[Tensor, Tensor]


def _optional_dependency_error(dependency: str) -> OptionalDependencyError:
    return OptionalDependencyError("Mamba", dependency, MAMBA_INSTALL_HINT)


def _import_mamba_class() -> type[nn.Module]:
    from mamba_ssm import Mamba

    return Mamba


def _load_mamba_class() -> type[nn.Module]:
    """Return the upstream Mamba class after checking its supported runtime."""
    if not sys.platform.startswith("linux"):
        raise _optional_dependency_error("a Linux/CUDA runtime")
    if not torch.cuda.is_available():
        raise _optional_dependency_error("CUDA-enabled PyTorch")

    try:
        mamba_class = _import_mamba_class()
    except (ImportError, OSError, RuntimeError) as error:
        raise _optional_dependency_error("mamba-ssm") from error
    return mamba_class


class _MambaResidualBlock(nn.Module):
    """Pre/post-norm residual wrapper around one upstream Mamba mixer."""

    def __init__(
        self,
        backend_class: type[nn.Module],
        params: SSMParams,
        layer_index: int,
    ):
        super().__init__()
        self.prenorm = params.prenorm
        self.norm = nn.LayerNorm(params.d_model)
        self.dropout = nn.Dropout(params.dropout)
        self.mixer = backend_class(
            d_model=params.d_model,
            d_state=params.d_state,
            d_conv=params.d_conv,
            expand=params.expand,
            layer_idx=layer_index,
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        if self.prenorm:
            x = self.norm(x)
        x = self.dropout(self.mixer(x))
        x = x + residual
        if not self.prenorm:
            x = self.norm(x)
        return x

    def allocate_state(self, batch_size: int, dtype: torch.dtype) -> MambaState:
        state = self.mixer.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=1,
            dtype=dtype,
        )
        if not isinstance(state, tuple) or len(state) != 2:
            raise RuntimeError(
                "mamba-ssm returned an unsupported inference-cache structure"
            )
        conv_state, ssm_state = state
        if not isinstance(conv_state, Tensor) or not isinstance(ssm_state, Tensor):
            raise RuntimeError("mamba-ssm inference caches must be tensors")
        return conv_state, ssm_state

    def step(self, x: Tensor, state: MambaState) -> tuple[Tensor, MambaState]:
        residual = x
        if self.prenorm:
            x = self.norm(x)
        output = self.mixer.step(x, state[0], state[1])
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError("mamba-ssm returned an unsupported step result")
        x, conv_state, ssm_state = output
        x = self.dropout(x) + residual
        if not self.prenorm:
            x = self.norm(x)
        return x, (conv_state, ssm_state)


class MambaModel(BaseNeuralFXModel):
    """Mamba stack with stateful, cached real-time inference.

    Full-sequence training uses the upstream parallel ``forward`` path. In
    evaluation mode, ``forward`` and ``process_sample`` preserve one upstream
    convolution/SSM cache per layer and use ``Mamba.step``. Inference state is
    deliberately separate from training graphs and is not a TBPTT mechanism.
    """

    model_type = "mamba"

    def __init__(self, config: ModelConfig):
        if config.type != "mamba":
            raise ValueError(f"MambaModel requires type 'mamba', got {config.type!r}")
        if not isinstance(config.params, SSMParams):
            raise ValueError(f"Expected SSMParams, got {type(config.params)}")

        backend_class = _load_mamba_class()
        super().__init__(config.sample_rate)
        self.config = config
        self.params = config.params
        self.input_projection = nn.Linear(config.input_size, self.params.d_model)
        try:
            self.blocks = nn.ModuleList(
                _MambaResidualBlock(backend_class, self.params, layer_index)
                for layer_index in range(self.params.num_layers)
            )
        except (OSError, RuntimeError, TypeError) as error:
            dependency = "a compatible mamba-ssm 2.x build"
            raise _optional_dependency_error(dependency) from error
        self.final_norm = nn.LayerNorm(self.params.d_model)
        self.output_projection = nn.Linear(
            self.params.d_model, config.output_size
        )
        self._stream_states: list[MambaState] | None = None
        self._stream_batch_size: int | None = None

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "MambaModel expects [batch, channels, time], "
                f"received shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} input channels, got {x.shape[1]}"
            )

    def _project_input(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        return self.input_projection(x.transpose(1, 2))

    def _project_output(self, x: Tensor) -> Tensor:
        x = self.output_projection(self.final_norm(x))
        return x.transpose(1, 2)

    def _forward_training(self, x: Tensor) -> Tensor:
        x = self._project_input(x)
        for block in self.blocks:
            x = block(x)
        return self._project_output(x)

    def _ensure_stream_state(self, x: Tensor) -> None:
        batch_size = x.shape[0]
        state_matches_input = self._stream_states is not None and all(
            conv_state.device == x.device
            and ssm_state.device == x.device
            and conv_state.dtype == x.dtype
            and ssm_state.dtype == x.dtype
            for conv_state, ssm_state in self._stream_states
        )
        if (
            state_matches_input
            and self._stream_batch_size == batch_size
        ):
            return
        self._stream_states = [
            block.allocate_state(batch_size, x.dtype) for block in self.blocks
        ]
        self._stream_batch_size = batch_size

    def _stream_step(self, x: Tensor) -> Tensor:
        self._ensure_stream_state(x)
        if self._stream_states is None:
            raise RuntimeError("Mamba inference state was not initialized")

        next_states: list[MambaState] = []
        for block, state in zip(self.blocks, self._stream_states):
            x, next_state = block.step(x, state)
            next_states.append(next_state)
        self._stream_states = next_states
        return x

    def _forward_streaming(self, x: Tensor) -> Tensor:
        x = self._project_input(x)
        outputs = [
            self._stream_step(x[:, index : index + 1])
            for index in range(x.shape[1])
        ]
        if not outputs:
            return self._project_output(x)
        return self._project_output(torch.cat(outputs, dim=1))

    def forward(
        self,
        x: Tensor,
        reset_state: bool = False,
        detach_state: bool = False,
    ) -> Tensor:
        if reset_state:
            self.reset_state()
        if detach_state:
            self.detach_state()
        if self.training:
            return self._forward_training(x)
        return self._forward_streaming(x)

    def reset_state(self) -> None:
        self._stream_states = None
        self._stream_batch_size = None

    def detach_state(self) -> None:
        if self._stream_states is None:
            return
        self._stream_states = [
            (conv_state.detach(), ssm_state.detach())
            for conv_state, ssm_state in self._stream_states
        ]

    def process_sample(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.ndim == 2:
            x = x.unsqueeze(-1)
        elif x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(
                "process_sample expects [channels], [batch, channels], or "
                "[batch, channels, 1]"
            )
        with torch.no_grad():
            output = self._project_input(x)
            output = self._stream_step(output)
            output = self._project_output(output)
        return output.squeeze()

    @property
    def receptive_field(self) -> int:
        """Mamba retains unbounded causal history; ``-1`` is the sentinel."""
        return -1

    @classmethod
    def from_config(cls, config: dict[str, Any] | ModelConfig) -> MambaModel:
        if isinstance(config, dict):
            model_type = config.get("type", "mamba")
            params = _load_model_params(model_type, config.get("params", {}))
            config = ModelConfig(
                type=model_type,
                params=params,
                input_size=config.get("input_size", 1),
                output_size=config.get("output_size", 1),
                sample_rate=config.get("sample_rate", 48_000),
            )
        return cls(config)

    def export_onnx(self, path: Path, opset_version: int = 17) -> None:
        del path, opset_version
        raise UnsupportedExportFormatError(self.model_type, "onnx")

    def export_torchscript(self, path: Path) -> None:
        del path
        raise UnsupportedExportFormatError(self.model_type, "torchscript")

    def export_rtneural(self, path: Path) -> None:
        del path
        raise UnsupportedExportFormatError(self.model_type, "rtneural")
