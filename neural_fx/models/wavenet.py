from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig, WaveNetParams, _load_model_params
from .base import BaseNeuralFXModel, UnsupportedExportError


class CausalConv1d(nn.Module):
    """One-dimensional convolution that never reads future samples."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.history_size = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self._cache: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.history_size, 0)))

    def reset_state(self) -> None:
        self._cache = None

    def process_sample(self, x: Tensor) -> Tensor:
        if self.history_size == 0:
            return self.conv(x)
        if (
            self._cache is None
            or self._cache.shape[0] != x.shape[0]
            or self._cache.device != x.device
            or self._cache.dtype != x.dtype
        ):
            self._cache = x.new_zeros(x.shape[0], x.shape[1], self.history_size)
        window = torch.cat([self._cache, x], dim=-1)
        self._cache = window[..., 1:].detach()
        return self.conv(window)


class DilatedResidualBlock(nn.Module):
    """Gated WaveNet residual layer with a skip output."""

    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.filter_conv = CausalConv1d(
            residual_channels, dilation_channels, kernel_size, dilation
        )
        self.gate_conv = CausalConv1d(
            residual_channels, dilation_channels, kernel_size, dilation
        )
        self.residual_projection = nn.Conv1d(
            dilation_channels, residual_channels, kernel_size=1
        )
        self.skip_projection = nn.Conv1d(
            dilation_channels, skip_channels, kernel_size=1
        )

    def _project(self, x: Tensor, filter_out: Tensor, gate_out: Tensor) -> tuple[Tensor, Tensor]:
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        residual = x + self.residual_projection(gated)
        return residual, self.skip_projection(gated)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._project(x, self.filter_conv(x), self.gate_conv(x))

    def process_sample(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._project(
            x,
            self.filter_conv.process_sample(x),
            self.gate_conv.process_sample(x),
        )

    def reset_state(self) -> None:
        self.filter_conv.reset_state()
        self.gate_conv.reset_state()


class WaveNetModel(BaseNeuralFXModel):
    """Causal WaveNet for training and real-time neural audio processing."""

    model_type = "wavenet"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.sample_rate)
        if not isinstance(config.params, WaveNetParams):
            raise TypeError(f"Expected WaveNetParams, got {type(config.params)}")
        self.config = config
        self.params = config.params
        self.input_projection = nn.Conv1d(
            config.input_size, self.params.residual_channels, kernel_size=1
        )
        dilations = [
            2**layer
            for _ in range(self.params.stacks)
            for layer in range(self.params.layers)
        ]
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    residual_channels=self.params.residual_channels,
                    dilation_channels=self.params.dilation_channels,
                    skip_channels=self.params.skip_channels,
                    kernel_size=self.params.kernel_size,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.output_projection = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.params.skip_channels, self.params.skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(self.params.skip_channels, config.output_size, 1),
        )
        self._input_history: Tensor | None = None

    def _forward_sequence(self, x: Tensor) -> Tensor:
        residual = self.input_projection(x)
        skip_sum: Tensor | None = None
        for block in self.blocks:
            residual, skip = block(residual)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        if skip_sum is None:
            raise RuntimeError("WaveNet requires at least one residual block")
        return self.output_projection(skip_sum)

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

        time = x.shape[-1]
        history_size = self.receptive_field - 1
        if self._input_history is not None:
            sequence = torch.cat([self._input_history, x], dim=-1)
        else:
            sequence = x
        output = self._forward_sequence(sequence)[..., -time:]
        if history_size > 0:
            self._input_history = sequence[..., -history_size:].detach()
        return output

    def process_sample(self, x: Tensor, reset: bool = False) -> Tensor:
        if reset:
            self.reset_state()
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.ndim == 2:
            x = x.unsqueeze(-1)
        residual = self.input_projection(x)
        skip_sum: Tensor | None = None
        with torch.no_grad():
            for block in self.blocks:
                residual, skip = block.process_sample(residual)
                skip_sum = skip if skip_sum is None else skip_sum + skip
            if skip_sum is None:
                raise RuntimeError("WaveNet requires at least one residual block")
            return self.output_projection(skip_sum).squeeze(-1)

    def reset_state(self) -> None:
        self._input_history = None
        for block in self.blocks:
            block.reset_state()

    def detach_state(self) -> None:
        if self._input_history is not None:
            self._input_history = self._input_history.detach()

    @property
    def receptive_field(self) -> int:
        dilation_sum = self.params.stacks * (2**self.params.layers - 1)
        return 1 + (self.params.kernel_size - 1) * dilation_sum

    @property
    def supported_export_formats(self) -> tuple[str, ...]:
        return ("onnx", "torchscript")

    @classmethod
    def from_config(cls, config: dict[str, Any] | ModelConfig) -> "WaveNetModel":
        if isinstance(config, dict):
            model_type = config.get("type", "wavenet")
            config = ModelConfig(
                type=model_type,
                params=_load_model_params(model_type, config.get("params", {})),
                input_size=config.get("input_size", 1),
                output_size=config.get("output_size", 1),
                sample_rate=config.get("sample_rate", 48_000),
            )
        if config.type != "wavenet":
            raise ValueError(f"Expected wavenet config, got {config.type}")
        return cls(config)

    def _export_wrapper(self) -> nn.Module:
        class ExportWrapper(nn.Module):
            def __init__(self, model: "WaveNetModel") -> None:
                super().__init__()
                self.model = model

            def forward(self, x: Tensor) -> Tensor:
                return self.model._forward_sequence(x)

        return ExportWrapper(self)

    def export_onnx(self, path: str | Path, opset_version: int = 17) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = self._export_wrapper().eval()
        dummy = torch.randn(1, self.config.input_size, 512)
        torch.onnx.export(
            wrapper,
            dummy,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "time"},
                "output": {0: "batch_size", 2: "time"},
            },
            opset_version=opset_version,
            dynamo=False,
        )

    def export_torchscript(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = self._export_wrapper().eval()
        traced = torch.jit.trace(
            wrapper, torch.randn(1, self.config.input_size, 512)
        )
        traced.save(str(path))

    def export_rtneural(self, path: str | Path) -> None:
        raise UnsupportedExportError(
            "WaveNet cannot be represented by the current RTNeural JSON exporter"
        )
