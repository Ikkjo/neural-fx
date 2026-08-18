import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..config import ModelConfig, SSMParams, _load_model_params
from .base import BaseNeuralFXModel, UnsupportedExportError


class DiagonalStateSpace(nn.Module):
    """Stable real diagonal state-space layer with FFT and recurrent modes."""

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.log_dt = nn.Parameter(torch.linspace(math.log(0.001), math.log(0.1), d_model))
        self.log_a = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(d_model, -1)
            .clone()
        )
        scale = d_state**-0.5
        self.b = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.c = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.d = nn.Parameter(torch.ones(d_model))
        self._state: Tensor | None = None

    def _discretize(self) -> tuple[Tensor, Tensor]:
        a = -torch.exp(self.log_a)
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        a_bar = torch.exp(dt * a)
        b_bar = ((a_bar - 1.0) / a) * self.b
        return a_bar, b_bar

    @staticmethod
    def _causal_convolution(x: Tensor, kernel: Tensor) -> Tensor:
        length = x.shape[-1]
        fft_size = 2 * length
        x_fft = torch.fft.rfft(x, n=fft_size)
        kernel_fft = torch.fft.rfft(kernel, n=fft_size)
        return torch.fft.irfft(x_fft * kernel_fft.unsqueeze(0), n=fft_size)[
            ..., :length
        ]

    def _run(self, x: Tensor, initial_state: Tensor | None) -> tuple[Tensor, Tensor]:
        a_bar, b_bar = self._discretize()
        length = x.shape[-1]
        steps = torch.arange(length, device=x.device, dtype=x.dtype)
        powers = a_bar.unsqueeze(-1) ** steps
        kernel = torch.sum((self.c * b_bar).unsqueeze(-1) * powers, dim=1)
        output = self._causal_convolution(x, kernel) + self.d.view(1, -1, 1) * x

        if initial_state is not None:
            state_powers = a_bar.unsqueeze(-1) ** (steps + 1)
            output = output + torch.einsum(
                "bhs,hsl,hs->bhl", initial_state, state_powers, self.c
            )

        reverse_steps = torch.arange(
            length - 1, -1, -1, device=x.device, dtype=x.dtype
        )
        input_weights = b_bar.unsqueeze(-1) * (
            a_bar.unsqueeze(-1) ** reverse_steps
        )
        final_state = torch.einsum("bhl,hsl->bhs", x, input_weights)
        if initial_state is not None:
            final_state = final_state + initial_state * (a_bar**length)
        return output, final_state

    def forward(self, x: Tensor) -> Tensor:
        output, self._state = self._run(x, self._state)
        return output

    def forward_stateless(self, x: Tensor) -> Tensor:
        output, _ = self._run(x, None)
        return output

    def process_sample(self, x: Tensor) -> Tensor:
        a_bar, b_bar = self._discretize()
        if (
            self._state is None
            or self._state.shape[0] != x.shape[0]
            or self._state.device != x.device
            or self._state.dtype != x.dtype
        ):
            self._state = x.new_zeros(x.shape[0], self.d_model, self.d_state)
        sample = x.squeeze(-1)
        self._state = (
            a_bar.unsqueeze(0) * self._state
            + b_bar.unsqueeze(0) * sample.unsqueeze(-1)
        )
        output = torch.sum(self.c.unsqueeze(0) * self._state, dim=-1)
        return (output + self.d.unsqueeze(0) * sample).unsqueeze(-1)

    def reset_state(self) -> None:
        self._state = None

    def detach_state(self) -> None:
        if self._state is not None:
            self._state = self._state.detach()


class S4DBlock(nn.Module):
    """Pre-normalized residual block around a diagonal state-space layer."""

    def __init__(self, d_model: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagonalStateSpace(d_model, d_state)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Conv1d(d_model, d_model, 1)

    def _normalize(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

    def _finish(self, residual: Tensor, x: Tensor) -> Tensor:
        return residual + self.output_projection(self.dropout(self.activation(x)))

    def forward(self, x: Tensor) -> Tensor:
        return self._finish(x, self.ssm(self._normalize(x)))

    def forward_stateless(self, x: Tensor) -> Tensor:
        return self._finish(x, self.ssm.forward_stateless(self._normalize(x)))

    def process_sample(self, x: Tensor) -> Tensor:
        return self._finish(x, self.ssm.process_sample(self._normalize(x)))

    def reset_state(self) -> None:
        self.ssm.reset_state()

    def detach_state(self) -> None:
        self.ssm.detach_state()


class S4DModel(BaseNeuralFXModel):
    """Portable diagonal S4-style model for long-memory audio effects."""

    model_type = "s4"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.sample_rate)
        if not isinstance(config.params, SSMParams):
            raise TypeError(f"Expected SSMParams, got {type(config.params)}")
        self.config = config
        self.params = config.params
        self.input_projection = nn.Conv1d(config.input_size, self.params.d_model, 1)
        self.blocks = nn.ModuleList(
            [
                S4DBlock(
                    d_model=self.params.d_model,
                    d_state=self.params.d_state,
                    dropout=self.params.dropout,
                )
                for _ in range(self.params.num_layers)
            ]
        )
        self.output_projection = nn.Conv1d(
            self.params.d_model, config.output_size, 1
        )

    def _forward_sequence(self, x: Tensor, stateless: bool = False) -> Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = block.forward_stateless(x) if stateless else block(x)
        return self.output_projection(x)

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
        return self._forward_sequence(x)

    def process_sample(self, x: Tensor, reset: bool = False) -> Tensor:
        if reset:
            self.reset_state()
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.ndim == 2:
            x = x.unsqueeze(-1)
        with torch.no_grad():
            x = self.input_projection(x)
            for block in self.blocks:
                x = block.process_sample(x)
            return self.output_projection(x).squeeze(-1)

    def reset_state(self) -> None:
        for block in self.blocks:
            block.reset_state()

    def detach_state(self) -> None:
        for block in self.blocks:
            block.detach_state()

    @property
    def receptive_field(self) -> float:
        return math.inf

    @property
    def supported_export_formats(self) -> tuple[str, ...]:
        return ("torchscript",)

    @classmethod
    def from_config(cls, config: dict[str, Any] | ModelConfig) -> "S4DModel":
        if isinstance(config, dict):
            model_type = config.get("type", "s4")
            config = ModelConfig(
                type=model_type,
                params=_load_model_params(model_type, config.get("params", {})),
                input_size=config.get("input_size", 1),
                output_size=config.get("output_size", 1),
                sample_rate=config.get("sample_rate", 48_000),
            )
        if config.type != "s4":
            raise ValueError(f"Expected s4 config, got {config.type}")
        return cls(config)

    def export_onnx(self, path: str | Path, opset_version: int = 17) -> None:
        raise UnsupportedExportError(
            "S4D FFT operators are not supported by the current ONNX exporter"
        )

    def export_torchscript(self, path: str | Path) -> None:
        class ExportWrapper(nn.Module):
            def __init__(self, model: "S4DModel") -> None:
                super().__init__()
                self.model = model

            def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
                x = self.model.input_projection(x)
                next_states = []
                for index, block in enumerate(self.model.blocks):
                    residual = x
                    normalized = block._normalize(x)
                    a_bar, b_bar = block.ssm._discretize()
                    sample = normalized.squeeze(-1)
                    next_state = (
                        a_bar.unsqueeze(0) * state[index]
                        + b_bar.unsqueeze(0) * sample.unsqueeze(-1)
                    )
                    ssm_output = torch.sum(
                        block.ssm.c.unsqueeze(0) * next_state, dim=-1
                    )
                    ssm_output = (
                        ssm_output + block.ssm.d.unsqueeze(0) * sample
                    ).unsqueeze(-1)
                    x = block._finish(residual, ssm_output)
                    next_states.append(next_state)
                return self.model.output_projection(x), torch.stack(next_states)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = ExportWrapper(self).eval()
        dummy_input = torch.randn(1, self.config.input_size, 1)
        dummy_state = torch.zeros(
            self.params.num_layers, 1, self.params.d_model, self.params.d_state
        )
        traced = torch.jit.trace(
            wrapper, (dummy_input, dummy_state)
        )
        traced.save(str(path))

    def export_rtneural(self, path: str | Path) -> None:
        raise UnsupportedExportError(
            "S4D cannot be represented by the current RTNeural JSON exporter"
        )
