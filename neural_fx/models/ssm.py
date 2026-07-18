"""Portable diagonal state-space (S4D) audio model.

The diagonal SSM parameterization and S4D-Lin initialization are adapted from
the official S4 repository's standalone S4D example:
https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py

That implementation is distributed under the Apache License 2.0. This module
is an independent, compact PyTorch adaptation for causal audio that adds
cached recurrent inference and the Neural-FX model contract.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import ModelConfig, SSMParams, _load_model_params
from .base import BaseNeuralFXModel
from .errors import UnsupportedExportFormatError


class S4DLayer(nn.Module):
    """Causal diagonal state-space layer with sequence and recurrent paths.

    The continuous system is discretized exactly with a zero-order hold:

    ``state[t] = exp(dt*A) * state[t-1] + ((exp(dt*A)-1)/A) * u[t]``

    The sequence path constructs the corresponding impulse response and uses
    FFT convolution. ``step`` evaluates the same recurrence directly.
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        if d_model <= 0 or d_state <= 0:
            raise ValueError("d_model and d_state must be positive")

        self.d_model = d_model
        self.d_state = d_state

        # S4D-Lin initialization. Negative real parts guarantee a stable
        # continuous system; linearly spaced imaginary parts cover timescales.
        self.log_dt = nn.Parameter(
            torch.empty(d_model).uniform_(math.log(1e-3), math.log(1e-1))
        )
        self.log_a_real = nn.Parameter(torch.full((d_state,), math.log(0.5)))
        self.a_imag = nn.Parameter(math.pi * torch.arange(d_state).float())

        scale = 1.0 / math.sqrt(d_state)
        self.c_real = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.d = nn.Parameter(torch.ones(d_model))

        self._state: Tensor | None = None

    @property
    def state(self) -> Tensor | None:
        """Current complex recurrent state, exposed for diagnostics/tests."""
        return self._state

    def _continuous_a(self) -> Tensor:
        return torch.complex(-torch.exp(self.log_a_real), self.a_imag)

    def _discretize(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return discrete A/B and complex C with shapes ``[H, N]``."""
        a = self._continuous_a().unsqueeze(0)
        dt = torch.exp(self.log_dt).unsqueeze(1)
        discrete_a = torch.exp(dt * a)
        discrete_b = (discrete_a - 1.0) / a
        c = torch.complex(self.c_real, self.c_imag)
        return discrete_a, discrete_b, c

    def _kernel(self, length: int) -> Tensor:
        """Construct the real causal impulse response ``[H, L]``."""
        discrete_a, discrete_b, c = self._discretize()
        steps = torch.arange(length, device=c.device, dtype=c.real.dtype)
        powers = discrete_a.unsqueeze(-1) ** steps
        impulse_state = discrete_b.unsqueeze(-1) * powers
        return 2.0 * (c.unsqueeze(-1) * impulse_state).real.sum(dim=1)

    def _real_kernel(self, length: int) -> Tensor:
        """Construct the same kernel using only real-valued exportable ops."""
        a_real = -torch.exp(self.log_a_real).unsqueeze(0)
        a_imag = self.a_imag.unsqueeze(0)
        dt = torch.exp(self.log_dt).unsqueeze(1)
        scaled_real = dt * a_real
        scaled_imag = dt * a_imag

        magnitude = torch.exp(scaled_real)
        discrete_a_real = magnitude * torch.cos(scaled_imag)
        discrete_a_imag = magnitude * torch.sin(scaled_imag)
        denominator = a_real.square() + a_imag.square()
        numerator_real = discrete_a_real - 1.0
        discrete_b_real = (
            numerator_real * a_real + discrete_a_imag * a_imag
        ) / denominator
        discrete_b_imag = (
            discrete_a_imag * a_real - numerator_real * a_imag
        ) / denominator

        steps = torch.arange(
            length, device=dt.device, dtype=dt.dtype
        ).view(1, 1, -1)
        power_magnitude = torch.exp(scaled_real.unsqueeze(-1) * steps)
        power_angle = scaled_imag.unsqueeze(-1) * steps
        power_real = power_magnitude * torch.cos(power_angle)
        power_imag = power_magnitude * torch.sin(power_angle)

        impulse_real = (
            discrete_b_real.unsqueeze(-1) * power_real
            - discrete_b_imag.unsqueeze(-1) * power_imag
        )
        impulse_imag = (
            discrete_b_real.unsqueeze(-1) * power_imag
            + discrete_b_imag.unsqueeze(-1) * power_real
        )
        response = (
            self.c_real.unsqueeze(-1) * impulse_real
            - self.c_imag.unsqueeze(-1) * impulse_imag
        )
        return 2.0 * response.sum(dim=1)

    @staticmethod
    def _causal_convolution(input: Tensor, kernel: Tensor, use_fft: bool) -> Tensor:
        length = input.shape[-1]
        if use_fft:
            fft_size = 2 * length
            input_fft = torch.fft.rfft(input, n=fft_size)
            kernel_fft = torch.fft.rfft(kernel, n=fft_size)
            return torch.fft.irfft(input_fft * kernel_fft, n=fft_size)[..., :length]

        # Conv1d computes cross-correlation, so reverse the impulse response.
        weight = kernel.flip(-1).unsqueeze(1)
        padded = F.pad(input, (length - 1, 0))
        return F.conv1d(padded, weight, groups=input.shape[1])

    def _initial_state_response(
        self, state: Tensor, length: int, discrete_a: Tensor, c: Tensor
    ) -> Tensor:
        steps = torch.arange(
            1, length + 1, device=c.device, dtype=c.real.dtype
        )
        powers = discrete_a.unsqueeze(-1) ** steps
        return 2.0 * (
            c.unsqueeze(0).unsqueeze(-1)
            * state.unsqueeze(-1)
            * powers.unsqueeze(0)
        ).real.sum(dim=2)

    @staticmethod
    def _final_state(
        input: Tensor,
        state: Tensor,
        discrete_a: Tensor,
        discrete_b: Tensor,
    ) -> Tensor:
        length = input.shape[-1]
        reverse_steps = torch.arange(
            length - 1,
            -1,
            -1,
            device=input.device,
            dtype=input.dtype,
        )
        powers = discrete_a.unsqueeze(-1) ** reverse_steps
        driven = (
            input.unsqueeze(2)
            * (discrete_b.unsqueeze(-1) * powers).unsqueeze(0)
        ).sum(dim=-1)
        return discrete_a.pow(length).unsqueeze(0) * state + driven

    def forward(
        self, input: Tensor, *, use_fft: bool = True, update_state: bool = True
    ) -> Tensor:
        """Process ``[batch, channels, time]`` while preserving stream state."""
        if input.ndim != 3 or input.shape[1] != self.d_model:
            raise ValueError(
                f"Expected [batch, {self.d_model}, time], got {tuple(input.shape)}"
            )
        if input.shape[-1] == 0:
            raise ValueError("S4DLayer does not accept empty sequences")

        discrete_a, discrete_b, c = self._discretize()
        batch = input.shape[0]
        if self._state is None:
            state = torch.zeros(
                batch,
                self.d_model,
                self.d_state,
                dtype=discrete_a.dtype,
                device=input.device,
            )
        else:
            if self._state.shape[0] != batch:
                raise ValueError(
                    "Cached S4D state batch size does not match input; "
                    "call reset_state() between differently sized streams"
                )
            state = self._state

        kernel = self._kernel(input.shape[-1])
        output = self._causal_convolution(input, kernel, use_fft)
        output = output + self._initial_state_response(
            state, input.shape[-1], discrete_a, c
        )
        output = output + self.d.view(1, -1, 1) * input

        if update_state:
            self._state = self._final_state(input, state, discrete_a, discrete_b)
        return output

    def forward_stateless_direct(self, input: Tensor) -> Tensor:
        """Real-only, stateless sequence path used by model exporters."""
        kernel = self._real_kernel(input.shape[-1])
        output = self._causal_convolution(input, kernel, use_fft=False)
        return output + self.d.view(1, -1, 1) * input

    def step(self, input: Tensor) -> Tensor:
        """Process one sample per stream using the cached recurrence."""
        if input.ndim != 2 or input.shape[1] != self.d_model:
            raise ValueError(
                f"Expected [batch, {self.d_model}], got {tuple(input.shape)}"
            )
        discrete_a, discrete_b, c = self._discretize()
        if self._state is None:
            self._state = torch.zeros(
                input.shape[0],
                self.d_model,
                self.d_state,
                dtype=discrete_a.dtype,
                device=input.device,
            )
        elif self._state.shape[0] != input.shape[0]:
            raise ValueError(
                "Cached S4D state batch size does not match input; "
                "call reset_state() between differently sized streams"
            )

        self._state = (
            discrete_a.unsqueeze(0) * self._state
            + discrete_b.unsqueeze(0) * input.unsqueeze(-1)
        )
        output = 2.0 * (c.unsqueeze(0) * self._state).real.sum(dim=-1)
        return output + self.d.unsqueeze(0) * input

    def reset_state(self) -> None:
        self._state = None

    def detach_state(self) -> None:
        if self._state is not None:
            self._state = self._state.detach()


class S4DBlock(nn.Module):
    """Normalized residual S4D block with channel mixing."""

    def __init__(self, params: SSMParams):
        super().__init__()
        self.prenorm = params.prenorm
        self.norm = nn.LayerNorm(params.d_model)
        self.s4 = S4DLayer(params.d_model, params.d_state)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(params.dropout)
        self.mix = nn.Conv1d(params.d_model, params.d_model, kernel_size=1)

    def _normalize_sequence(self, input: Tensor) -> Tensor:
        return self.norm(input.transpose(1, 2)).transpose(1, 2)

    def forward(
        self, input: Tensor, *, use_fft: bool = True, update_state: bool = True
    ) -> Tensor:
        residual = input
        hidden = self._normalize_sequence(input) if self.prenorm else input
        hidden = self.s4(hidden, use_fft=use_fft, update_state=update_state)
        hidden = self.mix(self.dropout(self.activation(hidden)))
        output = residual + hidden
        return output if self.prenorm else self._normalize_sequence(output)

    def step(self, input: Tensor) -> Tensor:
        residual = input
        hidden = self.norm(input) if self.prenorm else input
        hidden = self.s4.step(hidden)
        hidden = self.mix(
            self.dropout(self.activation(hidden)).unsqueeze(-1)
        ).squeeze(-1)
        output = residual + hidden
        return output if self.prenorm else self.norm(output)

    def forward_export(self, input: Tensor) -> Tensor:
        """Stateless real-valued path for TorchScript and ONNX."""
        residual = input
        hidden = self._normalize_sequence(input) if self.prenorm else input
        hidden = self.s4.forward_stateless_direct(hidden)
        hidden = self.mix(self.dropout(self.activation(hidden)))
        output = residual + hidden
        return output if self.prenorm else self._normalize_sequence(output)

    def reset_state(self) -> None:
        self.s4.reset_state()

    def detach_state(self) -> None:
        self.s4.detach_state()


class S4Model(BaseNeuralFXModel):
    """Stacked S4D model for causal neural audio effects."""

    model_type = "s4"

    def __init__(self, config: ModelConfig):
        super().__init__(config.sample_rate)
        if config.type != "s4":
            raise ValueError(f"Expected model type 's4', got {config.type!r}")
        if not isinstance(config.params, SSMParams):
            raise ValueError(f"Expected SSMParams, got {type(config.params)}")

        self.config = config
        self.params = config.params
        self.input_projection = nn.Conv1d(
            config.input_size, self.params.d_model, kernel_size=1
        )
        self.blocks = nn.ModuleList(
            S4DBlock(self.params) for _ in range(self.params.num_layers)
        )
        self.output_projection = nn.Conv1d(
            self.params.d_model, config.output_size, kernel_size=1
        )

    def _forward_sequence(
        self, input: Tensor, *, use_fft: bool, update_state: bool
    ) -> Tensor:
        hidden = self.input_projection(input)
        for block in self.blocks:
            hidden = block(
                hidden, use_fft=use_fft, update_state=update_state
            )
        return self.output_projection(hidden)

    def forward(
        self,
        input: Tensor,
        reset_state: bool = False,
        detach_state: bool = False,
    ) -> Tensor:
        if input.ndim != 3 or input.shape[1] != self.config.input_size:
            raise ValueError(
                f"Expected [batch, {self.config.input_size}, time], "
                f"got {tuple(input.shape)}"
            )
        if reset_state:
            self.reset_state()
        if detach_state:
            self.detach_state()
        return self._forward_sequence(input, use_fft=True, update_state=True)

    def _step(self, input: Tensor) -> Tensor:
        hidden = self.input_projection(input.unsqueeze(-1)).squeeze(-1)
        for block in self.blocks:
            hidden = block.step(hidden)
        return self.output_projection(hidden.unsqueeze(-1)).squeeze(-1)

    def _forward_export(self, input: Tensor) -> Tensor:
        hidden = self.input_projection(input)
        for block in self.blocks:
            hidden = block.forward_export(hidden)
        return self.output_projection(hidden)

    def process_sample(self, input: Tensor, reset: bool = False) -> Tensor:
        if reset:
            self.reset_state()
        original_ndim = input.ndim
        if original_ndim == 1:
            input = input.unsqueeze(0)
        elif original_ndim == 3 and input.shape[-1] == 1:
            input = input.squeeze(-1)
        if input.ndim != 2 or input.shape[1] != self.config.input_size:
            raise ValueError(
                f"Expected [channels] or [batch, {self.config.input_size}], "
                f"got {tuple(input.shape)}"
            )
        with torch.no_grad():
            output = self._step(input)
        return output.squeeze(0) if original_ndim == 1 else output

    def reset_state(self) -> None:
        for block in self.blocks:
            block.reset_state()

    def detach_state(self) -> None:
        for block in self.blocks:
            block.detach_state()

    @property
    def receptive_field(self) -> int:
        """Return -1 to denote the SSM's theoretically unbounded history."""
        return -1

    @classmethod
    def from_config(cls, config: dict[str, Any] | ModelConfig) -> "S4Model":
        if isinstance(config, dict):
            model_type = config.get("type", "s4")
            config = ModelConfig(
                type=model_type,
                params=_load_model_params(model_type, config.get("params", {})),
                input_size=config.get("input_size", 1),
                output_size=config.get("output_size", 1),
                sample_rate=config.get("sample_rate", 48_000),
            )
        return cls(config)

    def _export_wrapper(self) -> nn.Module:
        class ExportWrapper(nn.Module):
            def __init__(self, model: "S4Model"):
                super().__init__()
                self.model = model

            def forward(self, input: Tensor) -> Tensor:
                # Direct causal convolution avoids FFT operators unsupported
                # by common ONNX runtimes. The real-valued kernel formulation
                # also avoids complex tensors unsupported by ONNX.
                return self.model._forward_export(input)

        return ExportWrapper(self)

    def export_torchscript(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.eval()
        self.reset_state()
        wrapper = self._export_wrapper()
        example = torch.randn(1, self.config.input_size, 512)
        traced = torch.jit.trace(wrapper, example)
        traced.save(str(path))

    def export_onnx(self, path: str | Path, opset_version: int = 17) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.eval()
        self.reset_state()
        wrapper = self._export_wrapper()
        example = torch.randn(1, self.config.input_size, 512)
        torch.onnx.export(
            wrapper,
            example,
            path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,
        )

    def export_rtneural(self, path: str | Path) -> None:
        raise UnsupportedExportFormatError(self.model_type, "rtneural")
