"""Tests for the optional Mamba model and its platform gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from neural_fx.config import ModelConfig, SSMParams, load_config
from neural_fx.models.errors import (
    OptionalDependencyError,
    UnsupportedExportFormatError,
)
from neural_fx.models.mamba import MAMBA_INSTALL_HINT, MambaModel
import neural_fx.models.mamba as mamba_module


class FakeMamba(nn.Module):
    """Small CPU stand-in with the public cache/step shape used upstream."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        layer_idx: int,
    ):
        super().__init__()
        del expand
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.layer_idx = layer_idx
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * self.scale

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        del max_seqlen
        dtype = dtype or self.scale.dtype
        device = self.scale.device
        conv_state = torch.zeros(
            batch_size, self.d_model, self.d_conv, dtype=dtype, device=device
        )
        ssm_state = torch.zeros(
            batch_size, self.d_model, self.d_state, dtype=dtype, device=device
        )
        return conv_state, ssm_state

    def step(self, x, conv_state, ssm_state):
        previous = ssm_state[:, :, :1].transpose(1, 2)
        output = x * self.scale + previous
        next_value = output.transpose(1, 2)
        conv_state = conv_state.clone()
        ssm_state = ssm_state.clone()
        conv_state[:, :, 0:1] = next_value
        ssm_state[:, :, 0:1] = next_value
        return output, conv_state, ssm_state


@pytest.fixture
def mamba_config():
    return ModelConfig(
        type="mamba",
        params=SSMParams(
            d_model=8,
            d_state=4,
            num_layers=2,
            dropout=0.0,
            prenorm=True,
            d_conv=3,
            expand=2,
        ),
        input_size=1,
        output_size=1,
        sample_rate=44_100,
    )


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(mamba_module, "_load_mamba_class", lambda: FakeMamba)


def test_non_linux_platform_is_rejected(monkeypatch, mamba_config):
    monkeypatch.setattr(mamba_module.sys, "platform", "win32")

    with pytest.raises(OptionalDependencyError, match="Linux/CUDA") as error:
        MambaModel(mamba_config)

    assert MAMBA_INSTALL_HINT in str(error.value)


def test_linux_without_cuda_is_rejected(monkeypatch, mamba_config):
    monkeypatch.setattr(mamba_module.sys, "platform", "linux")
    monkeypatch.setattr(mamba_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(OptionalDependencyError, match="CUDA-enabled PyTorch"):
        MambaModel(mamba_config)


def test_missing_backend_is_actionable(monkeypatch, mamba_config):
    monkeypatch.setattr(mamba_module.sys, "platform", "linux")
    monkeypatch.setattr(mamba_module.torch.cuda, "is_available", lambda: True)

    def missing_backend():
        raise ModuleNotFoundError("No module named 'mamba_ssm'")

    monkeypatch.setattr(mamba_module, "_import_mamba_class", missing_backend)

    with pytest.raises(OptionalDependencyError, match="mamba-ssm") as error:
        MambaModel(mamba_config)

    assert MAMBA_INSTALL_HINT in str(error.value)


def test_incompatible_backend_api_is_actionable(monkeypatch, mamba_config):
    class IncompatibleMamba:
        def __init__(self, **kwargs):
            del kwargs
            raise TypeError("unsupported constructor")

    monkeypatch.setattr(
        mamba_module, "_load_mamba_class", lambda: IncompatibleMamba
    )

    with pytest.raises(OptionalDependencyError, match="compatible mamba-ssm 2.x"):
        MambaModel(mamba_config)


def test_invalid_model_config_fails_before_backend_load(monkeypatch, mamba_config):
    backend_loaded = False

    def mark_loaded():
        nonlocal backend_loaded
        backend_loaded = True
        return FakeMamba

    monkeypatch.setattr(mamba_module, "_load_mamba_class", mark_loaded)
    mamba_config.type = "s4"

    with pytest.raises(ValueError, match="requires type 'mamba'"):
        MambaModel(mamba_config)

    assert not backend_loaded


def test_training_forward_uses_parallel_backend(fake_backend, mamba_config):
    model = MambaModel(mamba_config)
    model.train()

    output = model(torch.randn(2, 1, 16), reset_state=True)

    assert output.shape == (2, 1, 16)
    assert model.sample_rate == 44_100
    assert model.receptive_field == -1
    assert model._stream_states is None


def test_eval_forward_preserves_state_across_blocks(fake_backend, mamba_config):
    torch.manual_seed(7)
    model = MambaModel(mamba_config).eval()
    audio = torch.randn(1, 1, 12)

    model.reset_state()
    full = model(audio)
    model.reset_state()
    chunked = torch.cat((model(audio[..., :5]), model(audio[..., 5:])), dim=-1)

    assert torch.allclose(full, chunked, atol=1e-6)
    assert model._stream_states is not None
    assert len(model._stream_states) == mamba_config.params.num_layers


def test_process_sample_matches_block_streaming(fake_backend, mamba_config):
    torch.manual_seed(8)
    model = MambaModel(mamba_config).eval()
    audio = torch.randn(1, 1, 10)

    model.reset_state()
    block_output = model(audio)
    model.reset_state()
    samples = [model.process_sample(audio[0, :, index]) for index in range(10)]
    sample_output = torch.stack(samples).view(1, 1, -1)

    assert torch.allclose(block_output, sample_output, atol=1e-6)


def test_reset_and_detach_clear_or_detach_caches(fake_backend, mamba_config):
    model = MambaModel(mamba_config).eval()
    model(torch.randn(1, 1, 2))
    assert model._stream_states is not None

    model._stream_states = [
        (conv.requires_grad_(), ssm.requires_grad_())
        for conv, ssm in model._stream_states
    ]
    model.detach_state()
    assert all(
        not tensor.requires_grad
        for state in model._stream_states
        for tensor in state
    )

    model.reset_state()
    assert model._stream_states is None
    assert model._stream_batch_size is None


def test_batch_size_change_reallocates_caches(fake_backend, mamba_config):
    model = MambaModel(mamba_config).eval()

    model(torch.randn(1, 1, 2))
    model(torch.randn(3, 1, 2))

    assert model._stream_batch_size == 3
    assert model._stream_states is not None
    assert all(state[0].shape[0] == 3 for state in model._stream_states)


@pytest.mark.parametrize("export_format", ["onnx", "torchscript", "rtneural"])
def test_unsupported_exports_are_explicit(
    fake_backend, mamba_config, tmp_path, export_format
):
    model = MambaModel(mamba_config)
    export_method = getattr(model, f"export_{export_format}")

    with pytest.raises(
        UnsupportedExportFormatError, match=export_format
    ):
        export_method(tmp_path / f"model.{export_format}")


def test_from_dict_uses_shared_ssm_params(fake_backend):
    model = MambaModel.from_config(
        {
            "type": "mamba",
            "input_size": 1,
            "output_size": 2,
            "sample_rate": 44_100,
            "params": {
                "d_model": 12,
                "d_state": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "prenorm": False,
                "d_conv": 4,
                "expand": 2,
            },
        }
    )

    assert isinstance(model.params, SSMParams)
    assert model.config.output_size == 2
    assert model.sample_rate == 44_100


def test_all_size_configs_disable_tbptt():
    config_dir = Path(__file__).parents[1] / "configs" / "models" / "mamba"

    configs = [load_config(path) for path in sorted(config_dir.glob("*.yaml"))]

    assert [config.name for config in configs] == [
        "mamba_large",
        "mamba_medium",
        "mamba_nano",
        "mamba_small",
        "mamba_xl",
    ]
    assert all(config.model.type == "mamba" for config in configs)
    assert all(config.training.tbptt is not None for config in configs)
    assert all(not config.training.tbptt.enabled for config in configs)


def _real_backend_available() -> bool:
    if not sys.platform.startswith("linux") or not torch.cuda.is_available():
        return False
    try:
        mamba_module._import_mamba_class()
    except (ImportError, OSError, RuntimeError):
        return False
    return True


@pytest.mark.skipif(
    not _real_backend_available(),
    reason="requires mamba-ssm on a compatible Linux/CUDA runtime",
)
def test_real_backend_forward_and_step():
    config = ModelConfig(
        type="mamba",
        params=SSMParams(d_model=8, d_state=8, num_layers=1),
    )
    model = MambaModel(config).cuda().eval()
    audio = torch.randn(1, 1, 4, device="cuda")

    output = model(audio)

    assert output.shape == audio.shape
    assert torch.isfinite(output).all()
