import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .config import NeuralFXConfig, config_from_dict, load_config
from .models import BaseNeuralFXModel, create_model_from_config

ConfigOrigin = Literal["explicit_file", "checkpoint", "sidecar"]
WeightsOrigin = Literal["checkpoint", "initialized"]


@dataclass(frozen=True)
class ModelProvenance:
    checkpoint_path: Path | None
    config_path: Path | None
    config_origin: ConfigOrigin
    weights_origin: WeightsOrigin


@dataclass(frozen=True)
class LoadedModel:
    model: BaseNeuralFXModel
    config: NeuralFXConfig
    provenance: ModelProvenance


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping")
    return checkpoint


def _load_artifact_config(
    checkpoint: Mapping[str, object] | None,
    checkpoint_path: Path | None,
    config_path: Path | None,
) -> tuple[NeuralFXConfig, ConfigOrigin, Path | None]:
    if config_path is not None:
        return load_config(config_path), "explicit_file", config_path

    if checkpoint is not None and "neural_fx_config" in checkpoint:
        config = checkpoint["neural_fx_config"]
        if not isinstance(config, dict):
            raise ValueError("Checkpoint configuration must contain a mapping")
        return config_from_dict(config), "checkpoint", None

    if checkpoint_path is not None:
        sidecar_path = checkpoint_path.with_suffix(".meta.json")
        if sidecar_path.is_file():
            metadata = json.loads(sidecar_path.read_text())
            if not isinstance(metadata, dict) or not isinstance(
                metadata.get("config"), dict
            ):
                raise ValueError("Checkpoint metadata does not contain a config")
            return config_from_dict(metadata["config"]), "sidecar", sidecar_path
        raise ValueError(
            "Checkpoint does not contain a complete neural-fx configuration. "
            "Pass --config for legacy checkpoints."
        )

    raise ValueError("A checkpoint or config is required")


def _model_state(checkpoint: Mapping[str, object]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint does not contain a model state dictionary")

    lightning_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if isinstance(key, str)
        and key.startswith("model.")
        and isinstance(value, torch.Tensor)
    }
    if lightning_state:
        return lightning_state

    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    return dict(state)


def load_model(
    *,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> LoadedModel:
    """Decode supported model artifacts using one configuration precedence rule."""
    resolved_checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else None
    )
    resolved_config = (
        Path(config_path).expanduser().resolve() if config_path is not None else None
    )
    checkpoint = (
        _load_checkpoint(resolved_checkpoint)
        if resolved_checkpoint is not None
        else None
    )
    config, config_origin, resolved_config_source = _load_artifact_config(
        checkpoint,
        resolved_checkpoint,
        resolved_config,
    )

    model = create_model_from_config(config.model)
    if checkpoint is not None:
        model.load_state_dict(_model_state(checkpoint), strict=True)
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        config=config,
        provenance=ModelProvenance(
            checkpoint_path=resolved_checkpoint,
            config_path=resolved_config_source,
            config_origin=config_origin,
            weights_origin="checkpoint" if checkpoint is not None else "initialized",
        ),
    )
