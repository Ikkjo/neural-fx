import dataclasses
from typing import Any, Callable

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import LossConfig, NeuralFXConfig
from ..data.dataset import AudioDataset
from ..data.transforms import build_augmentation_transform
from ..losses.audio_losses import ESR, MSE, MultiResolutionSTFTLoss
from ..preprocessing.latency import LatencyCalibration


class NeuralFXModule(L.LightningModule):
    """PyTorch Lightning module for training neural audio effects."""

    def __init__(
        self,
        model,
        config: NeuralFXConfig,
        train_latency: LatencyCalibration | None = None,
        val_latency: LatencyCalibration | None = None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.train_latency = train_latency
        self.val_latency = val_latency
        # Save hyperparameters explicitly to avoid -1 values that break TensorBoard
        # We manually extract scalar values from config to avoid nested dataclass issues
        hparams = {
            "model_type": config.model.type,
            "sample_rate": config.model.sample_rate,
            "batch_size": config.training.batch_size,
            "segment_length": config.training.segment_length,
            "lr": config.optimizer.lr,
            "optimizer": config.optimizer.type,
            "scheduler": config.lr_scheduler.type,
            "loss_type": config.loss.type,
        }
        self.save_hyperparameters(hparams)

        self.tbptt_config = config.training.tbptt
        self.burn_in = config.loss.mask_first
        self.loss_fn = self._build_loss(config.loss)
        self.validation_loss_fn = (
            self._build_loss(config.validation_loss)
            if config.validation_loss is not None
            else None
        )
        self._validation_predictions: list[Tensor] = []
        self._validation_targets: list[Tensor] = []

        # Build augmentation transform if enabled
        self.transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None
        if config.training.augmentation and config.training.augmentation.enabled:
            self.transform = build_augmentation_transform(
                random_gain=config.training.augmentation.random_gain,
                gain_min_db=config.training.augmentation.gain_min_db,
                gain_max_db=config.training.augmentation.gain_max_db,
                gain_p=config.training.augmentation.gain_p,
                add_noise=config.training.augmentation.add_noise,
                noise_min_snr_db=config.training.augmentation.noise_min_snr_db,
                noise_max_snr_db=config.training.augmentation.noise_max_snr_db,
                noise_p=config.training.augmentation.noise_p,
            )

    def _build_loss(
        self, loss_config: LossConfig
    ) -> Callable[[Tensor, Tensor], Tensor]:
        weights = loss_config.weights
        if weights is None:
            return MSE

        esr_weight = weights.esr
        mse_weight = weights.mse
        stft_weight = weights.stft
        if any(weight < 0 for weight in (esr_weight, mse_weight, stft_weight)):
            raise ValueError("Loss weights must be non-negative")
        if esr_weight == mse_weight == stft_weight == 0:
            raise ValueError("At least one loss weight must be positive")

        stft_loss_fn: MultiResolutionSTFTLoss | None = None
        if stft_weight > 0:
            if loss_config.stft is None or not loss_config.stft.enabled:
                raise ValueError("STFT loss weight requires an enabled STFT loss")
            stft_loss_fn = MultiResolutionSTFTLoss(
                fft_sizes=loss_config.stft.fft_sizes,
                hop_sizes=loss_config.stft.hop_sizes,
                win_sizes=loss_config.stft.win_sizes,
                sc_loss_weight=loss_config.stft.sc_weight,
                mag_loss_weight=loss_config.stft.mag_weight,
                mode=loss_config.stft.mode,
            )

        pre_emphasis = loss_config.pre_emphasis
        pre_emphasis_coeff = (
            pre_emphasis.coef
            if pre_emphasis is not None and pre_emphasis.enabled
            else None
        )

        def loss_fn(pred: Tensor, target: Tensor) -> Tensor:
            terms: list[Tensor] = []
            if esr_weight > 0:
                terms.append(
                    esr_weight
                    * ESR(
                        pred,
                        target,
                        pre_emphasis_coeff=pre_emphasis_coeff,
                        mode=loss_config.esr_mode,
                    )
                )
            if mse_weight > 0:
                terms.append(mse_weight * MSE(pred, target))
            if stft_loss_fn is not None:
                terms.append(stft_weight * stft_loss_fn(pred, target))
            return sum(terms[1:], start=terms[0])

        return loss_fn

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Store the complete typed configuration with every checkpoint."""
        checkpoint["neural_fx_config"] = dataclasses.asdict(self.config)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        batch_size, seq_len = x.shape

        if self.tbptt_config and self.tbptt_config.enabled:
            return self._training_step_tbptt(x, y, batch_idx)
        else:
            return self._training_step_full(x, y)

    def _training_step_full(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # [B, 1, T]
        y = y.unsqueeze(1)

        pred = self.model(x, reset_state=True)

        if pred.shape[-1] != y.shape[-1]:
            pred = pred[..., -y.shape[-1] :]

        if self.burn_in > 0:
            pred = pred[..., self.burn_in :]
            y = y[..., self.burn_in :]

        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _training_step_tbptt(self, x: Tensor, y: Tensor, batch_idx: int) -> Tensor:
        x = x.unsqueeze(1)  # [B, 1, T]
        y = y.unsqueeze(1)

        segment_length = x.shape[-1]
        truncate = segment_length // 2

        predictions: list[Tensor] = []
        targets: list[Tensor] = []

        for start in range(0, segment_length, truncate):
            end = min(start + truncate, segment_length)
            x_seg = x[..., start:end]
            y_seg = y[..., start:end]

            if start == 0:
                self.model.reset_state()
            else:
                self.model.detach_state()

            pred_seg = self.model(x_seg)

            effective_start = self.burn_in if start == 0 else 0
            if pred_seg.shape[-1] > effective_start:
                pred_loss = pred_seg[..., effective_start:]
                y_loss = y_seg[..., effective_start:]
                predictions.append(pred_loss)
                targets.append(y_loss)

        if predictions:
            loss = self.loss_fn(
                torch.cat(predictions, dim=-1), torch.cat(targets, dim=-1)
            )
            self.log("train_loss", loss, prog_bar=True)
            return loss
        return torch.tensor(0.0, device=x.device)

    def on_validation_epoch_start(self) -> None:
        self._validation_predictions.clear()
        self._validation_targets.clear()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor | None:
        x, y = batch
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        if batch_idx == 0 or self.validation_loss_fn is None:
            self.model.reset_state()
        pred = self.model(x)

        if self.validation_loss_fn is not None:
            self._validation_predictions.append(pred.detach())
            self._validation_targets.append(y.detach())
            return None

        if self.burn_in > 0:
            pred = pred[..., self.burn_in :]
            y = y[..., self.burn_in :]

        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.validation_loss_fn is None or not self._validation_predictions:
            return
        pred = torch.cat(self._validation_predictions, dim=-1)
        target = torch.cat(self._validation_targets, dim=-1)
        assert self.config.validation_loss is not None
        mask_first = self.config.validation_loss.mask_first
        loss = self.validation_loss_fn(
            pred[..., mask_first:], target[..., mask_first:]
        )
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        opt_type = self.config.optimizer.type.lower()
        lr = self.config.optimizer.lr
        weight_decay = self.config.optimizer.weight_decay

        if opt_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        scheduler_type = self.config.lr_scheduler.type.lower()
        gamma = self.config.lr_scheduler.gamma

        if scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=gamma
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        return [optimizer], [scheduler]

    def _create_train_dataset(self) -> AudioDataset:
        """Create training dataset."""
        input_context = 0
        if self.config.training.use_receptive_field_context:
            input_context = int(self.model.receptive_field) - 1
        return AudioDataset(
            input_path=self.config.data.train.input,
            target_path=self.config.data.train.target,
            segment_length=self.config.training.segment_length,
            sample_rate=self.config.sample_rate,
            random_segments=self.config.training.random_segments,
            transform=self.transform,
            latency_calibration=self.train_latency,
            normalize=self.config.data.normalize,
            input_context=input_context,
        )

    def _create_val_dataset(self) -> AudioDataset | None:
        """Create validation dataset if paths are provided."""
        if self.config.data.val is None:
            return None

        return AudioDataset(
            input_path=self.config.data.val.input,
            target_path=self.config.data.val.target,
            segment_length=(
                self.config.training.validation_segment_length
                or self.config.training.segment_length
            ),
            sample_rate=self.config.sample_rate,
            random_segments=False,  # Sequential for validation
            transform=None,  # No augmentation for validation
            latency_calibration=self.val_latency,
            normalize=self.config.data.normalize,
            include_partial_segment=self.validation_loss_fn is not None,
        )

    def train_dataloader(self) -> DataLoader:
        dataset = self._create_train_dataset()
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        dataset = self._create_val_dataset()
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=(
                1
                if self.validation_loss_fn is not None
                else self.config.training.batch_size
            ),
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
        )
