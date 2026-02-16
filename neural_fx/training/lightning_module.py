from typing import Callable

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import NeuralFXConfig, LossConfig
from ..data.dataset import AudioDataset
from ..data.transforms import build_augmentation_transform
from ..losses.audio_losses import ESR, MSE, MultiResolutionSTFTLoss


class NeuralFXModule(L.LightningModule):
    """PyTorch Lightning module for training neural audio effects."""

    def __init__(self, model, config: NeuralFXConfig):
        super().__init__()
        self.model = model
        self.config = config
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
        self.burn_in = self.tbptt_config.burn_in if self.tbptt_config else 0
        self.loss_fn = self._build_loss(config.loss)

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

    def _build_loss(self, loss_config: LossConfig):
        """Build loss function from config."""
        # Initialize STFT loss if enabled
        stft_loss_fn: MultiResolutionSTFTLoss | None = None
        if loss_config.stft and loss_config.stft.enabled:
            # Ensure fft_sizes is not None before passing
            fft_sizes = loss_config.stft.fft_sizes or [512, 1024, 2048]
            hop_sizes = loss_config.stft.hop_sizes
            win_sizes = loss_config.stft.win_sizes
            stft_loss_fn = MultiResolutionSTFTLoss(
                fft_sizes=fft_sizes,
                hop_sizes=hop_sizes,
                win_sizes=win_sizes,
                sc_loss_weight=loss_config.stft.sc_weight,
                mag_loss_weight=loss_config.stft.mag_weight,
            )

        def loss_fn(pred: Tensor, target: Tensor) -> Tensor:
            loss = torch.tensor(0.0, device=pred.device)

            weights = loss_config.weights
            if weights is None:
                # Default to MSE only
                return MSE(pred, target)

            # Get weight values with defaults
            esr_weight = getattr(weights, "esr", 0.0)
            mse_weight = getattr(weights, "mse", 1.0)
            stft_weight = getattr(weights, "stft", 0.0)

            # ESR loss
            if esr_weight > 0:
                loss = loss + esr_weight * ESR(pred, target)

            # MSE loss
            if mse_weight > 0:
                loss = loss + mse_weight * MSE(pred, target)

            # Multi-resolution STFT loss
            if stft_weight > 0 and stft_loss_fn is not None:
                loss = loss + stft_weight * stft_loss_fn(pred, target)

            # Default to MSE if no weights specified
            if loss.item() < 1e-8:
                loss = MSE(pred, target)

            return loss

        return loss_fn

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

        total_loss = torch.tensor(0.0, device=x.device)
        total_samples = 0

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
                loss = self.loss_fn(pred_loss, y_loss)

                # Weight by number of effective samples in this segment
                num_samples = pred_loss.numel()
                total_loss = total_loss + loss * num_samples
                total_samples += num_samples

        if total_samples > 0:
            avg_loss = total_loss / total_samples
            self.log("train_loss", avg_loss, prog_bar=True)
            return avg_loss
        return torch.tensor(0.0, device=x.device)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        self.model.reset_state()
        pred = self.model(x)

        if self.burn_in > 0:
            pred = pred[..., self.burn_in :]
            y = y[..., self.burn_in :]

        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        opt_type = self.config.optimizer.type.lower()
        lr = self.config.optimizer.lr

        if opt_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
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
        return AudioDataset(
            input_path=self.config.data.train.input,
            target_path=self.config.data.train.target,
            segment_length=self.config.training.segment_length,
            sample_rate=self.config.data.sample_rate,
            random_segments=self.config.training.random_segments,
            transform=self.transform,
        )

    def _create_val_dataset(self) -> AudioDataset | None:
        """Create validation dataset if paths are provided."""
        if self.config.data.val is None:
            return None

        return AudioDataset(
            input_path=self.config.data.val.input,
            target_path=self.config.data.val.target,
            segment_length=self.config.training.segment_length,
            sample_rate=self.config.data.sample_rate,
            random_segments=False,  # Sequential for validation
            transform=None,  # No augmentation for validation
        )

    def train_dataloader(self) -> DataLoader:
        dataset = self._create_train_dataset()
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        dataset = self._create_val_dataset()
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
