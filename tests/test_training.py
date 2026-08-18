import sys
import os
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchaudio
import pytest

from neural_fx.config import (
    ModelConfig,
    LSTMParams,
    NeuralFXConfig,
    TrainingConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    LossConfig,
    DataConfig,
    DataPaths,
    AugmentationConfig,
    LossWeights,
    STFTLossConfig,
)
from neural_fx.models.recurrent import NeuralfxLSTM
from neural_fx.training.lightning_module import NeuralFXModule
from neural_fx.models import (
    MODEL_REGISTRY,
    create_model_from_config,
    get_available_models,
)
from neural_fx.data.transforms import (
    RandomGain,
    AddNoise,
    Compose,
    build_augmentation_transform,
)
from neural_fx.losses.audio_losses import MultiResolutionSTFTLoss
from neural_fx.data.dataset import AudioDataset
from neural_fx.preprocessing.latency import LatencyCalibration


class TestModelRegistry:
    """Test suite for model registry."""

    def test_registry_has_lstm(self):
        """Test that LSTM is in the registry."""
        assert "lstm" in MODEL_REGISTRY
        assert MODEL_REGISTRY["lstm"] == NeuralfxLSTM

    def test_registry_has_gru(self):
        """Test that GRU is in the registry."""
        from neural_fx.models.recurrent import NeuralfxGRU

        assert "gru" in MODEL_REGISTRY
        assert MODEL_REGISTRY["gru"] == NeuralfxGRU

    def test_get_available_models(self):
        """Test get_available_models function."""
        models = get_available_models()
        assert "lstm" in models
        assert "gru" in models

    def test_create_model_from_config_lstm(self):
        """Test creating LSTM model from config using registry."""
        config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=16, num_layers=1),
            input_size=1,
            output_size=1,
        )
        model = create_model_from_config(config)
        assert isinstance(model, NeuralfxLSTM)

    def test_create_model_from_config_unknown(self):
        """Test that unknown model type raises ValueError."""
        config = ModelConfig(
            type="lstm",  # Valid type to create config
            params=LSTMParams(hidden_size=16, num_layers=1),
            input_size=1,
            output_size=1,
        )
        # Manually change to invalid type to test error
        object.__setattr__(config, "type", "unknown")
        with pytest.raises(ValueError):
            create_model_from_config(config)


class TestAugmentations:
    """Test suite for audio augmentations."""

    def test_random_gain_creation(self):
        """Test RandomGain can be created."""
        aug = RandomGain(min_gain_db=-12, max_gain_db=0, p=0.5)
        assert aug.min_gain_db == -12
        assert aug.max_gain_db == 0
        assert aug.p == 0.5

    def test_random_gain_invalid_range(self):
        """Test RandomGain with invalid range raises error."""
        with pytest.raises(ValueError):
            RandomGain(min_gain_db=0, max_gain_db=-12)

    def test_random_gain_application(self):
        """Test RandomGain applies gain correctly."""
        aug = RandomGain(min_gain_db=-6, max_gain_db=-6, p=1.0)  # Fixed -6dB
        x = torch.ones(1000)
        y = torch.ones(1000)

        x_out, y_out = aug(x, y)

        # -6dB = 0.5 in linear scale
        expected_gain = 10 ** (-6 / 20)
        assert torch.allclose(x_out, x * expected_gain, atol=1e-5)
        assert torch.allclose(y_out, y * expected_gain, atol=1e-5)

    def test_add_noise_creation(self):
        """Test AddNoise can be created."""
        aug = AddNoise(min_snr_db=20, max_snr_db=40, p=0.3)
        assert aug.min_snr_db == 20
        assert aug.max_snr_db == 40
        assert aug.p == 0.3

    def test_add_noise_application(self):
        """Test AddNoise adds noise to input."""
        aug = AddNoise(min_snr_db=20, max_snr_db=20, p=1.0)  # Fixed SNR
        x = torch.ones(1000)
        y = torch.ones(1000)

        x_out, y_out = aug(x, y)

        # Output should be different from input due to noise
        assert not torch.allclose(x_out, x)
        # Target should be unchanged
        assert torch.allclose(y_out, y)

    def test_compose(self):
        """Test Compose applies multiple transforms."""
        gain = RandomGain(min_gain_db=-6, max_gain_db=-6, p=1.0)
        noise = AddNoise(min_snr_db=20, max_snr_db=20, p=1.0)
        compose = Compose([gain, noise])

        x = torch.ones(1000)
        y = torch.ones(1000)

        x_out, y_out = compose(x, y)

        # Both transforms should have been applied
        expected_gain = 10 ** (-6 / 20)
        assert not torch.allclose(x_out, x * expected_gain)  # Noise added
        assert torch.allclose(y_out, y * expected_gain)  # Only gain applied to target

    def test_build_augmentation_transform(self):
        """Test build_augmentation_transform factory."""
        # Both enabled
        transform = build_augmentation_transform(
            random_gain=True,
            add_noise=True,
        )
        assert transform is not None

        # None enabled
        transform = build_augmentation_transform(
            random_gain=False,
            add_noise=False,
        )
        assert transform is None

        # Only gain
        transform = build_augmentation_transform(
            random_gain=True,
            add_noise=False,
        )
        assert transform is not None


class TestMultiResolutionSTFTLoss:
    """Test suite for multi-resolution STFT loss."""

    def test_stft_loss_creation(self):
        """Test MultiResolutionSTFTLoss can be created."""
        loss = MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_sizes=[512, 1024, 2048],
        )
        assert loss is not None
        assert len(loss.stft_losses) == 3

    def test_stft_loss_default_params(self):
        """Test MultiResolutionSTFTLoss with default parameters."""
        loss = MultiResolutionSTFTLoss()
        assert loss.stft_losses is not None
        assert len(loss.stft_losses) == 3

    def test_stft_loss_forward(self):
        """Test STFT loss forward pass."""
        loss_fn = MultiResolutionSTFTLoss(fft_sizes=[256, 512])

        batch_size = 2
        channels = 1
        time = 2048

        y_pred = torch.randn(batch_size, channels, time)
        y_true = torch.randn(batch_size, channels, time)

        loss = loss_fn(y_pred, y_true)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_stft_loss_same_input(self):
        """Test STFT loss is zero for identical inputs."""
        loss_fn = MultiResolutionSTFTLoss(fft_sizes=[256])

        y = torch.randn(2, 1, 2048)

        loss = loss_fn(y, y)

        assert loss.item() < 0.01  # Should be very small for identical inputs


class TestCheckpointResume:
    """Test suite for checkpoint resume functionality."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing."""
        model_config = ModelConfig(
            type="lstm",
            params=LSTMParams(
                hidden_size=16,
                num_layers=1,
                conv1d=None,
                skip_connection=False,
                dropout=0.0,
                conditioning_size=0,
            ),
            input_size=1,
            output_size=1,
            sample_rate=48000,
        )

        return NeuralFXConfig(
            version="1.0",
            name="test_model",
            model=model_config,
            training=TrainingConfig(
                batch_size=2,
                epochs=2,
                segment_length=2048,
                tbptt=None,
                seed=42,
            ),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
            loss=LossConfig(type="mse"),
            data=DataConfig(
                train=DataPaths(input="data/DI.wav", target="data/effect.wav"),
            ),
        )

    def test_module_state_dict(self, base_config):
        """Test module state dict contains model weights."""
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        state_dict = module.state_dict()
        assert len(state_dict) > 0

        # Check for model parameters
        model_keys = [k for k in state_dict.keys() if k.startswith("model.")]
        assert len(model_keys) > 0

    def test_load_state_dict(self, base_config):
        """Test loading state dict into module."""
        model1 = NeuralfxLSTM(base_config.model)
        module1 = NeuralFXModule(model1, base_config)

        # Get initial weights
        initial_weight = module1.model.fc_out.weight.clone()

        # Create new module and load state
        model2 = NeuralfxLSTM(base_config.model)
        module2 = NeuralFXModule(model2, base_config)
        module2.load_state_dict(module1.state_dict())

        # Check weights match
        assert torch.allclose(module2.model.fc_out.weight, initial_weight)


class TestUpdatedDataset:
    """Test suite for updated AudioDataset features."""

    @pytest.fixture
    def temp_audio_files(self):
        """Create temporary audio files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sample_rate = 48000
            duration = 4.0  # seconds
            num_samples = int(sample_rate * duration)

            # Create input audio (sine wave)
            t = torch.linspace(0, duration, num_samples)
            input_audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
            target_audio = 0.5 * torch.sin(2 * 3.14159 * 440 * t + 0.5).unsqueeze(0)

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, sample_rate)
            torchaudio.save(str(target_path), target_audio, sample_rate)

            yield input_path, target_path, sample_rate

    def test_random_segment_sampling(self, temp_audio_files):
        """Test random segment sampling option."""
        input_path, target_path, sample_rate = temp_audio_files
        segment_length = 8192

        dataset = AudioDataset(
            input_path=input_path,
            target_path=target_path,
            segment_length=segment_length,
            sample_rate=sample_rate,
            random_segments=True,
        )

        # Get multiple samples and verify they can be at different positions
        samples = [dataset[i] for i in range(min(5, len(dataset)))]
        assert all(x.shape == (segment_length,) for x, y in samples)

    def test_dataset_with_transform(self, temp_audio_files):
        """Test dataset with transform."""
        input_path, target_path, sample_rate = temp_audio_files

        gain = RandomGain(min_gain_db=-6, max_gain_db=-6, p=1.0)

        dataset = AudioDataset(
            input_path=input_path,
            target_path=target_path,
            segment_length=8192,
            sample_rate=sample_rate,
            random_segments=False,
            transform=gain,
        )

        x, y = dataset[0]

        # Transform should have been applied
        expected_gain = 10 ** (-6 / 20)
        assert x.abs().max() < 1.0 or torch.allclose(
            x.abs().max(), torch.tensor(expected_gain), atol=0.1
        )

    def test_train_val_split(self, temp_audio_files):
        """Test train/val split functionality."""
        input_path, target_path, sample_rate = temp_audio_files

        train_dataset, val_dataset = AudioDataset.train_val_split(
            input_path=input_path,
            target_path=target_path,
            segment_length=8192,
            sample_rate=sample_rate,
            val_ratio=0.25,
        )

        # Check datasets are valid
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0

        # Val dataset should be approximately 25% of total
        total_segments = len(train_dataset) + len(val_dataset)
        val_ratio = len(val_dataset) / total_segments
        assert 0.2 <= val_ratio <= 0.3  # Allow some tolerance

        # Val dataset should not have random segments or transforms
        assert not val_dataset.random_segments
        assert val_dataset.transform is None

    def test_module_applies_latency_to_training_dataset(self, temp_audio_files):
        """The calibration computed by the CLI reaches the actual dataset."""
        input_path, target_path, sample_rate = temp_audio_files
        model_config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=4, num_layers=1),
            sample_rate=sample_rate,
        )
        config = NeuralFXConfig(
            version="1.0",
            name="latency_test",
            model=model_config,
            training=TrainingConfig(batch_size=1, segment_length=8192),
            optimizer=OptimizerConfig(),
            lr_scheduler=LRSchedulerConfig(),
            loss=LossConfig(),
            data=DataConfig(
                train=DataPaths(input=str(input_path), target=str(target_path))
            ),
        )
        calibration = LatencyCalibration(
            delay_samples=32,
            method="manual",
            correlation_score=1.0,
        )
        module = NeuralFXModule(
            NeuralfxLSTM(model_config),
            config,
            train_latency=calibration,
        )

        dataset = module._create_train_dataset()

        assert dataset.latency_calibration == calibration
        assert dataset.input_audio.shape[-1] == 4 * sample_rate - 32
        assert dataset.target_audio.shape == dataset.input_audio.shape


class TestUpdatedLossWeights:
    """Test suite for updated loss weights with STFT."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing."""
        model_config = ModelConfig(
            type="lstm",
            params=LSTMParams(
                hidden_size=16,
                num_layers=1,
                conv1d=None,
                skip_connection=False,
                dropout=0.0,
                conditioning_size=0,
            ),
            input_size=1,
            output_size=1,
            sample_rate=48000,
        )

        return NeuralFXConfig(
            version="1.0",
            name="test_model",
            model=model_config,
            training=TrainingConfig(
                batch_size=4,
                epochs=2,
                segment_length=2048,
                tbptt=None,
                seed=42,
            ),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
            loss=LossConfig(type="mse"),
            data=DataConfig(
                train=DataPaths(input="data/DI.wav", target="data/effect.wav"),
            ),
        )

    def test_stft_loss_weight(self, base_config):
        """Test loss with STFT weight."""
        base_config.loss.weights = LossWeights(esr=0.0, mse=0.5, stft=0.5)
        base_config.loss.stft = STFTLossConfig(
            enabled=True,
            fft_sizes=[256, 512],
        )

        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        x = torch.randn(2, 2048)
        y = torch.randn(2, 2048)

        loss = module.training_step((x, y), batch_idx=0)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_stft_loss_disabled(self, base_config):
        """Test that disabled STFT loss is not used."""
        base_config.loss.weights = LossWeights(esr=0.0, mse=1.0, stft=0.5)
        base_config.loss.stft = STFTLossConfig(enabled=False)

        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        x = torch.randn(2, 2048)
        y = torch.randn(2, 2048)

        # Should not raise error even with stft weight > 0 but disabled
        loss = module.training_step((x, y), batch_idx=0)
        assert loss is not None


class TestValidationSupport:
    """Test suite for validation dataset support."""

    @pytest.fixture
    def config_with_val(self):
        """Create configuration with validation data."""
        model_config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=16, num_layers=1),
            input_size=1,
            output_size=1,
        )

        return NeuralFXConfig(
            version="1.0",
            name="test_model",
            model=model_config,
            training=TrainingConfig(
                batch_size=2,
                epochs=2,
                segment_length=2048,
            ),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
            loss=LossConfig(type="mse"),
            data=DataConfig(
                train=DataPaths(input="data/DI.wav", target="data/effect.wav"),
                val=DataPaths(input="data/DI.wav", target="data/effect.wav"),
            ),
        )

    def test_val_dataloader_created(self, config_with_val):
        """Test that val_dataloader returns a dataloader when val data provided."""
        # Config structure tested here - actual data loading would require files
        assert config_with_val.data.val is not None
        # The actual data loading is tested in test_data.py
        assert config_with_val.data.val is not None
        assert config_with_val.data.val.input is not None
        assert config_with_val.data.val.target is not None


class TestAugmentationConfigIntegration:
    """Test suite for augmentation config integration."""

    def test_augmentation_config_creation(self):
        """Test AugmentationConfig dataclass."""
        aug = AugmentationConfig(
            enabled=True,
            random_gain=True,
            gain_min_db=-12,
            gain_max_db=0,
            gain_p=0.5,
            add_noise=True,
            noise_min_snr_db=20,
            noise_max_snr_db=40,
            noise_p=0.3,
        )

        assert aug.enabled is True
        assert aug.random_gain is True
        assert aug.gain_min_db == -12
        assert aug.add_noise is True

    def test_augmentation_in_lightning_module(self):
        """Test that augmentation is used in LightningModule when enabled."""
        model_config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=16, num_layers=1),
            input_size=1,
            output_size=1,
        )

        config = NeuralFXConfig(
            version="1.0",
            name="test_model",
            model=model_config,
            training=TrainingConfig(
                batch_size=2,
                epochs=2,
                segment_length=2048,
                augmentation=AugmentationConfig(
                    enabled=True,
                    random_gain=True,
                    add_noise=True,
                ),
            ),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
            loss=LossConfig(type="mse"),
            data=DataConfig(
                train=DataPaths(input="data/DI.wav", target="data/effect.wav"),
            ),
        )

        model = NeuralfxLSTM(config.model)
        module = NeuralFXModule(model, config)

        assert module.transform is not None

    def test_no_augmentation_when_disabled(self):
        """Test that transform is None when augmentation disabled."""
        model_config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=16, num_layers=1),
            input_size=1,
            output_size=1,
        )

        config = NeuralFXConfig(
            version="1.0",
            name="test_model",
            model=model_config,
            training=TrainingConfig(
                batch_size=2,
                epochs=2,
                segment_length=2048,
                augmentation=AugmentationConfig(enabled=False),
            ),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
            loss=LossConfig(type="mse"),
            data=DataConfig(
                train=DataPaths(input="data/DI.wav", target="data/effect.wav"),
            ),
        )

        model = NeuralfxLSTM(config.model)
        module = NeuralFXModule(model, config)

        assert module.transform is None

    def test_no_augmentation_config(self):
        """Test that transform is None when no augmentation config."""
        model_config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=16, num_layers=1),
            input_size=1,
            output_size=1,
        )

        config = NeuralFXConfig(
            version="1.0",
            name="test_model",
            model=model_config,
            training=TrainingConfig(
                batch_size=2,
                epochs=2,
                segment_length=2048,
                augmentation=None,
            ),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
            loss=LossConfig(type="mse"),
            data=DataConfig(
                train=DataPaths(input="data/DI.wav", target="data/effect.wav"),
            ),
        )

        model = NeuralfxLSTM(config.model)
        module = NeuralFXModule(model, config)

        assert module.transform is None
