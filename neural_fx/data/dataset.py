from pathlib import Path
import warnings

from torch import Tensor
from torch.utils.data import Dataset
import torchaudio


class AudioDataset(Dataset):
    """Dataset for input/target audio file pairs."""

    def __init__(
        self,
        input_path: str | Path,
        target_path: str | Path,
        segment_length: int = 8192,
        sample_rate: int = 48000,
        normalize: bool = True,
    ):
        super().__init__()
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.normalize = normalize

        input_path = Path(input_path)
        target_path = Path(target_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")

        self.input_audio = self._load_audio(input_path)
        self.target_audio = self._load_audio(target_path)

        if self.input_audio.shape[-1] != self.target_audio.shape[-1]:
            min_len = min(self.input_audio.shape[-1], self.target_audio.shape[-1])
            warnings.warn(
                f"Input and target audio lengths differ "
                f"({self.input_audio.shape[-1]} vs {self.target_audio.shape[-1]}). "
                f"Truncating to {min_len} samples."
            )
            self.input_audio = self.input_audio[..., :min_len]
            self.target_audio = self.target_audio[..., :min_len]

        self.num_segments = self.input_audio.shape[-1] // segment_length
        self.total_length = self.num_segments * segment_length

    def _load_audio(self, path: Path) -> Tensor:
        audio, sr = torchaudio.load(str(path))

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if self.normalize:
            max_val = audio.abs().max()
            if max_val > 0:
                audio = audio / max_val

        return audio

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.segment_length
        end = start + self.segment_length

        x = self.input_audio[..., start:end]
        y = self.target_audio[..., start:end]

        return x.squeeze(0), y.squeeze(0)
