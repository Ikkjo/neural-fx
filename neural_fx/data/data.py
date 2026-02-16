from scipy.io import wavfile
from torchaudio.functional import resample as torchaudio_resample
import numpy as np
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


DATASET_DIR_INT = "neural_fx_dataset/"
DATASET_DIR_FLOAT = "neural_fx_dataset_float/"
DEFAULT_DATASET_DIR = DATASET_DIR_FLOAT


class Waveform:
    data = None
    sample_rate = None

    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate

    def __str__(self):
        return f"Waveform of size {len(self.data)} at sample rate {self.sample_rate}"

    def plot(self):
        plt.plot(self.data)
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.show()

    def resample(self, new_sample_rate):
        if self.__infer_dtype() == "int":
            print("Resampling not supported for integer data")
            return

        if self.__is_numpy():
            self.data = torchaudio_resample(
                torch.tensor(self.data), self.sample_rate, new_sample_rate
            ).numpy()
        else:
            self.data = torchaudio_resample(
                self.data, self.sample_rate, new_sample_rate
            )
        self.sample_rate = new_sample_rate

    def __is_numpy(self):
        return isinstance(self.data, np.ndarray)

    def __infer_dtype(self):
        return (
            "float"
            if self.data.dtype == np.float32 or self.data.dtype == torch.float32
            else "int"
        )


def read_wav(file_path, return_tensor=False):
    sample_rate, data = wavfile.read(file_path)
    if return_tensor:
        data = torch.tensor(data)
    return Waveform(data, sample_rate)


def write_wav(waveform, filename):
    if isinstance(waveform.data, torch.Tensor):
        waveform.data = waveform.data.numpy()
    wavfile.write(filename, waveform.sample_rate, waveform.data)


def load_dataset(path=DEFAULT_DATASET_DIR, return_tensors=False):
    dataset = {}
    for file in tqdm(os.listdir(path)):
        if file.endswith(".wav"):
            filename = file.split(".")[0]
            dataset[filename] = read_wav(os.path.join(path, file))
            if return_tensors:
                dataset[filename].data = torch.tensor(dataset[filename].data)

    return dataset


if __name__ == "__main__":
    dataset = load_dataset()
    waveform = read_wav("neural_fx_dataset_float/DI.wav", return_tensor=True)
    print(waveform)
    waveform.resample(16000)
    print(waveform)
