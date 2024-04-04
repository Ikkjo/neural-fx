import audio
from wavenet import WaveNet, get_dilation_array
import torch
from wavenet_utils import get_arguments, batch_data
import sys


TRAIN_ARGS, MODEL_ARGS = get_arguments(sys.argv)
NUM_LAYERS, NUM_CHANNELS, KERNEL_SIZE, DILATION_PARAMETER = (
    MODEL_ARGS["num_layers"],
    MODEL_ARGS["num_channels"],
    MODEL_ARGS["kernel_size"],
    MODEL_ARGS["dilation_parameter"],
)


print(
    f"Using hyperparameters: \nNumber of Layers: {NUM_LAYERS} \nNumber of Channels: {NUM_CHANNELS} \nKernel Size: {KERNEL_SIZE} \nDilation Parameter: {DILATION_PARAMETER}"
)


print("Creating model...")
dilation_array = get_dilation_array(NUM_LAYERS, DILATION_PARAMETER)
model = WaveNet(NUM_LAYERS, NUM_CHANNELS, KERNEL_SIZE, dilation_array)

print("Loading data...")
data = audio.load_dataset(return_tensors=True)
X = data["DI"].data
Y = data["ds1_gain_100"].data
X, Y = batch_data(
    data["DI"].data, data[TRAIN_ARGS["target_data"]].data, model.receptive_field
)
