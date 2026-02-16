import torch.nn as nn
from typing import Optional

RNN_CONFIG_PATH = "./config/models/rnn/"

# TODO: add conv1d layers


class RNNConfig:
    def __init__(self, config: dict, **kwargs):
        self.conv1d_strides = config["conv1d_strides"]
        self.conv1d_filters = config["conv1d_filters"]
        self.blocks = [
            RNNBlockConfig(layer_config) for layer_config in config["blocks"]
        ]
        for key, value in kwargs.items():
            setattr(self, key, value)


class RNNBlockConfig:
    def __init__(self, config: dict, **kwargs):
        self.block_type = config["block_type"]
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.hidden_units = config["hidden_units"]
        self.num_layers = config["num_layers"]
        self.lin_bias = config["lin_bias"]
        self.skip = config["skip"]
        for key, value in kwargs.items():
            setattr(self, key, value)


class NeuralFXRNN(nn.Module):
    def __init__(
        self,
        *args,
        config: RNNConfig,
        train_burn_in: Optional[int] = None,
        train_truncate: Optional[int] = None,
        skip: int = 0,
        **kwargs,
    ):
        super(NeuralFXRNN, self).__init__()

        # Create container for layers
        self.layers = nn.Sequential()
        # Create dictionary of possible block types
        self.block_types = {}
        self.block_types.update(dict.fromkeys(["LSTM", "GRU"], RNNBlock))

        self.conv1d_strides = config.conv1d_strides
        self.conv1d_filters = config.conv1d_filters
        self.train_burn_in = train_burn_in
        self.train_truncate = train_truncate

        self.skip = skip
        self.save_state = False
        self.input_size = None
        self.training_info = {
            "current_epoch": 0,
            "training_losses": [],
            "validation_losses": [],
            "train_epoch_av": 0.0,
            "val_epoch_av": 0.0,
            "total_time": 0.0,
            "best_val_loss": 1e12,
        }
        # If layers were specified, create layers
        for layer_config in config.blocks:
            self.add_layer(layer_config)

    # Define forward pass
    def forward(self, x):
        if not self.skip:
            return self.layers(x)
        else:
            res = x[:, :, 0 : self.skip]
            return self.layers(x) + res

    # Set hidden state to specified values, resets gradient tracking
    def detach_hidden(self):
        for each in self.layers:
            each.detach_hidden()

    def reset_hidden(self):
        for each in self.layers:
            each.reset_hidden()

    # Add layer to the network, params is a dictionary contains the layer keyword arguments
    def add_layer(self, params):
        # If this is the first layer, define the network input size
        if self.input_size:
            pass
        else:
            self.input_size = params["input_size"]

        self.layers.add_module(
            "block_" + str(1 + len(list(self.layers.children()))),
            self.block_types[params["block_type"]](params),
        )
        self.output_size = params["output_size"]

    def save_model(self, file_name, direc=""):
        if direc:
            pass
            # TODO use os dircheck
            # miscfuncs.dir_check(direc)

        model_data = {"model_data": {"model": "RecNet", "skip": 0}, "blocks": {}}
        for i, each in enumerate(self.layers):
            model_data["blocks"][str(i)] = each.params

        if self.training_info:
            model_data["training_info"] = self.training_info

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].tolist()
            model_data["state_dict"] = model_state
        # TODO: use json.save
        # miscfuncs.json_save(model_data, file_name, direc)


class RNNBlock(nn.Module):
    def __init__(self, params: RNNBlockConfig):
        super(RNNBlock, self).__init__()
        # TODO: see if check is needed, probably not
        # assert isinstance(
        #     params['input_size'], int), "an input_size of int type must be provided in 'params'"
        # assert isinstance(
        #     params['output_size'], int), "an output_size of int type must be provided in 'params'"
        # assert isinstance(
        #     params['hidden_size'], int), "an hidden_size of int type must be provided in 'params'"

        self.params = params
        # This just calls nn.LSTM() if 'block_type' is LSTM, nn.GRU() if GRU, etc
        self.rec = getattr(nn, params.block_type)(
            input_size=params.input_size,
            hidden_units=params.hidden_units,
            num_layers=params.num_layers,
            batch_first=True,
        )
        self.lin_bias = params.lin_bias
        self.lin = nn.Linear(params.hidden_units, params.output_size, self.lin_bias)
        self.hidden_units = None
        self.skip = params.skip

    def forward(self, x):
        if self.skip:
            # save the residual for the skip connection
            res = x[:, :, 0 : self.skip]
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x) + res
        else:
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if isinstance(self.hidden, tuple):
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    def reset_hidden(self):
        self.hidden = None
