import json
import random
import torch


def import_model_config(path):
    with open(path, "r") as file:
        model_config = json.load(file)
    return model_config


def get_arguments(argv, default_config_path="neural_fx/wavenet_train_config.json"):
    default_parameters = import_model_config(default_config_path)
    arguments = {}
    for argument in argv[1:]:
        if "=" not in argument:
            continuen
        key, value = argument.split("=")
        arguments[key] = value

    model_args_list = [
        "num_layers",
        "num_channels",
        "kernel_size",
        "dilation_parameter",
    ]
    train_args_list = [
        "target_data",
        "batch_size",
        "num_epochs",
        "learning_rate",
        "optimizer",
    ]
    model_args = {}
    train_args = {}
    for arg in model_args_list:
        if arg not in arguments:
            model_args[arg] = default_parameters["model"][arg]
        else:
            model_args[arg] = arguments[arg]
    for arg in train_args_list:
        if arg not in arguments:
            train_args[arg] = default_parameters["training_arguments"][arg]
        else:
            train_args[arg] = arguments[arg]

    return train_args, model_args


def batch_data(X, Y, receptive_field):
    dataset_len = X.shape[0]
    num_patches = dataset_len // receptive_field
    X_batched = torch.zeros(num_patches, 1, receptive_field)
    Y_batched = torch.zeros(num_patches, 1, receptive_field)
    for i in random.sample(range(num_patches), num_patches):
        X_batched[i, 0] = X[i * receptive_field : (i + 1) * receptive_field]
        Y_batched[i, 0] = Y[i * receptive_field : (i + 1) * receptive_field]

    return X_batched, Y_batched
