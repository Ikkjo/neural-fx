import json


def import_model_config(path):
    with open(path, "r") as file:
        model_config = json.load(file)
    return model_config


def get_hyperparameters_from_agruments(
    argv, default_config_path="neural_fx/model_config.json"
):
    default_parameters = import_model_config(default_config_path)
    arguments = {}
    for argument in argv[1:]:
        if "=" not in argument:
            continuen
        key, value = argument.split("=")
        arguments[key] = value
    num_layers = (
        int(default_parameters["num_layers"])
        if "num_layers" not in arguments
        else int(arguments["num_layers"])
    )
    num_channels = (
        int(default_parameters["num_channels"])
        if "num_channels" not in arguments
        else int(arguments["num_channels"])
    )
    kernel_size = (
        int(default_parameters["kernel_size"])
        if "kernel_size" not in arguments
        else int(arguments["kernel_size"])
    )
    dilation_parameter = (
        int(default_parameters["dilation_parameter"])
        if "dilation_parameter" not in arguments
        else int(arguments["dilation_parameter"])
    )
    return num_layers, num_channels, kernel_size, dilation_parameter
