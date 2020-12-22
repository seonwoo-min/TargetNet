# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import torch.nn as nn

from src.model.model import TargetNet


def get_model(model_cfg, with_esa, dropout_rate=None):
    """ get model considering model types """
    model = TargetNet(model_cfg, with_esa, dropout_rate)
    params = get_params_and_initialize(model)

    return model, params


def get_params_and_initialize(model):
    """
    parameter initialization
    get weights and biases for different weighty decay during training
    """
    params_with_decay, params_without_decay = [], []
    for name, param in model.named_parameters():
        if "weight" in name:
            if "bn" not in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
                params_with_decay.append(param)
            else:
                nn.init.ones_(param)
                params_without_decay.append(param)

        else:
            nn.init.zeros_(param)
            params_without_decay.append(param)

    return params_with_decay, params_without_decay

