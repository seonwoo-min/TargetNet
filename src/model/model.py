# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - TorchVision

import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class TargetNet(nn.Module):
    """ TargetNet for microRNA target prediction """
    def __init__(self, model_cfg, with_esa, dropout_rate):
        super(TargetNet, self).__init__()
        num_channels = model_cfg.num_channels
        num_blocks = model_cfg.num_blocks

        if not with_esa: self.in_channels, in_length = 8, 40
        else:            self.in_channels, in_length = 10, 50
        out_length = np.floor(((in_length - model_cfg.pool_size) / model_cfg.pool_size) + 1)

        self.stem = self._make_layer(model_cfg, num_channels[0], num_blocks[0], dropout_rate, stem=True)
        self.stage1 = self._make_layer(model_cfg, num_channels[1], num_blocks[1], dropout_rate)
        self.stage2 = self._make_layer(model_cfg, num_channels[2], num_blocks[2], dropout_rate)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.max_pool = nn.MaxPool1d(model_cfg.pool_size)
        self.linear = nn.Linear(int(num_channels[-1] * out_length), 1)

    def _make_layer(self, cfg, out_channels, num_blocks, dropout_rate, stem=False):
        layers = []
        for b in range(num_blocks):
            if stem: layers.append(Conv_Layer(self.in_channels, out_channels, cfg.stem_kernel_size, dropout_rate,
                                              post_activation= b < num_blocks - 1))
            else:    layers.append(ResNet_Block(self.in_channels, out_channels, cfg.block_kernel_size, dropout_rate,
                                                skip_connection=cfg.skip_connection))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dropout(self.relu(x))
        x = self.max_pool(x)
        x = x.reshape(len(x), -1)
        x = self.linear(x)

        return x


def conv_kx1(in_channels, out_channels, kernel_size, stride=1):
    """ kx1 convolution with padding without bias """
    layers = []
    padding = kernel_size - 1
    padding_left = padding // 2
    padding_right = padding - padding_left
    layers.append(nn.ConstantPad1d((padding_left, padding_right), 0))
    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False))
    return nn.Sequential(*layers)


class Conv_Layer(nn.Module):
    """
    CNN layer with/without activation
    -- Conv_kx1_ReLU-Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, post_activation):
        super(Conv_Layer, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.conv = conv_kx1(in_channels, out_channels, kernel_size)
        self.post_activation = post_activation

    def forward(self, x):
        out = self.conv(x)
        if self.post_activation:
            out = self.dropout(self.relu(out))

        return out


class ResNet_Block(nn.Module):
    """
    ResNet Block
    -- ReLU-Dropout-Conv_kx1 - ReLU-Dropout-Conv_kx1
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, skip_connection):
        super(ResNet_Block, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.conv1 = conv_kx1(in_channels, out_channels, kernel_size)
        self.conv2 = conv_kx1(out_channels, out_channels, kernel_size)
        self.skip_connection = skip_connection

    def forward(self, x):
        out = self.dropout(self.relu(x))
        out = self.conv1(out)

        out = self.dropout(self.relu(out))
        out = self.conv2(out)

        if self.skip_connection:
            out_c, x_c = out.shape[1], x.shape[1]
            if out_c == x_c: out += x
            else:            out += F.pad(x, (0, 0, 0, out_c - x_c))

        return out

