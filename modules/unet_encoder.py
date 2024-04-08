import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import OrderedDict

from modules.resnet_block import ResnetBasicBlock


class UnetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        block = ResnetBasicBlock

        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels * block.expansion,
                                   kernel_size=(1, 1), stride=(stride, stride), padding=0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels * block.expansion)),
            ]))
        self.res_block = block(in_channels, out_channels, stride, downsample)
        self.maxpool = nn.MaxPool2d(
            kernel_size=(2, 2), stride=2)

    def forward(self, x):
        out = self.res_block(x)
        out = self.maxpool(out)
        return out


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, channels_chain):
        """
            channels_chain: [in, ..., out]
        """
        super().__init__()
        self.module = self._make_layers(in_channels, channels_chain)

    def _make_layers(self, in_channels, channels_chain):
        assert len(channels_chain) >= 1
        layers = [
            UnetDownBlock(in_channels, channels_chain[0], stride=2, padding=2)
        ]
        in_channels = channels_chain[0]
        for out_channels in channels_chain[1:]:
            layers.append(UnetDownBlock(in_channels, out_channels, stride=1))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.module(x)
        return out
