import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import OrderedDict

from modules.resnet_block import ResnetBasicBlock

class UnetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        block = ResnetBasicBlock

        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels * block.expansion, 
                          (1, 1), (stride, stride), (0, 0), bias=False)),
                ('bn', nn.BatchNorm2d(out_channels * block.expansion)),
            ]))
        self.res_block = block(in_channels, out_channels, stride, downsample)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=1)

    def forward(self, x):
        out = self.res_block(x)
        out = self.maxpool(out)
        return out

class UnetEncoder(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.modules = self._make_layers(num_channels,
                                         [64, 128, 256, 512, 1024])

    def _make_layers(self, in_channels, out_channels_chain : list):
        assert len(out_channels_chain)>0
        layers = [
            UnetDownBlock(in_channels, out_channels_chain[0], 1)
        ]
        in_channels = out_channels_chain[0]
        for out_channels in out_channels_chain[1:]:
            layers.append(UnetDownBlock(in_channels, out_channels, 1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.modules(x)
        return out
