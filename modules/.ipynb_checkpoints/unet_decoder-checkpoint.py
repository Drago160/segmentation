import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import OrderedDict

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(OrderedDict([
            ('t_conv', nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size=4, stride=2, padding=1, output_padding=0)),
            ('bn', nn.BatchNorm2d(out_channels))
        ]))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        return out

class UnetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.modules = self._make_layers(
            in_channels, [256, 128, 64, 32, 16]
        )

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def _make_layers(self, in_channels, out_channels_chain):
        layers = []
        for out_channels in out_channels_chain:
            layers.append(
                UpsampleBlock(in_channels, out_channels)
            )
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.modules(x)
        out = self.final_conv(out)
        return out
