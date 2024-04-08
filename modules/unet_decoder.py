import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import OrderedDict


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int):
        super().__init__()

        self.conv1 = nn.Sequential(OrderedDict([
            ('t_conv', nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
             kernel_size=3, stride=stride, padding=1, output_padding=1)),
            ('bn', nn.BatchNorm2d(out_channels))
        ]))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        return out


class UnetDecoder(nn.Module):
    def __init__(self, channels_chain, out_classes):
        super().__init__()
        self.seq = self._make_layers(channels_chain)

        self.final_conv = nn.Conv2d(
            channels_chain[-1], out_classes, kernel_size=1, stride=1)

    def _make_layers(self, channels_chain):
        layers = []
        in_channels = channels_chain[0]
        for out_channels in channels_chain[1:]:
            layers.append(
                UpsampleBlock(in_channels, out_channels, stride=2)
            )
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.seq(x)
        out = self.final_conv(out)
        return out
