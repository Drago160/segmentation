import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import OrderedDict

class ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample = None):
        super().__init__()
        self.stride = stride
        self.downsample = downsample
        self.conv_block1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                               stride=stride, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu:', nn.ReLU())
        ]))
        self.conv_block2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu:', nn.ReLU())
        ]))
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = torch.add(residual, out)
        out = F.relu(out)
        return out
