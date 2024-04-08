import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.unet_encoder import UnetEncoder
from modules.unet_decoder import UnetDecoder

class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, [64, 128, 512, 1024])
        self.decoder = UnetDecoder([1024, 512, 256, 128, 64, 32], num_classes)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = F.softmax(out, dim=1)
        return out
