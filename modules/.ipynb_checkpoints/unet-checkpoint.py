import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.unet_encoder import UnetEncoder
from modules.unet_decoder import UnetDecoder

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UnetEncoder(3)
        self.decoder = UnetDecoder(1024, 3)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
