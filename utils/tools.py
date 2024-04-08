import torch
import cv2 as cv
import numpy as np

def insert_to_zero_dim(tens, other):
    """
        tens -> [other, tens]
    """
    return torch.cat((tens, other[None,]))

def dilate(mask, padding:int):
    mask = np.array(mask.float().detach())
    kernel = np.ones((padding, padding))
    dilated = cv.dilate(mask, kernel)
    return torch.Tensor(dilated - mask)
    