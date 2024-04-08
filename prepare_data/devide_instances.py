import torch
import numpy as np

def devide(insts):
    unique = np.unique(insts)
    masks = []
    for obj_col in unique:
        masks.append(insts.squeeze(0) == obj_col)
    return torch.stack(masks)
