from torch.utils.data import Dataset, DataLoader
import torch

class CVPPPDataset(Dataset):
    def __init__(self, imgs, targets):
        self.imgs = imgs
        self.targets = targets
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        return self.imgs[index], self.targets[index], index
