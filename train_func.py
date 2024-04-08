import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.unet import Unet

from losses.inst_losses import LossKeeper

from tqdm import tqdm

def train(model, optimizer,
          train_loader, val_loader, 
          train_loss_keeper, val_loss_keeper,
          epoch_num:int):

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epoch_num)):
        model.train()
        for img_b, _, idx in train_loader:
            optimizer.zero_grad()

            idx = int(idx)
            output = model(img_b)[0]
            loss = train_loss_keeper.forward(output, idx)
    
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        for img_b, _, idx in val_loader:
            optimizer.zero_grad()

            idx = int(idx)
            output = model(img_b)[0]
            with torch.no_grad():
                loss = val_loss_keeper.forward(output, idx)

            val_losses.append(loss.item())

    return model, train_losses, val_losses
