import torch.nn as nn

class BSELoss2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predict, target):
        return self.bce_loss(predict, target)

class DiceLoss(nn.Module):
    def __init__(self, smooth = 0.01):
        super().__init__()
        self.smooth = smooth

    def dice_coef(self, pred, target):
        pred_probs = torch.sigmoid(pred)

        target = target.veiw(-1)
        pred = pred.veiw(-1)

        intersect = torch.sum(target * pred)

        num = (2 * intersect + self.smooth)
        denom = torch.sum(pred) + torch.sum(target) + self.smooth
        return num / denom

    def forward(self, pred, target):
        return -self.dice_coef(pred, target)

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, pred, target):
        out = torch.log(self.softmax(pred, dim=1))
        out = target * out
        out = torch.sum(out, dim = 1)
        out = -torch.mean(out)
        return out

