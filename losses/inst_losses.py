import torch
import torch.nn as nn
import numpy as np
import utils.tools as tools

class LossKeeper(nn.Module):
    """
        У кажого объекта свой лосс 
        Который зависит от расположения объектов на таргете
        Данный класс хранит эти лоссы
        И применяет нужный в зависимости от индекса
    """
    def __init__(self, insts, num_colors: int, halo_coef: float, halo_margin: int,
                 device = 'cpu'):
        """
            insts: list of insts forall objects
            [obj_num, K, mask] (K для каждого объекта свое)
            all tensors: [obj_num, K, W, H], where [W, H] - mask
        """
        self.device = device
        super().__init__()

        def mask_size(mask):
            return int(torch.sum(mask))

        def get_obj_sizes(obj_masks):
            return [
                [mask_size(mask) for mask in masks]
                for masks in obj_masks
            ]

        self.obj_masks = insts
        self.obj_sizes = get_obj_sizes(insts)

        self.obj_halo_masks = [
            [tools.dilate(mask, halo_margin) for mask in masks]
            for masks in insts
        ]
        self.obj_halo_sizes = get_obj_sizes(
            self.obj_halo_masks)

        assert len(self.obj_halo_masks) == len(self.obj_masks)

        self.num_colors = num_colors
        self.mu = halo_coef

    def forward(self, y, obj_idx):
        """
            y: [C, W, H]
            0 color - background mask
            return loss (num)
        """
        inst_colors = self.get_colors(y, obj_idx)  # leaf -> color [K, ]

        result = 0
        for k, color in zip(range(len(self.obj_sizes[obj_idx])), inst_colors):
            result += self.get_avg_logprob(y, obj_idx, color, k)
        result -= self.get_avg_logprob(y, obj_idx, 0, 0)
        return -result

    def get_avg_logprob(self, y, obj_idx, color, k):
        """
            return 1/|Mk|*sum(log y[c, p]) for p in mask
            [,]
        """
        cnt = self.obj_sizes[obj_idx][k]
        return (1/cnt)*self.masked_sum_log(y[color],
                                           self.obj_masks[obj_idx][k])

    def masked_sum_log(self, y, mask):
        """
            return masked sum of logs
            sum log(y[p]) for p in mask
        """
        masked_y = y*mask
        zero_equal_mask = (masked_y==0).float()
        return torch.sum(torch.log(y*mask + zero_equal_mask + 1e-12))

    def get_colors(self, y, obj_idx):
        """
            return argmax(avg_logprob) [K, ]
        """
        K = len(self.obj_sizes[obj_idx])
        A = torch.Tensor([ # [C, K]
            self.get_color_avg_logprob(y, color, obj_idx)
            for color in range(self.num_colors)
        ])
        return torch.argmax(A.T, axis = 1)

    def get_color_avg_logprob(self, y, color, obj_idx):
        """
            y: [C, W, H]
            color in [0, C-1]
            obj_idx in [0, obj_num - 1]

            return: average logprob foreach pixel in inst area [K, ]
        """
        sizes = self.obj_sizes[obj_idx]  # [K, W, H]
        masks = self.obj_masks[obj_idx]
        halo_sizes = self.obj_halo_sizes[obj_idx]
        halo_masks = self.obj_halo_masks[obj_idx]

        classes_logprob = []
        # while k <= K
        for size, mask, halo_size, halo_mask, _, in zip(sizes,  # [W, H]
                                                        masks,
                                                        halo_sizes,
                                                        halo_masks,
                                                        range(1, len(sizes))):

            obj_part = (1/size)*self.masked_sum_log(y[color], mask)
            halo_part = self.mu*(1/halo_size) * self.masked_sum_log(1 - y[color], halo_mask)
            classes_logprob.append(obj_part + halo_part)
        return classes_logprob
