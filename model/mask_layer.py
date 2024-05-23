import torch
import torch.nn as nn


class MaskLayer(nn.Module):

    def __init__(self):
        super(MaskLayer, self).__init__()

    # mask is made of 0s/1s so it will just set to 0 any invalid move
    def forward(self, x, mask):
        return torch.mul(x, mask)
