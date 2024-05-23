"""
Layer that applies a mask to the input tensor. The mask is made of 0s and 1s,
which sets invalid moves to 0.
"""

import torch
from torch import nn


class MaskLayer(nn.Module):
    """
    Layer that applies a mask to the input tensor. The mask is made of 0s and 1s,
    which sets invalid moves to 0.

    Methods:
        forward(x, mask):
            Applies the mask to the input tensor, setting invalid moves to 0.
    """

    def __init__(self):
        """
        Initializes the MaskLayer.
        """
        super(MaskLayer, self).__init__()

    def forward(self, x, mask):
        """
        Applies the mask to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask tensor with 0s and 1s.

        Returns:
            torch.Tensor: The masked tensor.
        """
        return torch.mul(x, mask)
