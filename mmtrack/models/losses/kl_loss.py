# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmtrack.registry import MODELS


@MODELS.register_module()
class KLMCLoss(nn.Module):
    """KL-divergence loss for probabilistic regression.

    It is computed using Monte Carlo (MC) samples from an arbitrary
    distribution.

    Args:
        eps (float, optional): Defaults to 0.0.
    """

    def __init__(self, eps: float = 0.0):
        super().__init__()
        self.eps = eps

    def forward(self,
                scores: Tensor,
                sample_density: Tensor,
                gt_density: Tensor,
                mc_dim: int = -1) -> Tensor:
        """
        Args:
            scores (Tensor): predicted score values. It has shape
                (num_imgs, num_samples).
            sample_density (Tensor): probability density of the sample
                distribution. It has shape (num_imgs, num_samples).
            gt_density (Tensor): probability density of the ground truth
                distribution. It has shape (num_imgs, num_samples).
            mc_dim (int): dimension of the MC samples.

        Returns:
            torch.Tensor: Calculated loss
        """
        exp_val = scores - torch.log(sample_density + self.eps)

        loss = torch.logsumexp(
            exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim]) - torch.mean(
                scores * (gt_density / (sample_density + self.eps)),
                dim=mc_dim)

        return loss.mean()


@MODELS.register_module()
class KLGridLoss(nn.Module):
    """KL-divergence loss for probabilistic regression.

    It is computed using the grid integration strategy.
    """

    def forward(self,
                scores: Tensor,
                gt_density: Tensor,
                grid_dim: Union[Tuple, int] = -1,
                grid_scale: float = 1.0) -> Tensor:
        """
        Args:
            scores (Tensor): predicted score values. It has shape
                (num_imgs_per_seq, bs, score_map_size, score_map_size).
            gt_density (Tensor): probability density of the ground truth
                distribution. It has shape
                (num_imgs_per_seq, bs, score_map_size, score_map_size).
            grid_dim (int): dimension(s) of the grid.
            grid_scale (float): area of one grid cell.

        Returns:
            torch.Tensor: Calculated loss
        """
        score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)

        loss = torch.logsumexp(
            scores, dim=grid_dim) + math.log(grid_scale) - score_corr

        return loss.mean()
