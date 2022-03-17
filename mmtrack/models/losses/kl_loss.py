# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmdet.models import LOSSES


@LOSSES.register_module()
class KLMCLoss(nn.Module):
    """KL-divergence loss for probabilistic regression.

    It is computed using Monte Carlo (MC) samples from an arbitrary
    distribution.
    """

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density, mc_dim=-1):
        """Args:
            scores: predicted score values
            sample_density: probability density of the sample distribution
            gt_density: probability density of the ground truth distribution
            mc_dim: dimension of the MC samples"""

        exp_val = scores - torch.log(sample_density + self.eps)

        L = torch.logsumexp(
            exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim]) - torch.mean(
                scores * (gt_density / (sample_density + self.eps)),
                dim=mc_dim)

        return L.mean()


@LOSSES.register_module()
class KLGridLoss(nn.Module):
    """KL-divergence loss for probabilistic regression.

    It is computed using the grid integration strategy.
    """

    def forward(self, scores, gt_density, grid_dim=-1, grid_scale=1.0):
        """Args:
            scores: predicted score values
            gt_density: probability density of the ground truth distribution
            grid_dim: dimension(s) of the grid
            grid_scale: area of one grid cell"""

        score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)

        L = torch.logsumexp(
            scores, dim=grid_dim) + math.log(grid_scale) - score_corr

        return L.mean()
