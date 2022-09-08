# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmtrack.models import KLGridLoss, KLMCLoss


class TestKLMCLoss(TestCase):

    def test_kl_mc_loss(self):
        loss = KLMCLoss(eps=1e-9)
        scores = torch.Tensor([[0.1, 0.3], [0.1, 0.3]])
        sample_density = torch.Tensor([[0.001, 0.001], [0.001, 0.001]])
        gt_density = torch.Tensor([[0.001, 0.001], [0.001, 0.001]])
        assert torch.allclose(
            loss(scores, sample_density, gt_density), torch.tensor(6.9127))


class TestKLGridLoss(TestCase):

    def test_kl_grid_loss(self):
        loss = KLGridLoss()
        scores = torch.Tensor([[0.1, 0.3], [0.1, 0.3]])
        gt_density = torch.Tensor([[0.001, 0.001], [0.001, 0.001]])
        assert torch.allclose(
            loss(scores, gt_density), torch.tensor(0.89773887))
