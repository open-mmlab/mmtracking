# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmtrack.models.task_modules import SiameseRPNAnchorGenerator


class TestSiameseRPNAnchorGenerator(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.anchor_generator = SiameseRPNAnchorGenerator(
            strides=[(8, 8)], ratios=[0.33, 0.5, 1, 2, 3], scales=[8])

    def test_gen_2d_hanning_windows(self):
        multi_level_windows = self.anchor_generator.gen_2d_hanning_windows(
            [(4, 4)], device='cpu')
        assert len(multi_level_windows) == 1
        assert len(multi_level_windows[0]) == 4 * 4 * 5

    def test_gen_single_level_base_anchors(self):
        base_anchors = self.anchor_generator.gen_single_level_base_anchors(
            base_size=6,
            scales=torch.Tensor([8]),
            ratios=torch.Tensor([0.33, 0.5, 1, 2, 3]),
            center=[2, 2])
        assert base_anchors.shape == torch.Size([5, 4])
