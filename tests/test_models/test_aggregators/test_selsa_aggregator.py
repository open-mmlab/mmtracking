# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmtrack.models.aggregators import SelsaAggregator


class TestSelsaAggregator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = SelsaAggregator(in_channels=16, num_attention_blocks=4)
        cls.model.train()

    def test_forward(self):
        # Test embed_aggregator forward
        target_x = torch.randn(2, 16)
        ref_x = torch.randn(4, 16)
        agg_x = self.model(target_x, ref_x)
        assert agg_x.shape == target_x.shape
