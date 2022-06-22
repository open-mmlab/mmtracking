# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmtrack.models import EmbedAggregator


class TestEmbedAggregator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = EmbedAggregator(num_convs=3, channels=32, kernel_size=3)
        cls.model.train()

    def test_forward(self):
        with self.assertRaises(AssertionError):
            # Only support 'batch_size == 1' for target_x
            target_x = torch.randn(2, 32, 224, 224)
            ref_x = torch.randn(4, 32, 224, 224)
            agg_x = self.model(target_x, ref_x)

        # Test embed_aggregator forward
        target_x = torch.randn(1, 32, 224, 224)
        ref_x = torch.randn(4, 32, 224, 224)
        agg_x = self.model(target_x, ref_x)
        assert agg_x.shape == target_x.shape
