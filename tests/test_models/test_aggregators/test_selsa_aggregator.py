# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models.aggregators import SelsaAggregator


def test_selsa_aggregator():
    """Test selsa_aggregator."""
    # Test embed_aggregator forward
    model = SelsaAggregator(in_channels=16, num_attention_blocks=4)
    model.train()

    target_x = torch.randn(2, 16)
    ref_x = torch.randn(4, 16)
    agg_x = model(target_x, ref_x)
    assert agg_x.shape == target_x.shape
