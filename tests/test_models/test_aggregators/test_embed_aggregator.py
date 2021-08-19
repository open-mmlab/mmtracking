# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmtrack.models.aggregators import EmbedAggregator


def test_embed_aggregator():
    """Test embed_aggregator."""
    with pytest.raises(AssertionError):
        # The number of convs must be bigger than 1.
        model = EmbedAggregator(num_convs=0, channels=32, kernel_size=3)

    with pytest.raises(AssertionError):
        # Only support 'batch_size == 1' for target_x
        model = EmbedAggregator(num_convs=3, channels=32, kernel_size=3)
        model.train()

        target_x = torch.randn(2, 32, 224, 224)
        ref_x = torch.randn(4, 32, 224, 224)
        agg_x = model(target_x, ref_x)

    # Test embed_aggregator forward
    model = EmbedAggregator(num_convs=3, channels=32, kernel_size=3)
    model.train()

    target_x = torch.randn(1, 32, 224, 224)
    ref_x = torch.randn(4, 32, 224, 224)
    agg_x = model(target_x, ref_x)
    assert agg_x.shape == target_x.shape
