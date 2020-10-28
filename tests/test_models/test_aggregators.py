import pytest
import torch

from mmtrack.models.aggregators import StackedEmbedConvs


def test_stacked_embed_convs():
    """Test flownet_simple."""
    with pytest.raises(AssertionError):
        # The number of convs must be bigger than 1.
        model = StackedEmbedConvs(num_convs=0, channels=32, kernel_size=3)

    with pytest.raises(AssertionError):
        # Only support 'batch_size == 1' for target_x
        model = StackedEmbedConvs(num_convs=3, channels=32, kernel_size=3)
        model.train()

        target_x = torch.randn(2, 32, 224, 224)
        ref_x = torch.randn(4, 32, 224, 224)
        agg_x = model(target_x, ref_x)

    # Test stacked_embed_convs forward
    model = StackedEmbedConvs(num_convs=3, channels=32, kernel_size=3)
    model.train()

    target_x = torch.randn(1, 32, 224, 224)
    ref_x = torch.randn(4, 32, 224, 224)
    agg_x = model(target_x, ref_x)
    assert agg_x.shape == target_x.shape
