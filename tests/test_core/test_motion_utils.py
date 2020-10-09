import pytest
import torch

from mmtrack.core import flow_warp_feats


def test_flow_warp_feats():
    flow = torch.randn(2, 2, 100, 100)
    ref_x = torch.randn(2, 128, 256, 256)
    x = flow_warp_feats(flow, ref_x)
    assert x.shape == ref_x.shape

    with pytest.raises(AssertionError):
        # the length of ref_x.shape must be 4
        flow = torch.randn(2, 2, 100, 100)
        ref_x = torch.randn(2, 128, 256, 256, 256)
        x = flow_warp_feats(flow, ref_x)

    with pytest.raises(AssertionError):
        # the length of flow.shape must be 4
        flow = torch.randn(2, 2, 100, 100, 100)
        ref_x = torch.randn(2, 128, 256, 256)
        x = flow_warp_feats(flow, ref_x)

    with pytest.raises(AssertionError):
        # flow.shape[1] == 2
        flow = torch.randn(2, 3, 100, 100)
        ref_x = torch.randn(2, 128, 256, 256, 256)
        x = flow_warp_feats(flow, ref_x)
