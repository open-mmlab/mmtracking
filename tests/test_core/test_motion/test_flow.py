# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmtrack.core import flow_warp_feats


def test_flow_warp_feats():
    flow = torch.randn(2, 2, 10, 10)
    ref_x = torch.randn(2, 8, 32, 32)
    x = flow_warp_feats(ref_x, flow)
    assert x.shape == ref_x.shape

    with pytest.raises(AssertionError):
        # the length of ref_x.shape must be 4
        flow = torch.randn(2, 2, 10, 10)
        ref_x = torch.randn(2, 8, 32, 32, 32)
        x = flow_warp_feats(ref_x, flow)

    with pytest.raises(AssertionError):
        # the length of flow.shape must be 4
        flow = torch.randn(2, 2, 10, 10, 10)
        ref_x = torch.randn(2, 8, 32, 32)
        x = flow_warp_feats(ref_x, flow)

    with pytest.raises(AssertionError):
        # flow.shape[1] == 2
        flow = torch.randn(2, 3, 10, 10)
        ref_x = torch.randn(2, 8, 32, 32)
        x = flow_warp_feats(ref_x, flow)
