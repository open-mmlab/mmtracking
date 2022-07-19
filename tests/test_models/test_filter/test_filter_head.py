# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmtrack.models import FilterClassifierInitializer


@pytest.mark.skipif(
    not torch.cuda.is_available, reason='test case under gpu environment')
def test_filter_classifier_initializer():
    classifier_initializer = FilterClassifierInitializer(
        filter_size=4, feature_dim=8, feature_stride=16).to('cuda:0')
    feats = torch.randn(4, 8, 22, 22, device='cuda:0')
    bboxes = torch.randn(4, 4, device='cuda:0') * 100
    bboxes[:, 2:] += bboxes[:, :2]
    filter = classifier_initializer(feats, bboxes)
    assert filter.shape == torch.Size([1, 8, 4, 4])
