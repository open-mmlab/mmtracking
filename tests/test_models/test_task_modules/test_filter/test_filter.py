# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models.task_modules import apply_feat_transpose, apply_filter


def test_apply_filter():
    feat = torch.randn(3, 4, 8, 18, 18)
    filter = torch.randn(4, 8, 4, 4)
    scores = apply_filter(feat, filter)
    assert scores.shape == torch.Size([3, 4, 19, 19])
    feat = torch.randn(3, 8, 22, 22)
    filter = torch.randn(1, 8, 4, 4)
    scores = apply_filter(feat, filter)
    assert scores.shape == torch.Size([3, 1, 23, 23])


def test_apply_feat_transpose():
    feat = torch.randn(3, 4, 8, 18, 18)
    activation = torch.randn(3, 4, 19, 19)
    filter_grad = apply_feat_transpose(feat, activation, (4, 4))
    assert filter_grad.shape == torch.Size([4, 8, 4, 4])
    feat = torch.randn(3, 8, 22, 22)
    activation = torch.randn(3, 1, 23, 23)
    filter_grad = apply_feat_transpose(
        feat, activation, (4, 4), training=False)
    assert filter_grad.shape == torch.Size([1, 8, 4, 4])
