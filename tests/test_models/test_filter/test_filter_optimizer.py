# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models import PrDiMPFilterOptimizer


def test_prdimp_steepest_descent_newton():
    optimizer = PrDiMPFilterOptimizer(
        num_iters=5,
        feat_stride=16,
        init_step_length=1.0,
        init_filter_regular=0.05,
        gauss_sigma=0.9,
        alpha_eps=0.05,
        min_filter_regular=0.05,
        label_thres=0)
    filter = torch.randn(4, 8, 4, 4)
    feat = torch.randn(3, 4, 8, 22, 22)
    bboxes = torch.randn(3, 4, 4) * 100
    new_filter, filter_iters, losses = optimizer(filter, feat, bboxes)
    assert new_filter.shape == filter.shape
    assert len(filter_iters) == len(losses) == 6
