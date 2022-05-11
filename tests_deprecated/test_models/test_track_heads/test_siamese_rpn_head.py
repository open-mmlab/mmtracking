# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmtrack.models.track_heads import CorrelationHead, SiameseRPNHead


def test_correlation_head():
    self = CorrelationHead(16, 16, 2)
    kernel = torch.rand(1, 16, 7, 7)
    search = torch.rand(1, 16, 31, 31)
    out = self(kernel, search)
    assert out.size() == (1, 2, 25, 25)


def test_siamese_rpn_head_loss():
    """Tests siamese rpn head loss when truth is non-empty."""
    cfg = mmcv.Config(
        dict(
            anchor_generator=dict(
                type='SiameseRPNAnchorGenerator',
                strides=[8],
                ratios=[0.33, 0.5, 1, 2, 3],
                scales=[8]),
            in_channels=[16, 16, 16],
            weighted_sum=True,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[1., 1., 1., 1.]),
            loss_cls=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=1.2),
            train_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.6,
                    match_low_quality=False),
                sampler=dict(
                    type='RandomSampler',
                    num=64,
                    pos_fraction=0.25,
                    add_gt_as_proposals=False),
                num_neg=16,
                exemplar_size=127,
                search_size=255),
            test_cfg=dict(penalty_k=0.05, window_influence=0.42, lr=0.38)))

    self = SiameseRPNHead(**cfg)

    z_feats = tuple(
        [torch.rand(1, 16, 7, 7) for i in range(len(self.cls_heads))])
    x_feats = tuple(
        [torch.rand(1, 16, 31, 31) for i in range(len(self.cls_heads))])
    cls_score, bbox_pred = self.forward(z_feats, x_feats)

    gt_bboxes = [
        torch.Tensor([[0., 23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    bbox_targets = self.get_targets(gt_bboxes, cls_score.shape[2:], [True])
    gt_losses = self.loss(cls_score, bbox_pred, *bbox_targets)
    assert gt_losses['loss_rpn_cls'] > 0, 'cls loss should be non-zero'
    assert gt_losses[
        'loss_rpn_bbox'] >= 0, 'box loss should be non-zero or zero'

    gt_bboxes = [
        torch.Tensor([[0., 23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    bbox_targets = self.get_targets(gt_bboxes, cls_score.shape[2:], [False])
    gt_losses = self.loss(cls_score, bbox_pred, *bbox_targets)
    assert gt_losses['loss_rpn_cls'] > 0, 'cls loss should be non-zero'
    assert gt_losses['loss_rpn_bbox'] == 0, 'box loss should be zero'
