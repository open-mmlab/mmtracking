# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmtrack.models.roi_heads.bbox_heads import SelsaBBoxHead


class TestSelsaBBoxHead(TestCase):

    @classmethod
    def setUpClass(cls):
        selsa_bbox_head_config = dict(
            num_shared_fcs=2,
            in_channels=2,
            fc_out_channels=4,
            roi_feat_size=3,
            num_classes=10,
            aggregator=dict(
                type='SelsaAggregator', in_channels=4, num_attention_blocks=4),
            bbox_coder=dict(
                type='mmdet.DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.2, 0.2, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0))
        cls.model = SelsaBBoxHead(**selsa_bbox_head_config)

    def test_forward(self):
        x = torch.randn(2, 2, 3, 3)
        ref_x = torch.randn(3, 2, 3, 3)
        cls_scores, bbox_preds = self.model.forward(x, ref_x)
        assert cls_scores.shape == (2, 11)
        assert bbox_preds.shape == (2, 40)
