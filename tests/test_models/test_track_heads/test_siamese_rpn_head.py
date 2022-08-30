# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import mmengine
import numpy as np
import torch
from mmengine.structures import InstanceData

from mmtrack.models.track_heads import CorrelationHead, SiameseRPNHead
from mmtrack.structures import TrackDataSample
from mmtrack.testing import random_boxes
from mmtrack.utils import register_all_modules


class TestCorrelationHead(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.correlation_head = CorrelationHead(16, 16, 2)

    def test_forward(self):
        kernel = torch.rand(1, 16, 7, 7)
        search = torch.rand(1, 16, 31, 31)
        out = self.correlation_head(kernel, search)
        assert out.size() == (1, 2, 25, 25)


class TestSiameseRPNHead(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cfg = mmengine.Config(
            dict(
                anchor_generator=dict(
                    type='SiameseRPNAnchorGenerator',
                    strides=[8],
                    ratios=[0.33, 0.5, 1, 2, 3],
                    scales=[8]),
                in_channels=[1, 1, 1],
                weighted_sum=True,
                bbox_coder=dict(
                    type='mmdet.DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[1., 1., 1., 1.]),
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss',
                    reduction='sum',
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='mmdet.L1Loss', reduction='sum', loss_weight=1.2),
                train_cfg=dict(
                    assigner=dict(
                        type='mmdet.MaxIoUAssigner',
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.6,
                        match_low_quality=False,
                        iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
                    sampler=dict(
                        type='mmdet.RandomSampler',
                        num=64,
                        pos_fraction=0.25,
                        add_gt_as_proposals=False),
                    num_neg=16,
                    exemplar_size=127,
                    search_size=255),
                test_cfg=dict(penalty_k=0.05, window_influence=0.42, lr=0.38)))
        cls.siamese_rpn_head = SiameseRPNHead(**cfg)
        cls.rng = np.random.RandomState(0)

    def test_get_init_targets(self):
        bboxes = torch.randn(10, 4)
        (labels, labels_weights, bbox_targets,
         bbox_weights) = self.siamese_rpn_head._get_init_targets(
             bboxes, (25, 25))
        assert labels.shape == (25 * 25 * 5, )
        assert labels_weights.shape == (25 * 25 * 5, )
        assert bbox_targets.shape == (25 * 25 * 5, 4)
        assert bbox_weights.shape == (25 * 25 * 5, 4)

    def test_get_positive_pair_targets(self):
        gt_bboxes = random_boxes(1, 50)
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = torch.Tensor([True]).long()
        (labels, labels_weights, bbox_targets,
         bbox_weights) = self.siamese_rpn_head.get_targets(
             gt_instances, [25, 25])
        assert labels.shape == (1, 25 * 25 * 5)
        assert labels_weights.shape == (1, 25 * 25 * 5)
        assert bbox_targets.shape == (1, 25 * 25 * 5, 4)
        assert bbox_weights.shape == (1, 25 * 25 * 5, 4)

    def test_get_negative_pair_targets(self):
        gt_bboxes = random_boxes(1, 50)
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = torch.Tensor([False]).long()
        (labels, labels_weights, bbox_targets,
         bbox_weights) = self.siamese_rpn_head.get_targets(
             gt_instances, [25, 25])
        assert labels.shape == (1, 25 * 25 * 5)
        assert labels_weights.shape == (1, 25 * 25 * 5)
        assert bbox_targets.shape == (1, 25 * 25 * 5, 4)
        assert bbox_weights.shape == (1, 25 * 25 * 5, 4)

    def test_forward(self):
        z_feats = tuple([
            torch.rand(1, 1, 7, 7)
            for i in range(len(self.siamese_rpn_head.cls_heads))
        ])
        x_feats = tuple([
            torch.rand(1, 1, 31, 31)
            for i in range(len(self.siamese_rpn_head.cls_heads))
        ])
        cls_score, bbox_pred = self.siamese_rpn_head.forward(z_feats, x_feats)
        assert cls_score.shape == (1, 10, 25, 25)
        assert bbox_pred.shape == (1, 20, 25, 25)

    def test_get_targets(self):
        batch_gt_instances = []
        gt_bboxes = random_boxes(2, 50)
        gt_labels = torch.randint(2, (2, 1)).long()
        for i in range(2):
            gt_instances = InstanceData()
            gt_instances.bboxes = gt_bboxes[i:i + 1]
            gt_instances.labels = gt_labels[i]
            batch_gt_instances.append(gt_instances)
        (labels, labels_weights, bbox_targets,
         bbox_weights) = self.siamese_rpn_head.get_targets(
             batch_gt_instances, [25, 25])
        assert labels.shape == (2, 25 * 25 * 5)
        assert labels_weights.shape == (2, 25 * 25 * 5)
        assert bbox_targets.shape == (2, 25 * 25 * 5, 4)
        assert bbox_weights.shape == (2, 25 * 25 * 5, 4)

    def test_predict(self):
        z_feats = tuple([
            torch.rand(1, 1, 7, 7)
            for i in range(len(self.siamese_rpn_head.cls_heads))
        ])
        x_feats = tuple([
            torch.rand(1, 1, 31, 31)
            for i in range(len(self.siamese_rpn_head.cls_heads))
        ])
        prev_bbox = random_boxes(1, 50).squeeze()
        scale_factor = torch.Tensor([3.])

        data_sample = TrackDataSample()
        data_sample.set_metainfo(dict(ori_shape=(200, 200)))
        batch_data_samples = [data_sample]
        results = self.siamese_rpn_head.predict(z_feats, x_feats,
                                                batch_data_samples, prev_bbox,
                                                scale_factor)
        assert results[0].scores >= 0
        assert results[0].bboxes.shape == (1, 4)

    def test_loss(self):
        z_feats = tuple([
            torch.rand(1, 1, 7, 7)
            for i in range(len(self.siamese_rpn_head.cls_heads))
        ])
        x_feats = tuple([
            torch.rand(1, 1, 31, 31)
            for i in range(len(self.siamese_rpn_head.cls_heads))
        ])

        data_sample = TrackDataSample()
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.Tensor([True]).long()
        data_sample.search_gt_instances = gt_instances
        data_sample.gt_instances = deepcopy(gt_instances)
        data_sample.set_metainfo(dict(search_ori_shape=(200, 200)))
        batch_data_samples = [data_sample]

        gt_losses = self.siamese_rpn_head.loss(z_feats, x_feats,
                                               batch_data_samples)
        assert gt_losses['loss_rpn_cls'] > 0, 'cls loss should be non-zero'
        assert gt_losses[
            'loss_rpn_bbox'] >= 0, 'box loss should be non-zero or zero'

        gt_instances.labels = torch.Tensor([False]).long()
        gt_losses = self.siamese_rpn_head.loss(z_feats, x_feats,
                                               batch_data_samples)
        assert gt_losses['loss_rpn_cls'] > 0, 'cls loss should be non-zero'
        assert gt_losses['loss_rpn_bbox'] == 0, 'box loss should be zero'
