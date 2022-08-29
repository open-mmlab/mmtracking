# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import mmengine
import torch
from mmengine.structures import InstanceData

from mmtrack.registry import MODELS
from mmtrack.testing import demo_mm_inputs
from mmtrack.utils import register_all_modules


class TestSelsaRoIHead(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        selsa_bbox_roi_head_cfg = dict(
            type='mmtrack.SelsaRoIHead',
            _scope_='mmdet',
            bbox_roi_extractor=dict(
                type='mmtrack.SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=4,
                featmap_strides=[16]),
            bbox_head=dict(
                type='mmtrack.SelsaBBoxHead',
                num_shared_fcs=2,
                in_channels=4,
                fc_out_channels=4,
                roi_feat_size=7,
                num_classes=30,
                aggregator=dict(
                    type='mmtrack.SelsaAggregator',
                    in_channels=4,
                    num_attention_blocks=2),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.2, 0.2, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            train_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))
        cfg = mmengine.Config(selsa_bbox_roi_head_cfg)
        cls.roi_head = MODELS.build(cfg)
        assert cls.roi_head.with_bbox

    def _fake_inputs(self, img_size, proposal_len):
        """Create a fake proposal list and feature maps."""
        img_metas = [{
            'img_shape': (img_size, img_size),
            'scale_factor': 1,
        }]
        proposals_list = []
        for i in range(len(img_metas)):
            result = InstanceData(metainfo=img_metas[i])
            proposal = torch.randn(proposal_len, 4).to(device='cuda')
            result.bboxes = proposal
            proposals_list.append(result)

        feats = []
        for i in range(len(self.roi_head.bbox_roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 4, img_size // (2**(i + 2)),
                           img_size // (2**(i + 2))).to(device='cuda'))
        feats = tuple(feats)

        return proposals_list, feats

    def test_loss(self):
        if not torch.cuda.is_available():
            # RoI pooling only support in GPU
            return unittest.skip('test requires GPU and torch+cuda')

        self.roi_head = self.roi_head.cuda()

        proposal_list, feats = self._fake_inputs(256, 100)
        ref_proposal_list, ref_feats = self._fake_inputs(256, 100)

        # When truth is non-empty then both cls, box
        # should be nonzero for random inputs
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            image_shapes=[(3, 256, 256)],
            frame_id=0,
            num_items=[1],
            num_ref_imgs=2)
        batch_data_samples = []
        for data_sample in packed_inputs['data_samples']:
            batch_data_samples.append(data_sample.to(device='cuda'))
        out = self.roi_head.loss(feats, ref_feats, proposal_list,
                                 ref_proposal_list, batch_data_samples)
        loss_cls = out['loss_cls']
        loss_bbox = out['loss_bbox']
        self.assertGreater(loss_cls.sum(), 0, 'cls loss should be non-zero')
        self.assertGreater(loss_bbox.sum(), 0, 'box loss should be non-zero')

        # When there is no truth, the cls loss should be nonzero but
        # there should be no box and mask loss.
        proposal_list, feats = self._fake_inputs(256, 100)
        ref_proposal_list, ref_feats = self._fake_inputs(256, 100)
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            image_shapes=[(3, 256, 256)],
            num_items=[0],
            frame_id=0,
            num_ref_imgs=2)
        batch_data_samples = []
        for data_sample in packed_inputs['data_samples']:
            batch_data_samples.append(data_sample.to(device='cuda'))
        out = self.roi_head.loss(feats, ref_feats, proposal_list,
                                 ref_proposal_list, batch_data_samples)
        empty_cls_loss = out['loss_cls']
        empty_bbox_loss = out['loss_bbox']
        self.assertGreater(empty_cls_loss.sum(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_bbox_loss.sum(), 0,
            'there should be no box loss when there are no true boxes')

    def test_predict(self):
        if not torch.cuda.is_available():
            # RoI pooling only support in GPU
            return unittest.skip('test requires GPU and torch+cuda')

        self.roi_head = self.roi_head.cuda()

        proposal_list, feats = self._fake_inputs(256, 100)
        ref_proposal_list, ref_feats = self._fake_inputs(256, 100)
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            image_shapes=[(3, 256, 256)],
            frame_id=0,
            num_items=[1],
            num_ref_imgs=2)
        batch_data_samples = []
        for data_sample in packed_inputs['data_samples']:
            batch_data_samples.append(data_sample.to(device='cuda'))
        out = self.roi_head.predict(feats, ref_feats, proposal_list,
                                    ref_proposal_list, batch_data_samples)
        assert out[0]['bboxes'].shape[0] == out[0]['scores'].shape[0]
