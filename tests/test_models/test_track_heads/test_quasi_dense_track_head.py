# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmtrack.registry import MODELS
from mmtrack.testing import demo_mm_inputs, random_boxes
from mmtrack.utils import register_all_modules


def _fake_proposals(img_metas, proposal_len):
    """Create a fake proposal list."""
    results = []
    for i in range(len(img_metas)):
        result = InstanceData(metainfo=img_metas[i])
        proposal = random_boxes(proposal_len, 10).to(device='cpu')
        result.bboxes = proposal
        results.append(result)
    return results


class TestQuasiDenseTrackHead(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cfg = Config(
            dict(
                type='QuasiDenseTrackHead',
                roi_extractor=dict(
                    _scope_='mmdet',
                    type='SingleRoIExtractor',
                    roi_layer=dict(
                        type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]),
                embed_head=dict(
                    type='QuasiDenseEmbedHead',
                    num_convs=4,
                    num_fcs=1,
                    embed_channels=256,
                    norm_cfg=dict(type='GN', num_groups=32),
                    loss_track=dict(
                        type='MultiPosCrossEntropyLoss', loss_weight=0.25),
                    loss_track_aux=dict(
                        type='L2Loss',
                        neg_pos_ub=3,
                        pos_margin=0,
                        neg_margin=0.1,
                        hard_mining=True,
                        loss_weight=1.0)),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                train_cfg=dict(
                    assigner=dict(
                        _scope_='mmdet',
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        _scope_='mmdet',
                        type='CombinedSampler',
                        num=256,
                        pos_fraction=0.5,
                        neg_pos_ub=3,
                        add_gt_as_proposals=True,
                        pos_sampler=dict(type='InstanceBalancedPosSampler'),
                        neg_sampler=dict(type='RandomSampler')))))
        cls.track_head = MODELS.build(cfg)

    def test_quasi_dense_track_head_loss(self):
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            frame_id=0,
            num_ref_imgs=1,
            image_shapes=[(3, 256, 256)])
        img_metas = [{
            'img_shape': (256, 256, 3),
            'scale_factor': 1,
        }]
        proposal_list = _fake_proposals(img_metas, 10)
        feats = []
        for i in range(len(self.track_head.roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 256, 256 // (2**(i + 2)),
                           256 // (2**(i + 2))).to(device='cpu'))
        key_feats = tuple(feats)
        ref_feats = key_feats
        loss_track = self.track_head.loss(key_feats, ref_feats, proposal_list,
                                          proposal_list,
                                          [packed_inputs[0]['data_sample']])
        assert loss_track['loss_track'] >= 0, 'track loss should be zero'
        assert loss_track['loss_track_aux'] > 0, 'aux loss should be non-zero'

    def test_quasi_dense_track_head_predict(self):
        feats = []
        for i in range(len(self.track_head.roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 256, 256 // (2**(i + 2)),
                           256 // (2**(i + 2))).to(device='cpu'))
        feats = tuple(feats)
        track_feat = self.track_head.predict(
            feats, [torch.Tensor([[10, 10, 20, 20]])])
        assert track_feat.size() == (1, 256)
