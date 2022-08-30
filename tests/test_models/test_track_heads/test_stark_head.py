# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import mmengine
import torch
from mmengine.structures import InstanceData

from mmtrack.models.track_heads.stark_head import (CornerPredictorHead,
                                                   ScoreHead, StarkHead,
                                                   StarkTransformer)
from mmtrack.structures import TrackDataSample
from mmtrack.testing import random_boxes


class TestCornerPredictorHead(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bbox_head = CornerPredictorHead(8, 8, feat_size=20, stride=16)

    def test_corner_predictor_Head(self):
        inputs = torch.randn(1, 8, 20, 20)
        outputs = self.bbox_head(inputs)
        assert outputs.shape == (1, 4)


class TestScoreHead(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.score_head = ScoreHead(8, 8, 1, 3)

    def test_corner_predictor_Head(self):
        inputs = torch.randn(1, 1, 1, 8)
        outputs = self.score_head(inputs)
        assert outputs.shape == (1, 1)


class TestStarkTransformer(TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = mmengine.Config(
            dict(
                encoder=dict(
                    type='mmdet.DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=16,
                                num_heads=8,
                                attn_drop=0.1,
                                dropout_layer=dict(
                                    type='Dropout', drop_prob=0.1))
                        ],
                        ffn_cfgs=dict(
                            feedforward_channels=16,
                            embed_dims=16,
                            ffn_drop=0.1),
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='mmdet.DetrTransformerDecoder',
                    return_intermediate=False,
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=16,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                        ffn_cfgs=dict(
                            feedforward_channels=16,
                            embed_dims=16,
                            ffn_drop=0.1),
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm')))))
        cls.stark_transformer = StarkTransformer(**cfg)

    def test_stark_transformer(self):
        feat = torch.randn(20, 1, 16)
        mask = torch.zeros(1, 20, dtype=bool)
        query_embed = torch.randn(1, 16)
        pos_embed = torch.randn(20, 1, 16)
        out_dec, enc_mem = self.stark_transformer(feat, mask, query_embed,
                                                  pos_embed)
        assert out_dec.shape == (1, 1, 1, 16)
        assert enc_mem.shape == (20, 1, 16)


class TestStarkHead(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = dict(
            num_query=1,
            transformer=dict(
                type='StarkTransformer',
                encoder=dict(
                    type='mmdet.DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=16,
                                num_heads=8,
                                attn_drop=0.1,
                                dropout_layer=dict(
                                    type='Dropout', drop_prob=0.1))
                        ],
                        ffn_cfgs=dict(
                            feedforward_channels=16,
                            embed_dims=16,
                            ffn_drop=0.1),
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='mmdet.DetrTransformerDecoder',
                    return_intermediate=False,
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=16,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                        ffn_cfgs=dict(
                            feedforward_channels=16,
                            embed_dims=16,
                            ffn_drop=0.1),
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm'))),
            ),
            positional_encoding=dict(
                type='mmdet.SinePositionalEncoding',
                num_feats=8,
                normalize=True),
            bbox_head=dict(
                type='CornerPredictorHead',
                inplanes=16,
                channel=16,
                feat_size=20,
                stride=16),
            loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
            loss_iou=dict(type='mmdet.GIoULoss', loss_weight=2.0),
            test_cfg=dict(
                search_factor=5.0,
                search_size=320,
                template_factor=2.0,
                template_size=128,
                update_intervals=[200]))
        cls.stark_head_st1 = StarkHead(**cls.cfg)

        cls.cfg.update(
            dict(
                cls_head=dict(
                    type='ScoreHead',
                    input_dim=16,
                    hidden_dim=16,
                    output_dim=1,
                    num_layers=3,
                    use_bn=False),
                frozen_module=['transformer', 'bbox_head'],
                loss_cls=dict(type='mmdet.CrossEntropyLoss',
                              use_sigmoid=True)))
        cls.stark_head_st2 = StarkHead(**cls.cfg)

        cls.head_inputs = [
            dict(
                feat=(torch.rand(1, 16, 8, 8), ),
                mask=torch.zeros(1, 128, 128, dtype=bool)),
            dict(
                feat=(torch.rand(1, 16, 8, 8), ),
                mask=torch.zeros(1, 128, 128, dtype=bool)),
            dict(
                feat=(torch.rand(1, 16, 20, 20), ),
                mask=torch.zeros(1, 320, 320, dtype=bool))
        ]

        data_sample = TrackDataSample()
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.Tensor([True]).long()
        data_sample.search_gt_instances = gt_instances
        data_sample.gt_instances = deepcopy(gt_instances)
        data_sample.set_metainfo(
            dict(
                search_ori_shape=(500, 500),
                ori_shape=(500, 500),
                search_img_shape=[320, 320]))
        cls.batch_data_samples = [data_sample]

    def test_loss(self):
        """Test the forward of stark head in loss mode."""
        self.stark_head_st1.train()
        losses = self.stark_head_st1.loss(self.head_inputs,
                                          self.batch_data_samples)
        assert losses['loss_iou'] >= 0, 'iou loss should be non-zero or zero'
        assert losses['loss_bbox'] >= 0, 'box loss should be non-zero or zero'

        self.stark_head_st2.train()
        losses = self.stark_head_st2.loss(self.head_inputs,
                                          self.batch_data_samples)
        assert losses['loss_cls'] > 0, 'cls loss should be non-zero'

    def test_predict(self):
        """Test the forward of stark head in predict mode."""
        prev_bbox = random_boxes(1, 50).squeeze()
        scale_factor = torch.Tensor([3.])

        self.stark_head_st2.eval()
        results = self.stark_head_st2.predict(self.head_inputs,
                                              self.batch_data_samples,
                                              prev_bbox, scale_factor)
        assert results[0].scores >= 0
        assert results[0].bboxes.shape == (1, 4)
