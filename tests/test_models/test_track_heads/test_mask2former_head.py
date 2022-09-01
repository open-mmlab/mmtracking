# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import Config

from mmtrack.models.track_heads import Mask2FormerHead
from mmtrack.structures import TrackDataSample
from mmtrack.testing import demo_mm_inputs


class TestMask2FormerHead(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = Config(
            dict(
                type='Mask2FormerHead',
                in_channels=[256, 512, 1024,
                             2048],  # pass to pixel_decoder inside
                strides=[4, 8, 16, 32],
                feat_channels=256,
                out_channels=256,
                num_classes=40,
                num_queries=100,
                num_frames=2,
                num_transformer_feat_level=3,
                pixel_decoder=dict(
                    type='mmdet.MSDeformAttnPixelDecoder',
                    num_outs=3,
                    norm_cfg=dict(type='GN', num_groups=32),
                    act_cfg=dict(type='ReLU'),
                    encoder=dict(
                        type='mmdet.DetrTransformerEncoder',
                        num_layers=6,
                        transformerlayers=dict(
                            type='BaseTransformerLayer',
                            attn_cfgs=dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                num_heads=8,
                                num_levels=3,
                                num_points=4,
                                im2col_step=64,
                                dropout=0.0,
                                batch_first=False,
                                norm_cfg=None,
                                init_cfg=None),
                            ffn_cfgs=dict(
                                type='FFN',
                                embed_dims=256,
                                feedforward_channels=1024,
                                num_fcs=2,
                                ffn_drop=0.0,
                                act_cfg=dict(type='ReLU', inplace=True)),
                            operation_order=('self_attn', 'norm', 'ffn',
                                             'norm')),
                        init_cfg=None),
                    positional_encoding=dict(
                        type='mmdet.SinePositionalEncoding',
                        num_feats=128,
                        normalize=True),
                    init_cfg=None),
                enforce_decoder_input_project=False,
                positional_encoding=dict(
                    type='SinePositionalEncoding3D',
                    num_feats=128,
                    normalize=True),
                transformer_decoder=dict(
                    type='mmdet.DetrTransformerDecoder',
                    return_intermediate=True,
                    num_layers=9,
                    transformerlayers=dict(
                        type='mmdet.DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=False),
                        ffn_cfgs=dict(
                            embed_dims=256,
                            feedforward_channels=2048,
                            num_fcs=2,
                            act_cfg=dict(type='ReLU', inplace=True),
                            ffn_drop=0.0,
                            dropout_layer=None,
                            add_identity=True),
                        feedforward_channels=2048,
                        operation_order=('cross_attn', 'norm', 'self_attn',
                                         'norm', 'ffn', 'norm')),
                    init_cfg=None),
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=2.0,
                    reduction='mean',
                    class_weight=[1.0] * 40 + [0.1]),
                loss_mask=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='mean',
                    loss_weight=5.0),
                loss_dice=dict(
                    type='mmdet.DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    reduction='mean',
                    naive_dice=True,
                    eps=1.0,
                    loss_weight=5.0),
                train_cfg=dict(
                    num_points=12544,
                    oversample_ratio=3.0,
                    importance_sample_ratio=0.75,
                    assigner=dict(
                        type='mmdet.HungarianAssigner',
                        match_costs=[
                            dict(type='mmdet.ClassificationCost', weight=2.0),
                            dict(
                                type='mmdet.CrossEntropyLossCost',
                                weight=5.0,
                                use_sigmoid=True),
                            dict(
                                type='mmdet.DiceCost',
                                weight=5.0,
                                pred_act=True,
                                eps=1.0)
                        ]),
                    sampler=dict(type='mmdet.MaskPseudoSampler'))))

    def test_mask2former_head_loss(self):
        mask2former_head = Mask2FormerHead(**self.config)
        mask2former_head.init_weights()
        s = 256
        feats = [
            torch.rand(2, 256 * (2**i), s // stride, s // stride)
            for i, stride in enumerate([8, 16, 32, 64])
        ]
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            frame_id=0,
            num_items=[3],
            num_key_imgs=2,
            image_shapes=[(3, s, s)],
            num_classes=2,
            with_mask=True)
        data_sample = packed_inputs['data_samples'][0]
        data_sample.gt_instances['map_instances_to_img_idx'] = torch.tensor(
            [0, 0, 1])
        loss = mask2former_head.loss(feats, [data_sample])
        # loss_cls, loss_mask and loss_dice
        assert len(loss) == 30

    def test_mask2former_head_predict(self):
        mask2former_head = Mask2FormerHead(**self.config)
        mask2former_head.init_weights()
        s = 256
        # assume the video has 30 frames
        feats = [
            torch.rand(30, 256 * (2**i), s // stride, s // stride)
            for i, stride in enumerate([8, 16, 32, 64])
        ]

        img_metas = dict(
            img_shape=(s, s),
            ori_shape=(s, s),
            scale_factor=(1, 1),
            pad_shape=(s, s),
            batch_input_shape=(s, s))
        data_sample = TrackDataSample(metainfo=img_metas)
        results = mask2former_head.predict(feats, [data_sample])

        assert len(results) == 1
        assert len(results[0]) == 10
