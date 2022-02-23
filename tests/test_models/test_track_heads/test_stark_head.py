# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmtrack.models.track_heads.stark_head import (CornerPredictorHead,
                                                   ScoreHead, StarkHead,
                                                   StarkTransformer)


def test_corner_predictor_head():
    bbox_head = CornerPredictorHead(8, 8, feat_size=20, stride=16)
    inputs = torch.randn(1, 8, 20, 20)
    outputs = bbox_head(inputs)
    assert outputs.shape == (1, 4)


def test_score_head():
    score_head = ScoreHead(8, 8, 1, 3)
    inputs = torch.randn(1, 1, 1, 8)
    outputs = score_head(inputs)
    assert outputs.shape == (1, 1, 1)


def test_transormer_head():
    cfg = mmcv.Config(
        dict(
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=16,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1))
                    ],
                    ffn_cfgs=dict(
                        feedforward_channels=16, embed_dims=16, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
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
                        feedforward_channels=16, embed_dims=16, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))))
    self = StarkTransformer(**cfg)
    feat = torch.randn(20, 1, 16)
    mask = torch.zeros(1, 20, dtype=bool)
    query_embed = torch.randn(1, 16)
    pos_embed = torch.randn(20, 1, 16)
    out_dec, enc_mem = self.forward(feat, mask, query_embed, pos_embed)
    assert out_dec.shape == (1, 1, 1, 16)
    assert enc_mem.shape == (20, 1, 16)


def test_stark_head_loss():
    """Tests stark head loss when truth is non-empty."""
    head_cfg = dict(
        num_query=1,
        transformer=dict(
            type='StarkTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=16,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1))
                    ],
                    ffn_cfgs=dict(
                        feedforward_channels=16, embed_dims=16, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
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
                        feedforward_channels=16, embed_dims=16, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=8, normalize=True),
        bbox_head=dict(
            type='CornerPredictorHead',
            inplanes=16,
            channel=16,
            feat_size=20,
            stride=16),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        test_cfg=dict(
            search_factor=5.0,
            search_size=320,
            template_factor=2.0,
            template_size=128,
            update_intervals=[200]))
    cfg = mmcv.Config(head_cfg)

    self = StarkHead(**cfg)

    head_inputs = [
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
    track_results = self.forward(head_inputs)

    gt_bboxes = [
        torch.Tensor([[0., 23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [
        torch.Tensor([[0., 1]]),
    ]
    bboxes_losses = self.loss(track_results, gt_bboxes, gt_labels, (320, 320))
    assert bboxes_losses['loss_iou'] >= 0, 'iou loss should be'
    'non-zero or zero'
    assert bboxes_losses[
        'loss_bbox'] >= 0, 'bbox loss should be non-zero or zero'

    head_cfg.update(
        dict(
            cls_head=dict(
                type='ScoreHead',
                input_dim=16,
                hidden_dim=16,
                output_dim=1,
                num_layers=3,
                use_bn=False),
            frozen_module=['transformer', 'bbox_head'],
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True)))
    cfg = mmcv.Config(head_cfg)
    self = StarkHead(**cfg)
    track_results = self.forward(head_inputs)
    bboxes_losses = self.loss(track_results, gt_bboxes, gt_labels, (320, 320))
    assert bboxes_losses['loss_cls'] >= 0, 'iou loss should be'
    'non-zero or zero'
