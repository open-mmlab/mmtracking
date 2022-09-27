# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmtrack.models.track_heads.mixformer_head import (MixFormerHead,
                                                       MixFormerScoreDecoder)


def test_score_head():
    if not torch.cuda.is_available():
        return

    score_head = MixFormerScoreDecoder().cuda()
    search_feat = torch.randn(1, 384, 20, 20).cuda()
    template_feat = torch.randn(1, 384, 8, 8).cuda()
    search_box = torch.rand(1, 4).cuda()

    outputs = score_head(search_feat, template_feat, search_box)
    assert outputs.shape == (1, 1)


def test_mixformer_head():
    if not torch.cuda.is_available():
        return

    cfg = dict(
        bbox_head=dict(
            type='CornerPredictorHead',
            inplanes=384,
            channel=384,
            feat_size=20,
            stride=16),
        score_head=dict(
            type='MixFormerScoreDecoder',
            pool_size=4,
            feat_size=20,
            stride=16,
            num_heads=6,
            hidden_dim=384,
            num_layers=3))

    cfg = mmcv.Config(cfg)

    head = MixFormerHead(**cfg).cuda()

    template = torch.randn(1, 384, 8, 8).cuda()
    search = torch.randn(1, 384, 20, 20).cuda()
    outputs = head(template, search, run_score_head=True)
    assert outputs['pred_bboxes'].shape == (1, 1, 4)
    assert outputs['pred_scores'].shape == (1, 1)
