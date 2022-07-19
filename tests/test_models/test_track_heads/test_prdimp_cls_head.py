# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import mmcv
import torch

from mmtrack.models.track_heads.prdimp_cls_head import PrdimpClsHead


class TestLinearBlock(TestCase):

    def setUp(self):
        cfg = mmcv.Config(
            dict(
                in_dim=32,
                out_dim=16,
                filter_initializer=dict(
                    type='FilterClassifierInitializer',
                    filter_size=4,
                    feature_dim=16,
                    feature_stride=16),
                filter_optimizer=dict(
                    type='PrDiMPSteepestDescentNewton',
                    num_iters=5,
                    feat_stride=16,
                    init_step_length=1.0,
                    init_filter_regular=0.05,
                    gauss_sigma=0.9,
                    alpha_eps=0.05,
                    min_filter_regular=0.05,
                    label_thres=0),
                locate_cfg=dict(
                    no_target_min_score=0.04,
                    distractor_thres=0.8,
                    hard_neg_thres=0.5,
                    target_neighborhood_scale=2.2,
                    dispalcement_scale=0.8,
                    update_scale_when_uncertain=True),
                update_cfg=dict(
                    sample_memory_size=50,
                    normal_lr=0.01,
                    hard_neg_lr=0.02,
                    init_samples_min_weight=0.25,
                    train_skipping=20),
                optimizer_cfg=dict(
                    init_update_iters=10, update_iters=2, hard_neg_iters=1),
                test_cfg=dict(img_sample_size=352)))

        self.model = PrdimpClsHead(**cfg)

    def test_prdimp_cls_head_predict_mode(self):
        self.model.eval()
        backbone_feats = torch.randn(2, 32, 22, 22)
        target_bboxes = torch.rand(4, 4) * 150

        if torch.cuda.is_available():
            self.model = self.model.to('cuda:0')
            backbone_feats = backbone_feats.to('cuda:0')
            target_bboxes = target_bboxes.to('cuda:0')
            self.model.init_classifier(
                backbone_feats, target_bboxes, dropout_probs=[0.2, 0.2])
        else:
            self.target_filter = torch.randn(1, 16, 4, 4)
            cls_feats = self.model.get_cls_feats(backbone_feats)
            self.model.init_memory(cls_feats, target_bboxes)

        scores, test_feat = self.model(backbone_feats)
        sample_size = torch.Tensor([352., 352.])
        prev_bbox = torch.rand(4) * 150
        if torch.cuda.is_available():
            sample_size = sample_size.to('cuda:0')
            prev_bbox = prev_bbox.to('cuda:0')
        self.model.predict_by_feat(scores[:1], prev_bbox,
                                   target_bboxes[:1, :2], 4)
        if torch.cuda.is_available():
            self.model.update_classifier(target_bboxes[1], 1, False)
