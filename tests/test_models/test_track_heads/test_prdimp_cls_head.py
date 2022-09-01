# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmtrack.models.track_heads.prdimp_cls_head import PrDiMPClsHead
from mmtrack.structures import TrackDataSample


class TestLinearBlock(TestCase):

    def setUp(self):
        cfg = dict(
            in_dim=32,
            out_dim=16,
            filter_initializer=dict(
                type='FilterInitializer',
                filter_size=4,
                feature_dim=16,
                feature_stride=16),
            filter_optimizer=dict(
                type='PrDiMPFilterOptimizer',
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
            test_cfg=dict(img_sample_size=352),
            loss_cls=dict(type='KLGridLoss'),
            train_cfg=dict(
                feat_size=(18, 18),
                img_size=(288, 288),
                sigma_factor=0.05,
                end_pad_if_even=True,
                gauss_label_bias=0.,
                use_gauss_density=True,
                label_density_norm=True,
                label_density_threshold=0.,
                label_density_shrink=0,
                loss_weights=dict(cls_init=0.25, cls_iter=1., cls_final=0.25)))

        self.model = PrDiMPClsHead(**cfg)

    def test_prdimp_cls_head_predict(self):
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
            self.model.target_filter = torch.randn(1, 16, 4, 4)
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

    def test_prdimp_cls_head_loss(self):
        if not torch.cuda.is_available():
            return
        self.model.train()
        model = self.model.to('cuda:0')
        template_feats = (torch.randn(2, 32, 18, 18).to('cuda:0'), )
        search_feats = (torch.randn(2, 32, 18, 18).to('cuda:0'), )
        target_bboxes = (torch.rand(2, 4) * 150).to('cuda:0')

        gt_instances = InstanceData()
        gt_instances['bboxes'] = target_bboxes
        search_gt_instances = copy.deepcopy(gt_instances)

        data_sample = TrackDataSample()
        data_sample.gt_instances = gt_instances
        data_sample.search_gt_instances = search_gt_instances

        model.loss(template_feats, search_feats, [data_sample])
