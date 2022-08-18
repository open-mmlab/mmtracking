# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import mmengine
import pytest
import torch

from mmtrack.models.track_heads.iounet_head import IouNetHead, LinearBlock


class TestLinearBlock(TestCase):

    def setUp(self):
        self.model = LinearBlock(
            8, 8, 3, bias=True, batch_norm=True, relu=True)

    def test_forward(self):
        x = torch.randn(4, 8, 3, 3)
        output = self.model(x)
        assert output.shape == torch.Size([4, 8])


class TestIouNetHead(TestCase):

    def setUp(self):
        cfg = mmengine.Config(
            dict(
                in_dim=(16, 32),
                pred_in_dim=(16, 16),
                pred_inter_dim=(8, 8),
                bbox_cfg=dict(
                    num_init_random_boxes=9,
                    box_jitter_pos=0.1,
                    box_jitter_sz=0.5,
                    iounet_topk=3,
                    box_refine_step_length=2.5e-3,
                    box_refine_iter=10,
                    max_aspect_ratio=6,
                    box_refine_step_decay=1),
                test_cfg=dict(img_sample_size=352)))

        self.model = IouNetHead(**cfg)

    @pytest.mark.skipif(
        not torch.cuda.is_available, reason='test case under gpu environment')
    def test_prdimp_cls_head_predict_mode(self):

        backbone_feats = (torch.randn(1, 16, 22, 22, device='cuda:0'),
                          torch.randn(1, 32, 22, 22, device='cuda:0'))
        target_bboxes = torch.rand(1, 4, device='cuda:0') * 150

        model = self.model.to('cuda:0')
        model.eval()
        with torch.no_grad():
            model.init_iou_net(backbone_feats, target_bboxes)
            sample_center = torch.randn(1, 2, device='cuda:0') * 150
            model.predict(backbone_feats, None, target_bboxes, sample_center,
                          4)
