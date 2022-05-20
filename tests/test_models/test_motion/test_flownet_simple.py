# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmtrack.models.motion import FlowNetSimple


class TestFlowNetSimple(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.flownet = FlowNetSimple(img_scale_factor=0.5)
        cls.flownet.init_weights()
        cls.flownet.train()

    def test_forward(self):
        imgs = torch.randn(2, 6, 112, 224)
        metainfo = dict(img_shape=(112, 224, 3))
        preprocess_cfg = dict(
            mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))

        flow = self.flownet(imgs, metainfo, preprocess_cfg)
        assert flow.shape == torch.Size([2, 2, 112, 224])
