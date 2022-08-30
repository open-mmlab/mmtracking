# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from torch import nn

from mmtrack.engine.hooks import SiamRPNBackboneUnfreezeHook


class TestSiamRPNBackboneUnfreezeHook(TestCase):

    def test_before_train_epoch(self):
        runner = Mock()
        runner.model.backbone = Mock()
        runner.model.backbone.layer1 = nn.Conv2d(1, 1, 1)
        runner.model.backbone.layer2 = nn.Sequential(
            nn.Conv2d(1, 1, 1), nn.BatchNorm2d(2))
        for layer in ['layer1', 'layer2']:
            for param in getattr(runner.model.backbone, layer).parameters():
                param.requires_grad = False
        runner.model.backbone.layer2[1].eval()
        hook = SiamRPNBackboneUnfreezeHook(
            backbone_start_train_epoch=10, backbone_train_layers=['layer2'])

        runner.epoch = 9
        hook.before_train_epoch(runner)
        for layer in ['layer1', 'layer2']:
            for param in getattr(runner.model.backbone, layer).parameters():
                self.assertFalse(param.requires_grad)
        self.assertFalse(runner.model.backbone.layer2[1].training)

        runner.epoch = 10
        hook.before_train_epoch(runner)
        for param in getattr(runner.model.backbone, 'layer1').parameters():
            self.assertFalse(param.requires_grad)
        for param in getattr(runner.model.backbone, 'layer2').parameters():
            self.assertTrue(param.requires_grad)
        self.assertTrue(runner.model.backbone.layer2[1].training)
