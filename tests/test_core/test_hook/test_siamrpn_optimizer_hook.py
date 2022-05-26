# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.nn as nn

from mmtrack.core.hook import SiameseRPNOptimizerHook


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
        self.layer2 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))

    def forward(self, x):
        return self.layer2(self.layer1(x))


class TestSiameseRPNOptimizerHook(TestCase):

    def test_siamese_rpn_optimizer_hook(self):
        runner = Mock()
        runner.model = Mock()
        runner.model.module = Mock()
        runner.model.module.backbone = ToyModel()
        runner.epoch = 9
        runner.max_epochs = 20
        runner.model.module.backbone.train()
        for param in runner.model.module.backbone.layer2.parameters():
            param.requires_grad = False
        for m in runner.model.module.backbone.layer2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        hook = SiameseRPNOptimizerHook(
            backbone_start_train_epoch=10, backbone_train_layers=['layer2'])

        hook.before_train_epoch(runner)
        for param in runner.model.module.backbone.layer2.parameters():
            self.assertFalse(param.requires_grad)
        for m in runner.model.module.backbone.layer2.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.assertFalse(m.training)

        runner.epoch = 10
        hook.before_train_epoch(runner)
        for param in runner.model.module.backbone.layer2.parameters():
            self.assertTrue(param.requires_grad)
        for m in runner.model.module.backbone.layer2.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.assertTrue(m.training)
