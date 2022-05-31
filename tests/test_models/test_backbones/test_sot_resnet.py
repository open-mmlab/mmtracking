# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
import torch.nn as nn

from mmtrack.models import SOTResNet
from mmtrack.models.backbones.sot_resnet import SOTBottleneck, SOTResLayer


class TestSOTRestNet(TestCase):

    @classmethod
    def setUpClass(cls):
        with pytest.raises(AssertionError):
            # ResNet depth should be 50
            SOTResNet(20)
        cls.model = SOTResNet(
            depth=50,
            out_indices=(1, 2, 3),
            frozen_stages=3,
            strides=(1, 2, 1, 1),
            dilations=(1, 1, 2, 4),
            norm_eval=True,
            base_channels=1)
        cls.model.train()
        for num in range(1, 4):
            layer = getattr(cls.model, f'layer{num}')
            for param in layer.parameters():
                assert not param.requires_grad
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    assert not m.training

        for param in cls.model.layer4.parameters():
            assert param.requires_grad
        for m in cls.model.layer4.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert not m.training

    def test_forward(self):
        imgs = torch.randn(1, 3, 32, 32)
        feat = self.model(imgs)
        assert len(feat) == 3
        assert feat[0].shape == torch.Size([1, 8, 3, 3])
        assert feat[1].shape == torch.Size([1, 16, 3, 3])
        assert feat[2].shape == torch.Size([1, 32, 3, 3])


class TestSOTBottleneck(TestCase):

    @classmethod
    def setUpClass(cls):
        downsample = nn.Sequential(
            nn.Conv2d(4, 4, 3, stride=2, padding=1), nn.BatchNorm2d(4))
        cls.model = SOTBottleneck(
            4, 1, stride=2, dilation=2, downsample=downsample)

    def test_forward(self):
        x = torch.randn(1, 4, 8, 8)
        feats = self.model(x)
        assert feats.shape == torch.Size([1, 4, 4, 4])


class TestSOTResLayer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = SOTResLayer(SOTBottleneck, 4, 8, 2)
        cls.model.train()

    def test_forward(self):
        x = torch.randn(1, 4, 8, 8)
        feats = self.model(x)
        assert feats.shape == torch.Size([1, 32, 8, 8])
