# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmtrack.models.backbones import SOTResNet


def test_sot_resnet_backbone():
    """Test sot resnet backbone."""
    with pytest.raises(AssertionError):
        # ResNet depth should be 50
        SOTResNet(20)

    # Test SOTResNet50 with layers 2, 3, 4 out forward
    cfg = dict(
        depth=50,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        norm_eval=True)
    model = SOTResNet(**cfg)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 127, 127)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 512, 15, 15])
    assert feat[1].shape == torch.Size([1, 1024, 15, 15])
    assert feat[2].shape == torch.Size([1, 2048, 15, 15])

    imgs = torch.randn(1, 3, 255, 255)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 512, 31, 31])
    assert feat[1].shape == torch.Size([1, 1024, 31, 31])
    assert feat[2].shape == torch.Size([1, 2048, 31, 31])
