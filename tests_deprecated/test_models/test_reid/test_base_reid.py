# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmtrack.models import REID


@pytest.mark.parametrize('model_type', ['BaseReID'])
def test_base_reid(model_type):
    model_class = REID.get(model_type)
    backbone = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch')
    neck = dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1)
    head = dict(
        type='LinearReIDHead',
        num_fcs=1,
        in_channels=2048,
        fc_channels=1024,
        out_channels=128,
        num_classes=378,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_pairwise=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'))
    model = model_class(backbone=backbone, neck=neck, head=head)
    model.train()
    x = torch.randn(32, 3, 256, 128)
    label = torch.randperm(32)
    outputs = model.forward_train(x, label)
    assert isinstance(outputs, dict)
    assert len(outputs) == 3
    assert 'triplet_loss' in outputs
    assert 'ce_loss' in outputs
    assert 'accuracy' in outputs
    model.eval()
    x = torch.randn(1, 3, 256, 128)
    outputs = model.simple_test(x)
    assert outputs.shape == (1, 128)

    head['num_classes'] = None
    # when loss_pairwise is set, num_classes must be a current number
    with pytest.raises(TypeError):
        model = model_class(backbone=backbone, neck=neck, head=head)

    head['num_classes'] = 378
    head['loss'] = None
    # when loss_pairwise is set, num_classes will be ignored.
    with pytest.warns(UserWarning):
        model = model_class(backbone=backbone, neck=neck, head=head)

    head['loss_pairwise'] = None
    # two losses cannot be none at the same time
    with pytest.raises(ValueError):
        model = model_class(backbone=backbone, neck=neck, head=head)
