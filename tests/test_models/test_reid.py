import pytest
import torch

from mmtrack.models import REID


@pytest.mark.parametrize('model', ['BaseReID'])
def test_load_detections(model):
    model = REID.get(model)
    model = model(
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')))
    model.eval()
    x = torch.randn(1, 3, 256, 128)
    outputs = model.simple_test(x)
    assert outputs.shape == (1, 128)
