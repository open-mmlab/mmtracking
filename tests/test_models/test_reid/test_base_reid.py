# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from parameterized import parameterized

from mmtrack.core import ReIDDataSample
from mmtrack.registry import MODELS
from mmtrack.utils import register_all_modules
from ..utils import _get_model_cfg


class TestBaseReID(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        register_all_modules()

    @parameterized.expand([
        'reid/resnet50_b32x8_MOT17.py',
    ])
    def test_forward(self, cfg_file):
        model_cfg = _get_model_cfg(cfg_file)
        model = MODELS.build(model_cfg)
        inputs = torch.rand(4, 3, 256, 128)
        data_samples = [
            ReIDDataSample().set_gt_label(label) for label in (0, 0, 1, 1)
        ]

        # test mode='tensor'
        feats = model(inputs, mode='tensor')
        assert feats.shape == (4, 128)

        # test mode='loss'
        losses = model(inputs, data_samples, mode='loss')
        assert losses.keys() == {'triplet_loss', 'ce_loss', 'accuracy'}
        assert losses['ce_loss'].item() > 0
        assert losses['triplet_loss'].item() > 0
        assert 'top-1' in losses['accuracy']

        # test mode='predict'
        predictions = model(inputs, data_samples, mode='predict')
        for pred in predictions:
            assert isinstance(pred, ReIDDataSample)
            assert isinstance(pred.pred_feature, torch.Tensor)
            assert isinstance(pred.gt_label.label, torch.Tensor)
            assert pred.pred_feature.shape == (128, )
