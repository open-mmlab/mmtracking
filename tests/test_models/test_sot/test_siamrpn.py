# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmtrack.core import TrackDataSample
from mmtrack.registry import MODELS
from mmtrack.utils import register_all_modules
from ..utils import _demo_mm_inputs, _get_model_cfg


class TestSiameseRPN(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    @parameterized.expand([
        'sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py',
    ])
    def test_init(self, cfg_file):
        model = _get_model_cfg(cfg_file)

        model = MODELS.build(model)
        assert model.backbone
        assert model.neck
        assert model.head

    @parameterized.expand([
        ('sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py', ('cpu', 'cuda')),
    ])
    def test_siamese_rpn_forward_loss_mode(self, cfg_file, devices):
        _model = _get_model_cfg(cfg_file)

        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            # forward in ``loss`` mode
            packed_inputs = _demo_mm_inputs(
                batch_size=1,
                frame_id=0,
                num_key_imgs=1,
                ref_prefix='search',
                num_items=[1])
            batch_inputs, data_samples = model.data_preprocessor(
                packed_inputs, True)
            losses = model.forward(batch_inputs, data_samples, mode='loss')
            assert isinstance(losses, dict)

    @parameterized.expand([
        ('sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py', ('cpu', 'cuda')),
    ])
    def test_siamese_rpn_forward_predict_mode(self, cfg_file, devices):
        _model = _get_model_cfg(cfg_file)

        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            # forward in ``predict`` mode
            model.eval()
            with torch.no_grad():
                for i in range(3):
                    packed_inputs = _demo_mm_inputs(
                        batch_size=1,
                        frame_id=i,
                        num_key_imgs=1,
                        ref_prefix='search',
                        num_items=[1])
                    batch_inputs, data_samples = model.data_preprocessor(
                        packed_inputs, False)
                    batch_results = model.forward(
                        batch_inputs, data_samples, mode='predict')
                    assert len(batch_results) == 1
                    assert isinstance(batch_results[0], TrackDataSample)
