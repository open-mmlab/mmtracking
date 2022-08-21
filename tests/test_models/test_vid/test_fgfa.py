# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.testing import demo_mm_inputs, get_model_cfg
from mmtrack.utils import register_all_modules


class TestVideoDetector(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules()

    @parameterized.expand([
        'vid/fgfa/fgfa_faster-rcnn-resnet50-dc5_8x1bs-7e_imagenetvid.py',
    ])
    def test_init(self, cfg_file):
        model = get_model_cfg(cfg_file)
        model = MODELS.build(model)
        assert model.detector
        assert model.motion

    @parameterized.expand([
        ('vid/fgfa/fgfa_faster-rcnn-resnet50-dc5_8x1bs-7e_imagenetvid.py',
         ('cpu', 'cuda'))
    ])
    def test_fgfa_forward_loss_mode(self, cfg_file, devices):
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            _model = get_model_cfg(cfg_file)
            # _scope_ will be popped after build
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = demo_mm_inputs(
                batch_size=1, frame_id=0, num_ref_imgs=2)
            batch_inputs, data_samples = model.data_preprocessor(
                packed_inputs, True)

            # forward in ``loss`` mode
            losses = model.forward(batch_inputs, data_samples, mode='loss')
            assert isinstance(losses, dict)

    @parameterized.expand([
        ('vid/fgfa/fgfa_faster-rcnn-resnet50-dc5_8x1bs-7e_imagenetvid.py',
         ('cpu', 'cuda'))
    ])
    def test_fgfa_forward_predict_mode(self, cfg_file, devices):
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            _model = get_model_cfg(cfg_file)
            # _scope_ will be popped after build
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            # forward in ``predict`` mode
            model.eval()
            with torch.no_grad():
                for i in range(3):
                    packed_inputs = demo_mm_inputs(
                        batch_size=1, frame_id=i, num_ref_imgs=2)
                    batch_inputs, data_samples = model.data_preprocessor(
                        packed_inputs, False)
                    batch_results = model.forward(
                        batch_inputs, data_samples, mode='predict')
                    assert len(batch_results) == 1
                    assert isinstance(batch_results[0], TrackDataSample)
