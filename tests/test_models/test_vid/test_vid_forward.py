# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
# for register all models in mmdet
from mmdet.models import *  # noqa
from parameterized import parameterized

# for register all models in mmtrack
from mmtrack.models import *  # noqa
from mmtrack.registry import MODELS
from ..utils import _demo_mm_inputs, _get_model_cfg


class TestVideoDetector(TestCase):

    @parameterized.expand([
        'vid/dff/dff_faster_rcnn_r50_dc5_7e_imagenetvid.py',
    ])
    def test_init(self, cfg_file):
        model = _get_model_cfg(cfg_file)

        model = MODELS.build(model)
        assert model.detector
        assert model.motion
        assert model.device.type == 'cpu'

    @parameterized.expand([
        ('vid/dff/dff_faster_rcnn_r50_dc5_7e_imagenetvid.py', ('cpu', 'cuda')),
    ])
    def test_vid_dff_style_forward_train(self, cfg_file, devices):
        _model = _get_model_cfg(cfg_file)

        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            assert model.device.type == device

            packed_inputs = _demo_mm_inputs(
                batch_size=1, frame_id=0, num_ref_imgs=1)

            # Test forward train
            losses = model.forward(packed_inputs, return_loss=True)
            assert isinstance(losses, dict)

    @parameterized.expand([
        ('vid/dff/dff_faster_rcnn_r50_dc5_7e_imagenetvid.py', ('cpu', 'cuda')),
    ])
    def test_single_stage_forward_test(self, cfg_file, devices):
        _model = _get_model_cfg(cfg_file)
        _model.test_cfg.key_frame_interval = 2

        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            assert model.device.type == device

            packed_inputs = _demo_mm_inputs(
                batch_size=1, frame_id=0, num_ref_imgs=0)

            # Test forward test
            model.eval()
            with torch.no_grad():
                for i in range(3):
                    packed_inputs = _demo_mm_inputs(
                        batch_size=1, frame_id=i, num_ref_imgs=0)
                    batch_results = model.forward(
                        packed_inputs, return_loss=False)
                    assert len(batch_results) == 1
