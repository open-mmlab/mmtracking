# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest
from unittest import TestCase

import torch
from mmengine.logging import MessageHub
from parameterized import parameterized

from mmtrack.registry import MODELS
from mmtrack.testing import demo_mm_inputs, get_model_cfg
from mmtrack.utils import register_all_modules


class TestByteTrack(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    @parameterized.expand([
        'mot/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_'
        'test-mot17halfval.py',
    ])
    def test_bytetrack_init(self, cfg_file):
        model = get_model_cfg(cfg_file)
        model.detector.neck.out_channels = 1
        model.detector.bbox_head.in_channels = 1
        model.detector.bbox_head.feat_channels = 1

        model = MODELS.build(model)
        assert model.detector
        assert model.motion

    @parameterized.expand([
        ('mot/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_'
         'test-mot17halfval.py', ('cpu', 'cuda')),
    ])
    def test_bytetrack_forward_loss_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_bytetrack_forward_loss_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)
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
                batch_size=1, frame_id=0, num_ref_imgs=0, num_classes=1)
            batch_inputs, batch_data_samples = model.data_preprocessor(
                packed_inputs, True)
            # Test forward
            losses = model.forward(
                batch_inputs, batch_data_samples, mode='loss')
            assert isinstance(losses, dict)

    @parameterized.expand([
        ('mot/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_'
         'test-mot17halfval.py', ('cpu', 'cuda')),
    ])
    def test_bytetrack_forward_predict_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_bytetrack_forward_predict_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)

        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            _model = get_model_cfg(cfg_file)
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = demo_mm_inputs(
                batch_size=1, frame_id=0, num_ref_imgs=0, num_classes=1)
            batch_inputs, batch_data_samples = model.data_preprocessor(
                packed_inputs, True)

            # Test forward test
            model.eval()
            with torch.no_grad():
                batch_results = model.forward(
                    batch_inputs, batch_data_samples, mode='predict')
                assert len(batch_results) == 1
