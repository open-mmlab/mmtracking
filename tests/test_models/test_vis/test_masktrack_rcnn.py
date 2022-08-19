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


class TestMaskTrackRCNN(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    @parameterized.expand([
        'vis/masktrack_rcnn/masktrack-rcnn_resnet50-fpn_8x1bs-12e_youtubevis2019.py',  # noqa: E501
    ])
    def test_mask_track_rcnn_init(self, cfg_file):
        model = get_model_cfg(cfg_file)

        model = MODELS.build(model)
        assert model.detector
        assert model.track_head
        assert model.tracker

    @parameterized.expand([
        (
            'vis/masktrack_rcnn/masktrack-rcnn_resnet50-fpn_8x1bs-12e_youtubevis2019.py',  # noqa: E501
            ('cpu', 'cuda')),
    ])
    def test_mask_track_rcnn_forward_loss_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_mask_track_rcnn_forward_loss_mode-{time.time()}')
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
                batch_size=1,
                frame_id=0,
                num_ref_imgs=1,
                num_classes=1,
                with_mask=True)
            batch_inputs, batch_data_samples = model.data_preprocessor(
                packed_inputs, True)
            # Test forward
            losses = model.forward(
                batch_inputs, batch_data_samples, mode='loss')
            assert isinstance(losses, dict)

    @parameterized.expand([
        (
            'vis/masktrack_rcnn/masktrack-rcnn_resnet50-fpn_8x1bs-12e_youtubevis2019.py',  # noqa: E501
            ('cpu', 'cuda')),
    ])
    def test_mask_track_rcnn_forward_predict_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_mask_track_rcnn_forward_predict_mode-{time.time()}')
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
                batch_size=1,
                frame_id=0,
                num_ref_imgs=1,
                num_classes=1,
                with_mask=True)
            batch_inputs, batch_data_samples = model.data_preprocessor(
                packed_inputs, True)

            # Test forward test
            model.eval()
            with torch.no_grad():
                batch_results = model.forward(
                    batch_inputs, batch_data_samples, mode='predict')
                assert len(batch_results) == 1
