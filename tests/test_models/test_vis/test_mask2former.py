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


class TestMask2Former(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    @parameterized.expand([
        'vis/mask2former/mask2former_resnet50_8xb2-8e_youtubevis2019.py',
    ])
    def test_mask2former_init(self, cfg_file):
        model = get_model_cfg(cfg_file)

        model = MODELS.build(model)
        assert model.backbone
        assert model.track_head

    @parameterized.expand([
        ('vis/mask2former/mask2former_resnet50_8xb2-8e_youtubevis2019.py',
         ('cpu', 'cuda')),
    ])
    def test_mask2former_forward_loss_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_mask2former_forward_loss_mode-{time.time()}')
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
                num_key_imgs=2,
                num_classes=2,
                with_mask=True)
            batch_inputs, batch_data_samples = model.data_preprocessor(
                packed_inputs, True)
            # Test forward
            batch_data_samples[0].gt_instances[
                'map_instances_to_img_idx'] = torch.tensor([0], device=device)
            losses = model.forward(
                batch_inputs, batch_data_samples, mode='loss')
            assert isinstance(losses, dict)

    @parameterized.expand([
        ('vis/mask2former/mask2former_resnet50_8xb2-8e_youtubevis2019.py',
         ('cpu', 'cuda')),
    ])
    def test_mask2former_forward_predict_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_mask2former_forward_predict_mode-{time.time()}')
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
                batch_size=2,
                frame_id=0,
                image_shapes=[(3, 128, 128), (3, 128, 128)],
                num_classes=2,
                with_mask=True)
            batch_inputs, batch_data_samples = model.data_preprocessor(
                packed_inputs, True)

            # Test forward test
            model.eval()
            with torch.no_grad():
                batch_results = model.forward(
                    batch_inputs, batch_data_samples, mode='predict')
                assert len(batch_results) == 2
