# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.testing import demo_mm_inputs, get_model_cfg
from mmtrack.utils import register_all_modules


class TestStark(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    @parameterized.expand([
        'sot/stark/stark-st1_r50_8xb16-500e_got10k.py',
        'sot/stark/stark-st2_r50_8xb16-50e_got10k.py'
    ])
    def test_init(self, cfg_file):
        model = get_model_cfg(cfg_file)

        model = MODELS.build(model)
        assert model.backbone
        assert model.neck
        assert model.head

    # TODO: reduce the channels of models in all configs for speed up
    # unit test.
    @parameterized.expand([
        ('sot/stark/stark-st1_r50_8xb16-500e_got10k.py', ('cpu', 'cuda')),
        ('sot/stark/stark-st2_r50_8xb16-50e_got10k.py', ('cpu', 'cuda'))
    ])
    def test_stark_forward_loss_mode(self, cfg_file, devices):
        _model = get_model_cfg(cfg_file)

        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            # forward in ``loss`` mode
            packed_inputs = demo_mm_inputs(
                batch_size=1,
                frame_id=0,
                num_template_imgs=2,
                num_search_imgs=1,
                ref_prefix='search',
                image_shapes=[[(3, 128, 128), (3, 320, 320)]],
                num_items=[1])
            for data_sample in packed_inputs['data_samples']:
                data_sample.padding_mask = torch.zeros((2, 128, 128),
                                                       dtype=bool)
                data_sample.search_padding_mask = torch.zeros((1, 128, 128),
                                                              dtype=bool)
            out_data = model.data_preprocessor(packed_inputs, True)
            losses = model.forward(**out_data, mode='loss')
            assert isinstance(losses, dict)

    @parameterized.expand([
        ('sot/stark/stark-st1_r50_8xb16-500e_got10k.py', ('cpu', 'cuda')),
        ('sot/stark/stark-st2_r50_8xb16-50e_got10k.py', ('cpu', 'cuda'))
    ])
    def test_stark_forward_predict_mode(self, cfg_file, devices):
        _model = get_model_cfg(cfg_file)

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
                    packed_inputs = demo_mm_inputs(
                        batch_size=1,
                        frame_id=i,
                        num_key_imgs=1,
                        num_ref_imgs=0,
                        image_shapes=[(3, 320, 320)],
                        num_items=[1])
                    for data_sample in packed_inputs['data_samples']:
                        data_sample.padding_mask = torch.zeros((1, 320, 320),
                                                               dtype=bool)
                    out_data = model.data_preprocessor(packed_inputs, False)
                    batch_results = model.forward(**out_data, mode='predict')
                    assert len(batch_results) == 1
                    assert isinstance(batch_results[0], TrackDataSample)
