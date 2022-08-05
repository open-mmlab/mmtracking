# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from parameterized import parameterized

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.testing import demo_mm_inputs, get_model_cfg
from mmtrack.utils import register_all_modules


class TestPrdimp(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    @parameterized.expand(['sot/prdimp/prdimp_r50_50e_got10k.py'])
    def test_init(self, cfg_file):
        model = get_model_cfg(cfg_file)

        model = MODELS.build(model)
        assert model.backbone
        assert model.classifier
        assert model.bbox_regressor

    @pytest.mark.skipif(
        not torch.cuda.is_available, reason='test case under gpu environment')
    @parameterized.expand(['sot/prdimp/prdimp_r50_50e_got10k.py'])
    def test_prdimp_forward_predict_mode(self, cfg_file):
        _model = get_model_cfg(cfg_file)
        model = MODELS.build(_model)
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
                batch_inputs, data_samples = model.data_preprocessor(
                    packed_inputs, False)
                batch_results = model.forward(
                    batch_inputs, data_samples, mode='predict')
                assert len(batch_results) == 1
                assert isinstance(batch_results[0], TrackDataSample)

    @pytest.mark.skipif(
        not torch.cuda.is_available, reason='test case under gpu environment')
    @parameterized.expand(['sot/prdimp/prdimp_r50_50e_got10k.py'])
    def test_prdimp_forward_loss_mode(self, cfg_file):
        _model = get_model_cfg(cfg_file)
        model = MODELS.build(_model)
        model = model.cuda()

        # forward in ``loss`` mode
        model.train()
        packed_inputs = demo_mm_inputs(
            batch_size=2,
            frame_id=0,
            num_key_imgs=3,
            num_ref_imgs=3,
            image_shapes=[(3, 280, 280), (3, 280, 280)],
            ref_prefix='search',
            num_items=[3, 3])
        batch_inputs, data_samples = model.data_preprocessor(
            packed_inputs, True)
        losses = model.forward(batch_inputs, data_samples, mode='loss')
        assert isinstance(losses, dict)