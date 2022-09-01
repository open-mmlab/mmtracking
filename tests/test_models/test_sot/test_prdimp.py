# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from parameterized import parameterized

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.testing import demo_mm_inputs, get_model_cfg
from mmtrack.utils import register_all_modules


class TestPrDiMP(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    @parameterized.expand(['sot/prdimp/prdimp_r50_8xb10-50e_got10k.py'])
    def test_init(self, cfg_file):
        model = get_model_cfg(cfg_file)

        model = MODELS.build(model)
        assert model.backbone
        assert model.classifier
        assert model.bbox_regressor

    @parameterized.expand(['sot/prdimp/prdimp_r50_8xb10-50e_got10k.py'])
    def test_stark_forward_predict_mode(self, cfg_file):
        if not torch.cuda.is_available():
            return

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
                out_data = model.data_preprocessor(packed_inputs, False)
                batch_results = model.forward(**out_data, mode='predict')
                assert len(batch_results) == 1
                assert isinstance(batch_results[0], TrackDataSample)

    @parameterized.expand(['sot/prdimp/prdimp_r50_8xb10-50e_got10k.py'])
    def test_prdimp_forward_loss_mode(self, cfg_file):
        if not torch.cuda.is_available():
            return
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
        out_data = model.data_preprocessor(packed_inputs, True)
        losses = model.forward(**out_data, mode='loss')
        assert isinstance(losses, dict)
