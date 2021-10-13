# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import pytest
import torch

from .utils import _demo_mm_inputs, _get_config_module


@pytest.mark.parametrize('cfg_file', [
    'mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py',
    'mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
])
def test_mot_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)

    from mmtrack.models import build_model
    mot = build_model(model)
    mot.eval()

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10], with_track=True)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img2_metas = copy.deepcopy(img_metas)
        img2_metas[0]['frame_id'] = 1
        img_metas.extend(img2_metas)
        results = defaultdict(list)
        for one_img, one_meta in zip(img_list, img_metas):
            result = mot.forward([one_img], [[one_meta]], return_loss=False)
            for k, v in result.items():
                results[k].append(v)
