# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import pytest
import torch

from .utils import _demo_mm_inputs, _get_config_module


@pytest.mark.parametrize('cfg_file', [
    'sot/siamese_rpn/siamese_rpn_r50_1x_lasot.py',
    'sot/siamese_rpn/siamese_rpn_r50_1x_vot2018.py'
])
def test_sot_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)

    from mmtrack.models import build_model
    sot = build_model(model)

    # Test forward train with a non-empty truth batch
    input_shape = (1, 3, 127, 127)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[1])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']

    search_input_shape = (1, 3, 255, 255)
    search_mm_inputs = _demo_mm_inputs(search_input_shape, num_items=[1])
    search_img = search_mm_inputs.pop('imgs')[None]
    search_img_metas = search_mm_inputs.pop('img_metas')
    search_gt_bboxes = search_mm_inputs['gt_bboxes']
    img_inds = search_gt_bboxes[0].new_full((search_gt_bboxes[0].size(0), 1),
                                            0)
    search_gt_bboxes[0] = torch.cat((img_inds, search_gt_bboxes[0]), dim=1)

    losses = sot.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        search_img=search_img,
        search_img_metas=search_img_metas,
        search_gt_bboxes=search_gt_bboxes,
        is_positive_pairs=[True],
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = sot._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    losses = sot.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        search_img=search_img,
        search_img_metas=search_img_metas,
        search_gt_bboxes=search_gt_bboxes,
        is_positive_pairs=[False],
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = sot._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward test
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img_metas.extend(copy.deepcopy(img_metas))
        for i in range(len(img_metas)):
            img_metas[i]['frame_id'] = i
        gt_bboxes.extend(copy.deepcopy(gt_bboxes))
        results = defaultdict(list)
        for one_img, one_meta, one_gt_bboxes in zip(img_list, img_metas,
                                                    gt_bboxes):
            result = sot.forward([one_img], [[one_meta]],
                                 gt_bboxes=[one_gt_bboxes],
                                 return_loss=False)
            for k, v in result.items():
                results[k].append(v)
