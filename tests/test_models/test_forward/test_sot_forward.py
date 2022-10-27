# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import pytest
import torch

from mmtrack.models import build_model
from .utils import _demo_mm_inputs, _get_config_module


@pytest.mark.parametrize('cfg_file', [
    'sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py',
    'sot/siamese_rpn/siamese_rpn_r50_20e_vot2018.py'
])
def test_siamrpn_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)

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


def test_stark_forward():
    # test stage-1 forward
    config = _get_config_module('sot/stark/stark_st1_r50_500e_got10k.py')
    model = copy.deepcopy(config.model)

    from mmtrack.models import build_model
    sot = build_model(model)

    # Test forward train with a non-empty truth batch
    input_shape = (2, 3, 128, 128)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[1, 1])
    imgs = mm_inputs.pop('imgs')[None]
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    padding_mask = torch.zeros((2, 128, 128), dtype=bool)
    padding_mask[0, 100:128, 100:128] = 1
    padding_mask = padding_mask[None]

    search_input_shape = (1, 3, 320, 320)
    search_mm_inputs = _demo_mm_inputs(search_input_shape, num_items=[1])
    search_img = search_mm_inputs.pop('imgs')[None]
    search_img_metas = search_mm_inputs.pop('img_metas')
    search_gt_bboxes = search_mm_inputs['gt_bboxes']
    search_padding_mask = torch.zeros((1, 320, 320), dtype=bool)
    search_padding_mask[0, 0:20, 0:20] = 1
    search_padding_mask = search_padding_mask[None]
    img_inds = search_gt_bboxes[0].new_full((search_gt_bboxes[0].size(0), 1),
                                            0)
    search_gt_bboxes[0] = torch.cat((img_inds, search_gt_bboxes[0]), dim=1)

    losses = sot.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        padding_mask=padding_mask,
        search_img=search_img,
        search_img_metas=search_img_metas,
        search_gt_bboxes=search_gt_bboxes,
        search_padding_mask=search_padding_mask,
        return_loss=True)
    assert isinstance(losses, dict)
    assert losses['loss_bbox'] > 0
    loss, _ = sot._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # test stage-2 forward
    config = _get_config_module('sot/stark/stark_st2_r50_50e_got10k.py')
    model = copy.deepcopy(config.model)
    sot = build_model(model)
    search_gt_labels = [torch.ones((1, 2))]

    losses = sot.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        padding_mask=padding_mask,
        search_img=search_img,
        search_img_metas=search_img_metas,
        search_gt_bboxes=search_gt_bboxes,
        search_padding_mask=search_padding_mask,
        search_gt_labels=search_gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    assert losses['loss_cls'] > 0
    loss, _ = sot._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()


@pytest.mark.parametrize('cfg_file', [
    'sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py',
    'sot/siamese_rpn/siamese_rpn_r50_20e_vot2018.py',
    'sot/stark/stark_st2_r50_50e_got10k.py',
    'sot/mixformer/mixformer_cvt_500e_got10k.py'
])
def test_sot_test_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)
    sot = build_model(model)
    sot.eval()

    device = torch.device('cpu')
    if config.model.type == 'MixFormer':
        if not torch.cuda.is_available():
            return
        else:
            device = torch.device('cuda')
    sot = sot.to(device)

    input_shape = (1, 3, 127, 127)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[1])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']

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
            one_img = one_img.to(device)
            one_gt_bboxes = one_gt_bboxes.to(device)
            result = sot.forward([one_img], [[one_meta]],
                                 gt_bboxes=[one_gt_bboxes],
                                 return_loss=False)
            for k, v in result.items():
                results[k].append(v)
