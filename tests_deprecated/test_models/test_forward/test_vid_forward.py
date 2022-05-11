# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import pytest
import torch

from .utils import _demo_mm_inputs, _get_config_module


@pytest.mark.parametrize(
    'cfg_file', ['vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py'])
def test_vid_dff_style_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)

    from mmtrack.models import build_model
    vid = build_model(model)

    # Test forward train with a non-empty truth batch
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']

    ref_input_shape = (1, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[11])
    ref_img = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']

    losses = vid.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_img,
        ref_img_metas=ref_img_metas,
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_masks=gt_masks,
        ref_gt_masks=ref_gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = vid._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']

    ref_input_shape = (1, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[0])
    ref_img = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']

    losses = vid.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_img,
        ref_img_metas=ref_img_metas,
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_masks=gt_masks,
        ref_gt_masks=ref_gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = vid._parse_losses(losses)
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
        results = defaultdict(list)
        for one_img, one_meta in zip(img_list, img_metas):
            result = vid.forward([one_img], [[one_meta]], return_loss=False)
            for k, v in result.items():
                results[k].append(v)


@pytest.mark.parametrize('cfg_file', [
    'vid/fgfa/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid.py',
    'vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid.py',
    'vid/temporal_roi_align/'
    'selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid.py',
])
def test_vid_fgfa_style_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)

    from mmtrack.models import build_model
    vid = build_model(model)

    # Test forward train with a non-empty truth batch
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']

    ref_input_shape = (2, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[9, 11])
    ref_img = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_img_metas[1]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']

    losses = vid.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_img,
        ref_img_metas=[ref_img_metas],
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_masks=gt_masks,
        ref_gt_masks=ref_gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = vid._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']

    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[0, 0])
    ref_imgs = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_img_metas[1]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']

    losses = vid.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_imgs,
        ref_img_metas=[ref_img_metas],
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_masks=gt_masks,
        ref_gt_masks=ref_gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = vid._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward test with frame_stride=1 and frame_range=[-1,0]
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img_metas.extend(copy.deepcopy(img_metas))
        for i in range(len(img_metas)):
            img_metas[i]['frame_id'] = i
            img_metas[i]['num_left_ref_imgs'] = 1
            img_metas[i]['frame_stride'] = 1
        ref_imgs = [ref_imgs.clone(), imgs[[0]][None].clone()]
        ref_img_metas = [
            copy.deepcopy(ref_img_metas),
            copy.deepcopy([img_metas[0]])
        ]
        results = defaultdict(list)
        for one_img, one_meta, ref_img, ref_img_meta in zip(
                img_list, img_metas, ref_imgs, ref_img_metas):
            result = vid.forward([one_img], [[one_meta]],
                                 ref_img=[ref_img],
                                 ref_img_metas=[[ref_img_meta]],
                                 return_loss=False)
            for k, v in result.items():
                results[k].append(v)
