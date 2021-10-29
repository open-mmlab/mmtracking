# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import pytest
import torch

from .utils import _demo_mm_inputs, _get_config_module


@pytest.mark.parametrize(
    'cfg_file',
    ['vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py'])
def test_vis_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)

    from mmtrack.models import build_model
    vis = build_model(model)

    # Test forward train with a non-empty truth batch
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10], with_track=True)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_instance_ids = mm_inputs['gt_instance_ids']
    gt_masks = mm_inputs['gt_masks']

    ref_input_shape = (1, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(
        ref_input_shape, num_items=[11], with_track=True)
    ref_img = ref_mm_inputs.pop('imgs')
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']
    ref_gt_instance_ids = ref_mm_inputs['gt_instance_ids']

    losses = vis.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_img,
        ref_img_metas=ref_img_metas,
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_instance_ids=gt_instance_ids,
        gt_masks=gt_masks,
        ref_gt_instance_ids=ref_gt_instance_ids,
        ref_gt_masks=ref_gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = vis._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0], with_track=True)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_instance_ids = mm_inputs['gt_instance_ids']
    gt_masks = mm_inputs['gt_masks']

    ref_input_shape = (1, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(
        ref_input_shape, num_items=[0], with_track=True)
    ref_img = ref_mm_inputs.pop('imgs')
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']
    ref_gt_instance_ids = ref_mm_inputs['gt_instance_ids']

    losses = vis.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_img,
        ref_img_metas=ref_img_metas,
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_instance_ids=gt_instance_ids,
        gt_masks=gt_masks,
        ref_gt_instance_ids=ref_gt_instance_ids,
        ref_gt_masks=ref_gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = vis._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward test
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img2_metas = copy.deepcopy(img_metas)
        img2_metas[0]['frame_id'] = 1
        img_metas.extend(img2_metas)
        results = defaultdict(list)
        for one_img, one_meta in zip(img_list, img_metas):
            result = vis.forward([one_img], [[one_meta]],
                                 rescale=True,
                                 return_loss=False)
            for k, v in result.items():
                results[k].append(v)
