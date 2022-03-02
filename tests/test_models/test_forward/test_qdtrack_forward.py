# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch

from mmtrack.datasets.pipelines.processing import MatchInstances
from .utils import _demo_mm_inputs, _get_config_module


@pytest.mark.parametrize('cfg_file', [
    'mot/qdtrack/MOT17/qdtrack_frcnn_r101_fpn_4e_mot17.py',
    'mot/qdtrack/MOT17/qdtrack_frcnn_r101_fpn_4e_mot17_crowdhuman.py'
])
def test_qdtrack_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)

    from mmtrack.models import build_model
    qdtrack = build_model(model)

    # Test forward train with a non-empty truth batch
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(
        input_shape, num_items=[10], num_classes=2, with_track=True)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_instance_ids = mm_inputs['gt_instance_ids']
    gt_masks = mm_inputs['gt_masks']

    ref_input_shape = (1, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(
        ref_input_shape, num_items=[10], num_classes=2, with_track=True)
    ref_img = ref_mm_inputs.pop('imgs')
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']
    ref_gt_instance_ids = ref_mm_inputs['gt_instance_ids']

    match_tool = MatchInstances()
    gt_match_indices, _ = match_tool._match_gts(gt_instance_ids[0],
                                                ref_gt_instance_ids[0])
    gt_match_indices = [torch.tensor(gt_match_indices)]

    losses = qdtrack.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        gt_match_indices=gt_match_indices,
        ref_img=ref_img,
        ref_img_metas=ref_img_metas,
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        ref_gt_masks=ref_gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = qdtrack._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(
        input_shape, num_items=[0], num_classes=2, with_track=True)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_instance_ids = mm_inputs['gt_instance_ids']
    gt_masks = mm_inputs['gt_masks']

    ref_mm_inputs = _demo_mm_inputs(
        ref_input_shape, num_items=[0], num_classes=2, with_track=True)
    ref_img = ref_mm_inputs.pop('imgs')
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']
    ref_gt_instance_ids = ref_mm_inputs['gt_instance_ids']

    gt_match_indices, _ = match_tool._match_gts(gt_instance_ids[0],
                                                ref_gt_instance_ids[0])
    gt_match_indices = [torch.tensor(gt_match_indices)]

    losses = qdtrack.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        gt_match_indices=gt_match_indices,
        ref_img=ref_img,
        ref_img_metas=ref_img_metas,
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        ref_gt_masks=ref_gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    assert torch.isnan(losses['loss_track'])
    assert torch.isnan(losses['loss_track_aux'])
    losses.pop('loss_track')
    losses.pop('loss_track_aux')
    loss, _ = qdtrack._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()
