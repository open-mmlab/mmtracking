# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmdet.core import build_assigner, build_sampler

from mmtrack.models.track_heads import QuasiDenseEmbedHead


def test_quasi_dense_embed_head():
    cfg = mmcv.Config(
        dict(
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)))

    self = QuasiDenseEmbedHead(**cfg)

    gt_match_indices = [torch.tensor([0, 1])]
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874],
                      [23.6667, 23.8757, 228.6326, 153.8874]])
    ]
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874],
                      [23.6667, 23.8757, 228.6326, 153.8874]])
    ]
    gt_labels = [torch.LongTensor([1, 1])]

    feats = torch.rand(2, 256, 7, 7)
    key_sampling_results = _dummy_bbox_sampling(feats, proposal_list,
                                                gt_bboxes, gt_labels)
    ref_sampling_results = key_sampling_results

    key_embeds = self.forward(feats)
    ref_embeds = key_embeds

    match_feats = self.match(key_embeds, ref_embeds, key_sampling_results,
                             ref_sampling_results)
    asso_targets = self.get_targets(gt_match_indices, key_sampling_results,
                                    ref_sampling_results)
    loss_track = self.loss(*match_feats, *asso_targets)
    assert loss_track['loss_track'] >= 0, 'track loss should be zero'
    assert loss_track['loss_track_aux'] > 0, 'aux loss should be non-zero'


def _dummy_bbox_sampling(feats, proposal_list, gt_bboxes, gt_labels):
    """Create sample results that can be passed to Head.get_targets."""
    num_imgs = len(proposal_list)
    assign_config = dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        ignore_iof_thr=-1)
    sampler_config = dict(
        type='CombinedSampler',
        num=4,
        pos_fraction=0.5,
        neg_pos_ub=3,
        add_gt_as_proposals=True,
        pos_sampler=dict(type='InstanceBalancedPosSampler'),
        neg_sampler=dict(type='RandomSampler'))
    bbox_assigner = build_assigner(assign_config)
    bbox_sampler = build_sampler(sampler_config)
    gt_bboxes_ignore = [None for _ in range(num_imgs)]
    sampling_results = []
    for i in range(num_imgs):
        assign_result = bbox_assigner.assign(proposal_list[i], gt_bboxes[i],
                                             gt_bboxes_ignore[i], gt_labels[i])
        sampling_result = bbox_sampler.sample(
            assign_result,
            proposal_list[i],
            gt_bboxes[i],
            gt_labels[i],
            feats=feats)
        sampling_results.append(sampling_result)

    return sampling_results
