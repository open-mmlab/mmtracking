# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmdet.core import build_assigner, build_sampler

from mmtrack.models.track_heads import RoIEmbedHead


def test_roi_embed_head_loss():
    """Test roi embed head loss when truth is non-empty."""
    cfg = mmcv.Config(
        dict(
            num_convs=2,
            num_fcs=2,
            roi_feat_size=7,
            in_channels=16,
            fc_out_channels=32))

    self = RoIEmbedHead(**cfg)

    x = torch.rand(2, 16, 7, 7)
    ref_x = torch.rand(2, 16, 7, 7)
    num_x_per_img = [1, 1]
    num_x_per_ref_img = [1, 1]
    similarity_scores = self.forward(x, ref_x, num_x_per_img,
                                     num_x_per_ref_img)

    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2]), torch.LongTensor([2])]
    gt_instance_ids = [torch.LongTensor([2]), torch.LongTensor([2])]
    ref_gt_instance_ids = [torch.LongTensor([2]), torch.LongTensor([2])]
    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    track_targets = self.get_targets(sampling_results, gt_instance_ids,
                                     ref_gt_instance_ids)
    gt_losses = self.loss(similarity_scores, *track_targets)
    assert gt_losses['loss_match'] > 0, 'match loss should be non-zero'
    assert gt_losses[
        'match_accuracy'] >= 0, 'match accuracy should be non-zero or zero'


def _dummy_bbox_sampling(proposal_list, gt_bboxes, gt_labels):
    """Create sample results that can be passed to Head.get_targets."""
    num_imgs = len(proposal_list)
    feat = torch.rand(1, 1, 3, 3)
    assign_config = dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        ignore_iof_thr=-1)
    sampler_config = dict(
        type='RandomSampler',
        num=512,
        pos_fraction=0.25,
        neg_pos_ub=-1,
        add_gt_as_proposals=False)
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
            feats=feat)
        sampling_results.append(sampling_result)

    return sampling_results
