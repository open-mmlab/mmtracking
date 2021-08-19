# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler

from mmtrack.models.roi_heads.bbox_heads import SelsaBBoxHead


def test_selsa_bbox_head_loss():
    """Tests selsa_bbox_head loss when truth is empty and non-empty."""
    selsa_bbox_head_config = dict(
        num_shared_fcs=2,
        in_channels=8,
        fc_out_channels=16,
        roi_feat_size=3,
        aggregator=dict(
            type='SelsaAggregator', in_channels=16, num_attention_blocks=4))
    self = SelsaBBoxHead(**selsa_bbox_head_config)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    target_cfg = mmcv.Config(dict(pos_weight=1))

    # Test bbox loss when truth is empty
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    rois = bbox2roi([res.bboxes for res in sampling_results])
    dummy_feats = torch.rand(num_sampled, 8, 3, 3)
    ref_dummy_feats = torch.rand(2 * num_sampled, 8, 3, 3)
    cls_scores, bbox_preds = self.forward(dummy_feats, ref_dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) == 0, 'empty gt loss should be zero'

    # Test bbox loss when truth is non-empty
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)
    rois = bbox2roi([res.bboxes for res in sampling_results])

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8, 3, 3)
    ref_dummy_feats = torch.rand(2 * num_sampled, 8, 3, 3)
    cls_scores, bbox_preds = self.forward(dummy_feats, ref_dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) > 0, 'box-loss should be non-zero'


def _dummy_bbox_sampling(proposal_list, gt_bboxes, gt_labels):
    """Create sample results that can be passed to BBoxHead.get_targets."""
    num_imgs = 1
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
        add_gt_as_proposals=True)
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
