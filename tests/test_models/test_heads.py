import mmcv
import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler

from mmtrack.models.roi_heads.bbox_heads import SelsaBBoxHead
from mmtrack.models.track_heads import CorrelationHead, SiameseRPNHead


def test_correlation_head():
    self = CorrelationHead(16, 16, 2)
    kernel = torch.rand(1, 16, 7, 7)
    search = torch.rand(1, 16, 31, 31)
    out = self(kernel, search)
    assert out.size() == (1, 2, 25, 25)


def test_siamese_rpn_head_loss():
    """Tests siamese rpn head loss when truth is non-empty."""
    cfg = mmcv.Config(
        dict(
            anchor_generator=dict(
                type='SiameseRPNAnchorGenerator',
                strides=[8],
                ratios=[0.33, 0.5, 1, 2, 3],
                scales=[8]),
            in_channels=[16, 16, 16],
            weighted_sum=True,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[1., 1., 1., 1.]),
            loss_cls=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=1.2),
            train_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.6,
                    match_low_quality=False),
                sampler=dict(
                    type='RandomSampler',
                    num=64,
                    pos_fraction=0.25,
                    add_gt_as_proposals=False),
                num_neg=16,
                exemplar_size=127,
                search_size=255),
            test_cfg=dict(penalty_k=0.05, window_influence=0.42, lr=0.38)))

    self = SiameseRPNHead(**cfg)

    z_feats = tuple(
        [torch.rand(1, 16, 7, 7) for i in range(len(self.cls_heads))])
    x_feats = tuple(
        [torch.rand(1, 16, 31, 31) for i in range(len(self.cls_heads))])
    cls_score, bbox_pred = self.forward(z_feats, x_feats)

    gt_bboxes = [
        torch.Tensor([[0., 23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    bbox_targets = self.get_targets(gt_bboxes, cls_score.shape[2:], [True])
    gt_losses = self.loss(cls_score, bbox_pred, *bbox_targets)
    assert gt_losses['loss_rpn_cls'] > 0, 'cls loss should be non-zero'
    assert gt_losses[
        'loss_rpn_bbox'] >= 0, 'box loss should be non-zero or zero'

    gt_bboxes = [
        torch.Tensor([[0., 23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    bbox_targets = self.get_targets(gt_bboxes, cls_score.shape[2:], [False])
    gt_losses = self.loss(cls_score, bbox_pred, *bbox_targets)
    assert gt_losses['loss_rpn_cls'] > 0, 'cls loss should be non-zero'
    assert gt_losses['loss_rpn_bbox'] == 0, 'box loss should be zero'


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
