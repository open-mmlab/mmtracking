# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

from mmcv.runner import BaseModule
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor


@HEADS.register_module()
class MaskTrackRCNNTrackRoIHead(BaseModule, metaclass=ABCMeta):
    """The track roi head of MaskTrack R-CNN.

    This module is proposed in `MaskTrack R-CNN
    <https://arxiv.org/abs/1905.04804>`_.

    Args:
        bbox_roi_extractor (dict): Configuration of roi extractor.
        track_head (dict): Configuration of track head.
        train_cfg (dict): Configuration when training.
        test_cfg (dict): Configuration when testing.
        init_cfg (dict): Configuration of initialization.
    """

    def __init__(self,
                 bbox_roi_extractor=None,
                 track_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if track_head is not None:
            self.init_track_head(bbox_roi_extractor, track_head)

        self.init_assigner_sampler()

    def init_track_head(self, bbox_roi_extractor, track_head):
        """Initialize ``track_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.track_head = build_head(track_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    @property
    def with_track(self):
        """bool: whether the mulit-object tracker has a track head"""
        return hasattr(self, 'track_head') and self.track_head is not None

    def forward_train(self,
                      x,
                      ref_x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      ref_gt_bboxes,
                      gt_labels,
                      gt_instance_ids,
                      ref_gt_instance_ids,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level image features.

            ref_x (list[Tensor]): list of multi-level ref_img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            ref_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                reference image with shape (num_gts, 4) in
                [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box.

            gt_instance_ids (None | list[Tensor]): specify the instance id for
                each ground truth bbox.

            ref_gt_instance_ids (None | list[Tensor]): specify the instance id
                for each ground truth bbox of reference images.

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_track:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()

        if self.with_track:
            track_results = self._track_forward_train(x, ref_x,
                                                      sampling_results,
                                                      ref_gt_bboxes,
                                                      gt_instance_ids,
                                                      ref_gt_instance_ids)
            losses.update(track_results['loss_track'])

        return losses

    def _track_forward_train(self, x, ref_x, sampling_results, ref_gt_bboxes,
                             gt_instance_ids, ref_gt_instance_ids, **kwargs):
        """Run forward function and calculate loss for track head in
        training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        ref_rois = bbox2roi(ref_gt_bboxes)
        ref_bbox_feats = self.bbox_roi_extractor(
            ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)

        num_bbox_per_img = [len(res.bboxes) for res in sampling_results]
        num_bbox_per_ref_img = [
            len(ref_gt_bbox) for ref_gt_bbox in ref_gt_bboxes
        ]

        similarity_scores = self.track_head(bbox_feats, ref_bbox_feats,
                                            num_bbox_per_img,
                                            num_bbox_per_ref_img)

        track_targets = self.track_head.get_targets(sampling_results,
                                                    gt_instance_ids,
                                                    ref_gt_instance_ids)
        loss_track = self.track_head.loss(similarity_scores, *track_targets)
        track_results = dict(loss_track=loss_track)

        return track_results
