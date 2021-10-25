# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

from mmcv.runner import BaseModule
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor


@HEADS.register_module()
class RoITrackHead(BaseModule, metaclass=ABCMeta):
    """The roi track head.

    This module is used in multi-object tracking methods, such as MaskTrack
    R-CNN.

    Args:
        roi_extractor (dict): Configuration of roi extractor. Defaults to None.
        embed_head (dict): Configuration of embed head. Defaults to None.
        train_cfg (dict): Configuration when training. Defaults to None.
        test_cfg (dict): Configuration when testing. Defaults to None.
        init_cfg (dict): Configuration of initialization. Defaults to None.
    """

    def __init__(self,
                 roi_extractor=None,
                 embed_head=None,
                 regress_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if embed_head is not None:
            self.init_embed_head(roi_extractor, embed_head)

        if regress_head is not None:
            raise NotImplementedError('Regression head is not supported yet.')

        self.init_assigner_sampler()

    def init_embed_head(self, roi_extractor, embed_head):
        """Initialize ``embed_head``"""
        self.roi_extractor = build_roi_extractor(roi_extractor)
        self.embed_head = build_head(embed_head)

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
        """bool: whether the mulit-object tracker has a embed head"""
        return hasattr(self, 'embed_head') and self.embed_head is not None

    def extract_roi_feats(self, x, bboxes):
        """Extract roi features."""
        rois = bbox2roi(bboxes)
        bbox_feats = self.roi_extractor(x[:self.roi_extractor.num_inputs],
                                        rois)
        num_bbox_per_img = [len(bbox) for bbox in bboxes]
        return bbox_feats, num_bbox_per_img

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
        bboxes = [res.bboxes for res in sampling_results]
        bbox_feats, num_bbox_per_img = self.extract_roi_feats(x, bboxes)
        ref_bbox_feats, num_bbox_per_ref_img = self.extract_roi_feats(
            ref_x, ref_gt_bboxes)

        similarity_logits = self.embed_head(bbox_feats, ref_bbox_feats,
                                            num_bbox_per_img,
                                            num_bbox_per_ref_img)

        track_targets = self.embed_head.get_targets(sampling_results,
                                                    gt_instance_ids,
                                                    ref_gt_instance_ids)
        loss_track = self.embed_head.loss(similarity_logits, *track_targets)
        track_results = dict(loss_track=loss_track)

        return track_results

    def simple_test(self, roi_feats, prev_roi_feats):
        """Test without augmentations."""
        return self.embed_head(roi_feats, prev_roi_feats, [roi_feats.shape[0]],
                               [prev_roi_feats.shape[0]])[0]
