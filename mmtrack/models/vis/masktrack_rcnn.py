# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import build_detector, build_head

from mmtrack.models.mot import BaseMultiObjectTracker
from ..builder import MODELS, build_tracker


@MODELS.register_module()
class MaskTrackRCNN(BaseMultiObjectTracker):
    """Video Instance Segmentation.

    This video instance segmentor is the implementation of`MaskTrack R-CNN
    <https://arxiv.org/abs/1905.04804>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        init_cfg (dict): Configuration of initialization. Defaults to None.
    """

    def __init__(self,
                 detector=None,
                 track_head=None,
                 tracker=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if detector is not None:
            self.detector = build_detector(detector)
        assert hasattr(self.detector, 'roi_head'), \
            'MaskTrack R-CNN only supports two stage detectors.'

        if track_head is not None:
            self.track_head = build_head(track_head)
        if tracker is not None:
            self.tracker = build_tracker(tracker)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_instance_ids=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ref_gt_instance_ids=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box.

            ref_img (Tensor): of shape (N, C, H, W) encoding input reference
                images. Typically these should be mean centered and std scaled.

            ref_img_metas (list[dict]): list of reference image info dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                reference image with shape (num_gts, 4) in
                [tl_x, tl_y, br_x, br_y] format.

            ref_gt_labels (list[Tensor]): class indices corresponding to each
                box.

            gt_instance_ids (None | list[Tensor]): specify the instance id for
                each ground truth bbox.

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | list[Tensor]) : true segmentation masks for each
                box used if the architecture supports a segmentation task.

            proposals (None | list[Tensor]) : override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

            ref_gt_instance_ids (None | list[Tensor]): specify the instance id
                for each ground truth bbox of reference images.

            ref_gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes of reference images can be ignored when computing the
                loss.

            ref_gt_masks (None | list[Tensor]) : true segmentation masks for
                each box of reference images used if the architecture supports
                a segmentation task.

            ref_proposals (None | list[Tensor]) : override rpn proposals with
                custom proposals of reference images. Use when `with_rpn` is
                False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.detector.extract_feat(img)
        ref_x = self.detector.extract_feat(ref_img)

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            losses_rpn, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(losses_rpn)
        else:
            proposal_list = proposals

        losses_detect = self.detector.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(losses_detect)

        losses_track = self.track_head.forward_train(
            x, ref_x, img_metas, proposal_list, gt_bboxes, ref_gt_bboxes,
            gt_labels, gt_instance_ids, ref_gt_instance_ids, gt_bboxes_ignore,
            **kwargs)
        losses.update(losses_track)

        return losses

    # TODO: Support simple_test
    def simple_test(self, img, img_metas, **kwargs):
        """Test function with a single scale."""
        pass
