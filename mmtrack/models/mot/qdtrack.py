# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models import build_detector, build_head

from mmtrack.core import outs2results, results2outs
from mmtrack.models.mot import BaseMultiObjectTracker
from ..builder import MODELS, build_tracker


@MODELS.register_module()
class QDTrack(BaseMultiObjectTracker):
    """Quasi-Dense Similarity Learning for Multiple Object Tracking.

    This multi object tracker is the implementation of `QDTrack
    <https://arxiv.org/abs/2006.06664>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        freeze_detector (bool): If True, freeze the detector weights.
            Defaults to False.
    """

    def __init__(self,
                 detector=None,
                 track_head=None,
                 tracker=None,
                 freeze_detector=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if detector is not None:
            self.detector = build_detector(detector)

        if track_head is not None:
            self.track_head = build_head(track_head)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.freeze_detector = freeze_detector
        if self.freeze_detector:
            self.freeze_module('detector')

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        """Forward function during training.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to
                ref_gt_instance_ids of the same tracklet in a pair of images.
            ref_img (Tensor): of shape (N, C, H, W) encoding input reference
                images. Typically these should be mean centered and std scaled.
            ref_img_metas (list[dict]): list of reference image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape',
                and 'img_norm_cfg'.
            ref_gt_bboxes (list[Tensor]): Ground truth bboxes of the
                reference image, each item has a shape (num_gts, 4).
            ref_gt_labels (list[Tensor]): Ground truth labels of all
                reference images, each has a shape (num_gts,).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            ref_gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes
                of reference images to be ignored,
                each item has a shape (num_ignored_gts, 4).
            ref_gt_masks (list[Tensor]) : Masks for each reference bbox,
                has a shape (num_gts, h , w).

        Returns:
            dict[str : Tensor]: All losses.
        """
        x = self.detector.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

        roi_losses = self.detector.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)

        losses.update(roi_losses)

        ref_x = self.detector.extract_feat(ref_img)
        ref_proposals = self.detector.rpn_head.simple_test_rpn(
            ref_x, ref_img_metas)

        track_losses = self.track_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_match_indices, ref_x, ref_img_metas, ref_proposals,
            ref_gt_bboxes, ref_gt_labels, gt_bboxes_ignore, gt_masks,
            ref_gt_bboxes_ignore)

        losses.update(track_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test forward.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): whether to rescale the bboxes.

        Returns:
            dict[str : Tensor]: Track results.
        """
        # TODO inherit from a base tracker
        assert self.with_track_head, 'track head must be implemented.'  # noqa
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)
        proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)

        det_results = self.detector.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

        bbox_results = det_results[0]
        num_classes = len(bbox_results)
        outs_det = results2outs(bbox_results=bbox_results)

        det_bboxes = torch.tensor(outs_det['bboxes']).to(img)
        det_labels = torch.tensor(outs_det['labels']).to(img).long()

        track_bboxes, track_labels, track_ids = self.tracker.track(
            img_metas=img_metas,
            feats=x,
            model=self,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id)

        track_bboxes = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)['bbox_results']

        return dict(det_bboxes=bbox_results, track_bboxes=track_bboxes)
