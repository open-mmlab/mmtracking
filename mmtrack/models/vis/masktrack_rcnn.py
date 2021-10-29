# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models import build_detector, build_head

from mmtrack.core import outs2results, results2outs
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

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)

        proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)

        det_results = self.detector.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        assert len(det_results[0]) == 2, 'There are no mask results.'
        bbox_results = det_results[0][0]
        mask_results = det_results[0][1]
        num_classes = len(bbox_results)

        outs_det = results2outs(
            bbox_results=bbox_results,
            mask_results=mask_results,
            mask_shape=img_metas[0]['ori_shape'][:2])
        det_bboxes = torch.tensor(outs_det['bboxes']).to(img)
        det_labels = torch.tensor(outs_det['labels']).to(img).long()
        det_masks = torch.tensor(outs_det['masks']).to(img).bool()

        (track_bboxes, track_labels, track_masks,
         track_ids) = self.tracker.track(
             img=img,
             img_metas=img_metas,
             model=self,
             feats=x,
             bboxes=det_bboxes,
             labels=det_labels,
             masks=det_masks,
             frame_id=frame_id,
             rescale=rescale,
             **kwargs)

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            masks=track_masks,
            ids=track_ids,
            num_classes=num_classes)
        return dict(
            track_bboxes=track_results['bbox_results'],
            track_masks=track_results['mask_results'])
