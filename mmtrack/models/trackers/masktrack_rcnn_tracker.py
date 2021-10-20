# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
class MaskTrackRCNNTracker(BaseTracker):
    """Tracker for MaskTrack R-CNN.

    Args:
        det_score_coefficient (float): The coefficient of `det_score` when
            computing match score.
        iou_coefficient (float): The coefficient of `ious` when computing
            match score.
        label_coefficient (float): The coefficient of `label_deltas` when
            computing match score.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 det_score_coefficient=1.0,
                 iou_coefficient=2.0,
                 label_coefficient=10.0,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.det_score_coefficient = det_score_coefficient
        self.iou_coefficient = iou_coefficient
        self.label_coefficient = label_coefficient

    def compute_match_score(self, similarity_scores, det_scores, ious,
                            label_deltas):
        """Computing the match score.

        Args:
            similarity_scores (torch.Tensor): of shape (num_current_bboxes,
                num_previous_bboxes + 1). Denoting the similarity scores from
                track head.
            det_scores (torch.Tensor): of shape (num_current_bboxes, 1).
                Denoting the detection score of bboxes of current frame.
            ious (torch.Tensor): of shape (num_current_bboxes,
                num_previous_bboxes). Denoting the ious between current frame
                bboxes and previous frame bboxes.
            label_deltas (torch.Tensor): of shape (num_current_bboxes,
                num_previous_bboxes). Denoting whether the predicted category
                is the same between current frame bboxes and previous frame
                bboxes.

        Returns:
            torch.Tensor: The matching score of shape (num_current_bboxes,
            num_previous_bboxes + 1)
        """
        iou_dummy = ious.new_zeros(ious.shape[0], 1)
        ious = torch.cat((iou_dummy, ious), dim=1)

        label_deltas_dummy = label_deltas.new_ones(label_deltas.shape[0], 1)
        label_deltas = torch.cat((label_deltas_dummy, label_deltas), dim=1)

        match_score = similarity_scores.log()
        match_score += self.det_score_coefficient * det_scores.log()
        match_score += self.iou_coefficient * ious
        match_score += self.label_coefficient * label_deltas

        return match_score

    @force_fp32(apply_to=('img', 'feats'))
    def track(self,
              img,
              img_metas,
              model,
              feats,
              bboxes,
              labels,
              masks,
              frame_id,
              rescale=False,
              **kwargs):
        """Tracking forward function.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            model (nn.Module): VIS model.
            feats (tuple): Backbone features of the input image.
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            masks (Tensor): of shape (N, H, W)
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            tuple: Tracking results.
        """
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            return bboxes, labels, masks, ids

        re_bboxes = bboxes.clone()
        if rescale:
            re_bboxes[:, :4] *= torch.tensor(img_metas[0]['scale_factor']).to(
                bboxes.device)
        roi_feats, _ = model.track_head.extract_roi_feats(feats, [re_bboxes])

        if self.empty:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
        else:
            prev_roi_feats = self.get('roi_feats')
            similarity_logits = model.track_head.simple_test(
                roi_feats, prev_roi_feats)
            similarity_scores = similarity_logits.softmax(dim=1)

            prev_bboxes = self.get('bboxes')
            ious = bbox_overlaps(bboxes[:, :4], prev_bboxes)

            prev_labels = self.get('labels')
            label_deltas = (labels.view(-1, 1) == prev_labels).float()

            match_scores = self.compute_match_score(similarity_scores,
                                                    bboxes[:, 4].view(-1, 1),
                                                    ious, label_deltas)
            _, match_ids = match_scores.max(dim=1)

            ids = match_ids.new_zeros(match_ids.shape[0]) - 1
            best_match_scores = prev_bboxes.new_zeros(
                prev_bboxes.shape[0]) - 1e6
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    ids[idx] = self.num_tracks
                    self.num_tracks += 1
                else:
                    match_score = match_scores[idx, match_id]
                    # TODO: fix the bug where multiple candidate might match
                    # with the same previous object.
                    if match_score > best_match_scores[match_id - 1]:
                        ids[idx] = self.ids[match_id - 1]
                        best_match_scores[match_id - 1] = match_score

        valid_inds = ids > -1
        ids = ids[valid_inds]
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        masks = masks[valid_inds]
        roi_feats = roi_feats[valid_inds]

        self.update(
            ids=ids,
            bboxes=bboxes[:, :4],
            scores=bboxes[:, 4],
            labels=labels,
            masks=masks,
            roi_feats=roi_feats,
            frame_ids=frame_id)
        return bboxes, labels, masks, ids
