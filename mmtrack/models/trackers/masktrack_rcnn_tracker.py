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
        match_weights (dict[str : float]): The Weighting factor when computing
        the match score. It contains keys as follows:

            - det_score (float): The coefficient of `det_score` when computing
                match score.
            - iou (float): The coefficient of `ious` when computing match
                score.
            - det_label (float): The coefficient of `label_deltas` when
                computing match score.

        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 match_weights=dict(det_score=1.0, iou=2.0, det_label=10.0),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.match_weights = match_weights

    def get_match_score(self, bboxes, labels, prev_bboxes, prev_labels,
                        similarity_logits):
        """Get the match score.

        Args:
            bboxes (torch.Tensor): of shape (num_current_bboxes, 5) in
                [tl_x, tl_y, br_x, br_y, score] format. Denoting the detection
                bboxes of current frame.
            labels (torch.Tensor): of shape (num_current_bboxes, )
            prev_bboxes (torch.Tensor): of shape (num_previous_bboxes, 5) in
                [tl_x, tl_y, br_x, br_y, score] format.  Denoting the
                detection bboxes of previous frame.
            prev_labels (torch.Tensor): of shape (num_previous_bboxes, )
            similarity_logits (torch.Tensor): of shape (num_current_bboxes,
                num_previous_bboxes + 1). Denoting the similarity logits from
                track head.

        Returns:
            torch.Tensor: The matching score of shape (num_current_bboxes,
            num_previous_bboxes + 1)
        """
        similarity_scores = similarity_logits.softmax(dim=1)

        ious = bbox_overlaps(bboxes[:, :4], prev_bboxes[:, :4])
        iou_dummy = ious.new_zeros(ious.shape[0], 1)
        ious = torch.cat((iou_dummy, ious), dim=1)

        label_deltas = (labels.view(-1, 1) == prev_labels).float()
        label_deltas_dummy = label_deltas.new_ones(label_deltas.shape[0], 1)
        label_deltas = torch.cat((label_deltas_dummy, label_deltas), dim=1)

        match_score = similarity_scores.log()
        match_score += self.match_weights['det_score'] * \
            bboxes[:, 4].view(-1, 1).log()
        match_score += self.match_weights['iou'] * ious
        match_score += self.match_weights['det_label'] * label_deltas

        return match_score

    def assign_ids(self, match_scores):
        num_prev_bboxes = match_scores.shape[1] - 1
        _, match_ids = match_scores.max(dim=1)

        ids = match_ids.new_zeros(match_ids.shape[0]) - 1
        best_match_scores = match_scores.new_zeros(num_prev_bboxes) - 1e6
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
        return ids, best_match_scores

    @force_fp32(apply_to=('img', 'feats', 'bboxes'))
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

        rescaled_bboxes = bboxes.clone()
        if rescale:
            rescaled_bboxes[:, :4] *= torch.tensor(
                img_metas[0]['scale_factor']).to(bboxes.device)
        roi_feats, _ = model.track_head.extract_roi_feats(
            feats, [rescaled_bboxes])

        if self.empty:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
        else:
            prev_bboxes = self.get('bboxes')
            prev_labels = self.get('labels')
            prev_roi_feats = self.get('roi_feats')

            similarity_logits = model.track_head.simple_test(
                roi_feats, prev_roi_feats)
            match_scores = self.get_match_score(bboxes, labels, prev_bboxes,
                                                prev_labels, similarity_logits)
            ids, best_match_scores = self.assign_ids(match_scores)

        valid_inds = ids > -1
        ids = ids[valid_inds]
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        masks = masks[valid_inds]
        roi_feats = roi_feats[valid_inds]

        self.update(
            ids=ids,
            bboxes=bboxes,
            labels=labels,
            masks=masks,
            roi_feats=roi_feats,
            frame_ids=frame_id)
        return bboxes, labels, masks, ids
