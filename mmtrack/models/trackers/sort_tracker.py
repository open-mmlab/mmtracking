# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps
from motmetrics.lap import linear_sum_assignment

from mmtrack.core import imrenormalize
from mmtrack.core.bbox import bbox_xyxy_to_cxcyah
from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
class SortTracker(BaseTracker):
    """Tracker for DeepSORT.

    Args:
        obj_score_thr (float, optional): Threshold to filter the objects.
            Defaults to 0.3.
        reid (dict, optional): Configuration for the ReID model.

            - num_samples (int, optional): Number of samples to calculate the
                feature embeddings of a track. Default to 10.
            - image_scale (tuple, optional): Input scale of the ReID model.
                Default to (256, 128).
            - img_norm_cfg (dict, optional): Configuration to normalize the
                input. Default to None.
            - match_score_thr (float, optional): Similarity threshold for the
                matching process. Default to 2.0.
        match_iou_thr (float, optional): Threshold of the IoU matching process.
            Defaults to 0.7.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 obj_score_thr=0.3,
                 reid=dict(
                     num_samples=10,
                     img_scale=(256, 128),
                     img_norm_cfg=None,
                     match_score_thr=2.0),
                 match_iou_thr=0.7,
                 num_tentatives=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thr = obj_score_thr
        self.reid = reid
        self.match_iou_thr = match_iou_thr
        self.num_tentatives = num_tentatives

    @property
    def confirmed_ids(self):
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    def init_track(self, id, obj):
        """Initialize a track."""
        super().init_track(id, obj)
        self.tracks[id].tentative = True
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)

    def update_track(self, id, obj):
        """Update a track."""
        super().update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)

    def pop_invalid_tracks(self, frame_id):
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            # case1: disappeared frames >= self.num_frames_retrain
            case1 = frame_id - v['frame_ids'][-1] >= self.num_frames_retain
            # case2: tentative tracks but not matched in this frame
            case2 = v.tentative and v['frame_ids'][-1] != frame_id
            if case1 or case2:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    @force_fp32(apply_to=('img', ))
    def track(self,
              img,
              img_metas,
              model,
              bboxes,
              labels,
              frame_id,
              rescale=False,
              **kwargs):
        """Tracking forward function.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            model (nn.Module): MOT model.
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            tuple: Tracking results.
        """
        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if self.with_reid:
            if self.reid.get('img_norm_cfg', False):
                reid_img = imrenormalize(img, img_metas[0]['img_norm_cfg'],
                                         self.reid['img_norm_cfg'])
            else:
                reid_img = img.clone()

        valid_inds = bboxes[:, -1] > self.obj_score_thr
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]

        if self.empty or bboxes.size(0) == 0:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
            if self.with_reid:
                embeds = model.reid.simple_test(
                    self.crop_imgs(reid_img, img_metas, bboxes[:, :4].clone(),
                                   rescale))
        else:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)

            # motion
            if model.with_motion:
                self.tracks, costs = model.motion.track(
                    self.tracks, bbox_xyxy_to_cxcyah(bboxes))

            active_ids = self.confirmed_ids
            if self.with_reid:
                embeds = model.reid.simple_test(
                    self.crop_imgs(reid_img, img_metas, bboxes[:, :4].clone(),
                                   rescale))
                # reid
                if len(active_ids) > 0:
                    track_embeds = self.get(
                        'embeds',
                        active_ids,
                        self.reid.get('num_samples', None),
                        behavior='mean')
                    reid_dists = torch.cdist(track_embeds,
                                             embeds).cpu().numpy()

                    valid_inds = [list(self.ids).index(_) for _ in active_ids]
                    reid_dists[~np.isfinite(costs[valid_inds, :])] = np.nan

                    row, col = linear_sum_assignment(reid_dists)
                    for r, c in zip(row, col):
                        dist = reid_dists[r, c]
                        if not np.isfinite(dist):
                            continue
                        if dist <= self.reid['match_score_thr']:
                            ids[c] = active_ids[r]

            active_ids = [
                id for id in self.ids if id not in ids
                and self.tracks[id].frame_ids[-1] == frame_id - 1
            ]
            if len(active_ids) > 0:
                active_dets = torch.nonzero(ids == -1).squeeze(1)
                track_bboxes = self.get('bboxes', active_ids)
                ious = bbox_overlaps(
                    track_bboxes, bboxes[active_dets][:, :-1]).cpu().numpy()
                dists = 1 - ious
                row, col = linear_sum_assignment(dists)
                for r, c in zip(row, col):
                    dist = dists[r, c]
                    if dist < 1 - self.match_iou_thr:
                        ids[active_dets[c]] = active_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes[:, :4],
            scores=bboxes[:, -1],
            labels=labels,
            embeds=embeds if self.with_reid else None,
            frame_ids=frame_id)
        return bboxes, labels, ids
