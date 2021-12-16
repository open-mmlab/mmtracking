# Copyright (c) OpenMMLab. All rights reserved.
import lap
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
class ByteTracker(BaseTracker):
    """Tracker for ByteTrack.

    Args:
        high_det_score (float): Threshold of the first matching detection bbox.
        low_det_score (float): Threshold of the second matching detection bbox.
        init_track_score (float): Threshold of initializing a new tracklet.
        weight_iou (bool): Whether using detection scores to weight IOU.
        first_match_iou_thr (float): Threshold of the first matching threshold.
        second_match_iou_thr (float, optional): Threshold of the second
            matching threshold. Defaults to 0.5.
        tentative_match_iou_thr (float, optional): Threshold of the tentative
            matching threshold. Defaults to 0.5.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 high_det_score,
                 low_det_score,
                 init_track_score,
                 weight_iou,
                 first_match_iou_thr,
                 second_match_iou_thr=0.5,
                 tentative_match_iou_thr=0.3,
                 num_tentatives=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.high_det_score = high_det_score
        self.low_det_score = low_det_score
        self.init_track_score = init_track_score

        self.weight_iou = weight_iou
        self.first_match_iou_thr = first_match_iou_thr
        self.second_match_iou_thr = second_match_iou_thr
        self.tentative_match_iou_thr = tentative_match_iou_thr

        self.num_tentatives = num_tentatives

    @property
    def confirmed_ids(self):
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    @property
    def unconfirmed_ids(self):
        """Unconfirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if track.tentative]
        return ids

    def init_track(self, id, obj):
        """Initialize a track."""
        super().init_track(id, obj)
        if self.tracks[id].frame_ids[-1] == 0:
            self.tracks[id].tentative = False
        else:
            self.tracks[id].tentative = True
        bbox = self.xyxy2xyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
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
        bbox = self.xyxy2xyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
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

    def get_data_from_inds(self, inds, **kwargs):
        """Get data based on indexes."""
        out = []
        for k, v in kwargs.items():
            out.append(v[inds])
        return tuple(out)

    def xyxy2xyah(self, bboxes):
        """Transform bounding boxes from [x1, y1, x2, y2] format to.

        [cx, cy, ratio, h] format.
        """
        cx = (bboxes[:, 2] + bboxes[:, 0]) / 2
        cy = (bboxes[:, 3] + bboxes[:, 1]) / 2
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        xyah = torch.stack([cx, cy, w / h, h], -1)
        return xyah

    def xyah2xyxy(self, bboxes):
        """Transform bounding boxes from [cx, cy, ratio, h] format to.

        [x1, y1, x2, y2] format.
        """
        cx, cy, ratio, h = bboxes.split((1, 1, 1, 1), dim=-1)
        w = ratio * h
        x1y1x2y2 = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
        return torch.cat(x1y1x2y2, dim=-1)

    def assign_ids(self, ids, det_bboxes, weight_iou=False, match_iou_thr=0.5):
        """Assign ids.

        Args:
            ids (list[int]): Tracking ids.
            det_bboxes (Tensor): of shape (N, 5)
            weight_iou (bool, optional): Whether using detection scores to
                weight IOU. Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.

        Returns:
            tuple(int): The assigning ids.
        """
        # get track_bboxes
        track_bboxes = np.zeros((0, 4))
        for id in ids:
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes = self.xyah2xyxy(track_bboxes)

        # compute distance
        ious = bbox_overlaps(track_bboxes, det_bboxes[:, :4])
        if weight_iou:
            ious *= det_bboxes[:, 4][None]
        dists = (1 - ious).cpu().numpy()

        # bipartite match
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col

    @force_fp32(apply_to=('img', 'bboxes'))
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

        if self.empty or bboxes.size(0) == 0:
            valid_inds = bboxes[:, -1] > self.init_track_score
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks

        else:
            # 0. init
            ids = torch.full((bboxes.size(0), ), -1).to(labels)

            # get the detection bboxes for the first association
            first_det_inds = bboxes[:, -1] > self.high_det_score
            (first_det_bboxes, first_det_labels,
             first_det_ids) = self.get_data_from_inds(
                 inds=first_det_inds, bboxes=bboxes, labels=labels, ids=ids)

            # get the detection bboxes for the second association
            second_det_inds = (~first_det_inds) & (
                bboxes[:, -1] > self.low_det_score)
            (second_det_bboxes, second_det_labels,
             second_det_ids) = self.get_data_from_inds(
                 inds=second_det_inds, bboxes=bboxes, labels=labels, ids=ids)

            # 1. use Kalman Filter to predict current location
            for id in self.confirmed_ids:
                # track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                (self.tracks[id].mean,
                 self.tracks[id].covariance) = self.kf.predict(
                     self.tracks[id].mean, self.tracks[id].covariance)

            # 2. first match
            first_match_track_inds, first_match_det_inds = self.assign_ids(
                self.confirmed_ids, first_det_bboxes, self.weight_iou,
                self.first_match_iou_thr)
            # '-1' mean a detection box is not matched with tracklets in
            # previous frame
            valid = first_match_det_inds > -1
            first_det_ids[valid] = torch.tensor(
                self.confirmed_ids)[first_match_det_inds[valid]].to(labels)

            (first_match_det_bboxes, first_match_det_labels,
             first_match_det_ids) = self.get_data_from_inds(
                 inds=valid,
                 bboxes=first_det_bboxes,
                 labels=first_det_labels,
                 ids=first_det_ids)
            (first_unmatch_det_bboxes, first_unmatch_det_labels,
             first_unmatch_det_ids) = self.get_data_from_inds(
                 inds=~valid,
                 bboxes=first_det_bboxes,
                 labels=first_det_labels,
                 ids=first_det_ids)
            assert (first_match_det_ids > -1).all()
            assert (first_unmatch_det_ids == -1).all()

            # 3. use unmatched detection bboxes from the first match to match
            # the unconfirmed tracks
            (tentative_match_track_inds,
             tentative_match_det_inds) = self.assign_ids(
                 self.unconfirmed_ids, first_unmatch_det_bboxes,
                 self.weight_iou, self.tentative_match_iou_thr)
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            # 4. second match for unmatched tracks from the first match
            first_unmatch_track_ids = []
            for i, id in enumerate(self.confirmed_ids):
                # tracklet is not matched in the first match
                case_1 = first_match_track_inds[i] == -1
                # tracklet is not lost in the previous frame
                case_2 = self.tracks[id].frame_ids[-1] == frame_id - 1
                if case_1 and case_2:
                    first_unmatch_track_ids.append(id)

            second_match_track_inds, second_match_det_inds = self.assign_ids(
                first_unmatch_track_ids, second_det_bboxes, False,
                self.second_match_iou_thr)
            valid = second_match_det_inds > -1
            second_det_ids[valid] = torch.tensor(first_unmatch_track_ids)[
                second_match_det_inds[valid]].to(ids)

            # 5. gather all matched detection bboxes from step 2-4
            # we only keep matched detection bboxes in second match, which
            # means the id != -1
            valid = second_det_ids > -1
            bboxes = torch.cat(
                (first_match_det_bboxes, first_unmatch_det_bboxes), dim=0)
            bboxes = torch.cat((bboxes, second_det_bboxes[valid]), dim=0)

            labels = torch.cat(
                (first_match_det_labels, first_unmatch_det_labels), dim=0)
            labels = torch.cat((labels, second_det_labels[valid]), dim=0)

            ids = torch.cat((first_match_det_ids, first_unmatch_det_ids),
                            dim=0)
            ids = torch.cat((ids, second_det_ids[valid]), dim=0)

            # 6. assign new ids
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()

        self.update(ids=ids, bboxes=bboxes, labels=labels, frame_ids=frame_id)
        return bboxes, labels, ids
