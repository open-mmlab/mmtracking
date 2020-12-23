import numpy as np
import torch
from mmdet.core import bbox_overlaps
from scipy.optimize import linear_sum_assignment

from mmtrack.core import imrenormalize
from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
class SortTracker(BaseTracker):

    def __init__(self,
                 obj_score_thr=0.3,
                 reid=dict(
                     num_samples=100,
                     img_scale=(256, 128),
                     img_norm_cfg=None,
                     match_score_thr=2.0),
                 match_iou_thr=0.7,
                 num_tentatives=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.obj_score_thr = obj_score_thr
        self.reid = reid
        self.match_iou_thr = match_iou_thr
        self.num_tentatives = num_tentatives

    def xyxy2xyah(self, bboxes):
        cx = (bboxes[:, 2] + bboxes[:, 0]) / 2
        cy = (bboxes[:, 3] + bboxes[:, 1]) / 2
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        xyah = torch.stack([cx, cy, w / h, h], -1)
        return xyah

    @property
    def confirmed_ids(self):
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    def init_track(self, id, obj):
        super().init_track(id, obj)
        self.tracks[id].tentative = True
        bbox = self.xyxy2xyah(obj['bboxes'])  # size = (5, )
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)

    def update_track(self, id, obj):
        super().update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = self.xyxy2xyah(obj['bboxes'])  # size = (5, )
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)

    def pop_invalid_tracks(self, frame_id):
        invalid_ids = []
        for k, v in self.tracks.items():
            case1 = frame_id - v['frame_ids'][-1] >= self.num_frames_retain
            case2 = v.tentative and v['frame_ids'][-1] != frame_id
            if case1 or case2:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def track(self,
              img,
              img_metas,
              model,
              feats,
              bboxes,
              labels,
              frame_id,
              rescale=False,
              **kwargs):
        if model.with_reid:
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
            if model.with_reid:
                embeds = model.reid.simple_test(
                    self.crop_imgs(reid_img, img_metas, bboxes[:, :4].clone(),
                                   rescale))
        else:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)

            # motion
            if model.with_motion:
                self.tracks, costs = model.motion.track(
                    self.tracks, self.xyxy2xyah(bboxes))

            if model.with_reid:
                embeds = model.reid.simple_test(
                    self.crop_imgs(reid_img, img_metas, bboxes[:, :4].clone(),
                                   rescale))
                # reid
                active_ids = self.confirmed_ids
                if len(active_ids) > 0:
                    track_embeds = self.get(
                        'embeds',
                        active_ids,
                        self.reid.get('num_samples', None),
                        behavior='mean')
                    reid_dists = torch.cdist(track_embeds,
                                             embeds).cpu().numpy()
                    gating_inds = costs == np.nan
                    reid_dists[gating_inds] = np.nan

                    row, col = linear_sum_assignment(reid_dists)
                    for r, c in zip(row, col):
                        dist = reid_dists[r, c]
                        if dist <= self.reid['match_score_thr']:
                            ids[c] = active_ids[r]

            active_ids = [id for id in self.ids if id not in ids]
            active_inds = torch.nonzero(ids == -1).squeeze(0)
            track_bboxes = self.get('bboxes', active_ids)
            ious = bbox_overlaps(track_bboxes, bboxes[active_inds][:, :-1])
            dists = 1 - ious
            dists[dists > 1 - self.match_iou_thr] = np.nan
            row, col = linear_sum_assignment(dists)
            for r, c in zip(row, col):
                dist = dists[r, c]
                if not np.isfinite(dist):
                    continue
                ids[active_inds[c]] = active_ids[r]

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
            embeds=embeds if model.with_reid else None,
            frame_ids=frame_id)

        return bboxes, labels, ids
