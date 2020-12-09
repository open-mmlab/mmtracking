import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import bbox_overlaps, multiclass_nms
from scipy.optimize import linear_sum_assignment

from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
class TracktorTracker(BaseTracker):

    def __init__(self,
                 obj_score_thr=0.5,
                 regression=dict(
                     obj_score_thr=0.5,
                     nms=dict(type='nms', iou_threshold=0.6),
                     match_iou_thr=0.3),
                 reid=dict(
                     num_samples=10,
                     img_scale=(256, 128),
                     img_norm_cfg=None,
                     match_score_thr=2.0,
                     match_iou_thr=0.2),
                 **kwargs):
        super().__init__(**kwargs)
        self.obj_score_thr = obj_score_thr
        self.regression = self.regression
        self.reid = self.reid

    def regress_tracks(self, x, img_metas, detector, frame_id, rescale=False):
        memo = self.memo
        bboxes = memo.bboxes[memo.frame_id == frame_id - 1]
        ids = memo.ids[memo.frame_id == frame_id - 1]
        if rescale:
            bboxes *= img_metas[0]['scale_factor']
        (track_bboxes,
         track_scores) = self.detector.roi_head.simple_test_bboxes(
             x, img_metas, bboxes, None, rescale=rescale)
        track_bboxes, track_labels, valid_inds = multiclass_nms(
            track_bboxes,
            track_scores,
            self.regression['obj_score_thr'],
            self.regression['nms'],
            return_inds=True)

        track_bboxes = track_bboxes[valid_inds]
        track_labels = track_labels[valid_inds]
        ids = ids[valid_inds]
        return track_bboxes, track_labels, ids

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
        valid_inds = bboxes[:, -1] > self.obj_score_thr
        if self.reid is not None:
            if self.reid.get('img_norm_cfg', False):
                reid_img = self.imrenormalize(img,
                                              img_metas[0]['img_norm_cfg'],
                                              self.reid['img_norm_cfg'])
            else:
                reid_img = img.clone()

        if not self.empty:

            if model.with_cmc:
                if model.with_linear_motion:
                    num_samples = model.linear_motion.num_samples
                else:
                    num_samples = 1
                self.tracks = model.cmc.track(self.last_img, img, self.tracks,
                                              num_samples, frame_id)

            if model.with_linear_motion:
                self.tracks = model.linear_motion.track(self.tracks, frame_id)

            prop_bboxes, prop_labels, prop_ids = self.regress_tracks(
                feats, img_metas, model.detector, frame_id, rescale)
            # prop_embeds = model.reid.forward_test(
            # self.crop_imgs(reid_img, img_metas, prop_bboxes, rescale))

            ious = bbox_overlaps(bboxes[:, :-1], prop_bboxes[:, :-1])
            valid_inds &= (ious < self.regression['match_iou_thr']).all(dim=1)
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            embeds = model.reid.forward_test(
                self.crop_imgs(reid_img, img_metas, bboxes, rescale))

            reid_track_ids = [_ not in prop_ids for _ in self.ids]

            track_embeds = self.get('embeds', reid_track_ids,
                                    self.reid.get('num_samples', None))
            reid_dists = torch.cdist(track_embeds, embeds).cpu().numpy()
            track_bboxes = self.get('bboxes', reid_track_ids)
            ious = bbox_overlaps(track_bboxes[:, :-1], bboxes[:, :-1])
            iou_masks = ious < self.reid['match_iou_thr']
            reid_dists[~iou_masks] = np.nan

            ids = torch.full((track_bboxes.size(0), ), -1, dtype=torch.long)
            row, col = linear_sum_assignment(reid_dists)
            for r, c in zip(row, col):
                dist = reid_dists[r, c]
                if not np.isfinite(dist):
                    continue
                # if dist <= self.reid['match_score_thr']:
        else:
            ids = torch.arange(bboxes.shape[0])

        self.last_img = img
        return bboxes, labels, ids

    def crop_imgs(self, img, img_metas, bboxes, rescale=False):
        h, w, _ = img_metas[0]['img_shape']
        img = img[:, :, :h, :w]
        if rescale:
            bboxes *= img_metas[0]['scale_factor']
        bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], min=0, max=w)
        bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], min=0, max=h)

        crop_imgs = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            crop_img = img[:, :, y1:y2, x1:x2]
            if self.reid.get('img_scale', False):
                crop_img = F.interpolate(
                    crop_img,
                    size=self.reid['img_scale'],
                    mode='bilinear',
                    align_corners=False)
            crop_imgs.append(crop_img)

        return torch.cat(crop_imgs, dim=0)

    def imrenormalize(self, img, img_norm_cfg, new_img_norm_cfg):

        def _imrenormalize(img, img_norm_cfg, new_img_norm_cfg):
            img = mmcv.imdenormalize(img, **img_norm_cfg)
            img = mmcv.imnormalize(img, **new_img_norm_cfg)
            return img

        if isinstance(img, torch.Tensor):
            assert img.ndim == 4 and img.shape[0] == 1
            new_img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            new_img = _imrenormalize(new_img, img_norm_cfg, new_img_norm_cfg)
            new_img = new_img.transpose(2, 0, 1)[None]
            return torch.from_numpy(new_img).to(img.device)
        else:
            return _imrenormalize(img, img_norm_cfg, new_img_norm_cfg)
