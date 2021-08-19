# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps, multiclass_nms
from scipy.optimize import linear_sum_assignment

from mmtrack.core import imrenormalize
from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
class TracktorTracker(BaseTracker):
    """Tracker for Tracktor.

    Args:
        obj_score_thr (float, optional): Threshold to filter the objects.
            Defaults to 0.3.
        reid (dict, optional): Configuration for the ReID model.

            - obj_score_thr (float, optional): Threshold to filter the
                regressed objects. Default to 0.5.
            - nms (dict, optional): NMS configuration to filter the regressed
                objects. Default to `dict(type='nms', iou_threshold=0.6)`.
            - match_iou_thr (float, optional): Minimum IoU when matching
                objects with IoU. Default to 0.3.
        reid (dict, optional): Configuration for the ReID model.

            - num_samples (int, optional): Number of samples to calculate the
                feature embeddings of a track. Default to 10.
            - image_scale (tuple, optional): Input scale of the ReID model.
                Default to (256, 128).
            - img_norm_cfg (dict, optional): Configuration to normalize the
                input. Default to None.
            - match_score_thr (float, optional): Similarity threshold for the
                matching process. Default to 2.0.
            - match_iou_thr (float, optional): Minimum IoU when matching
                objects with embedding similarity. Default to 0.2.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

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
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thr = obj_score_thr
        self.regression = regression
        self.reid = reid

    def regress_tracks(self, x, img_metas, detector, frame_id, rescale=False):
        """Regress the tracks to current frame."""
        memo = self.memo
        bboxes = memo.bboxes[memo.frame_ids == frame_id - 1]
        ids = memo.ids[memo.frame_ids == frame_id - 1]
        if rescale:
            bboxes *= torch.tensor(img_metas[0]['scale_factor']).to(
                bboxes.device)
        track_bboxes, track_scores = detector.roi_head.simple_test_bboxes(
            x, img_metas, [bboxes], None, rescale=rescale)
        track_bboxes, track_labels, valid_inds = multiclass_nms(
            track_bboxes[0],
            track_scores[0],
            0,
            self.regression['nms'],
            return_inds=True)
        ids = ids[valid_inds]

        valid_inds = track_bboxes[:, -1] > self.regression['obj_score_thr']
        return track_bboxes[valid_inds], track_labels[valid_inds], ids[
            valid_inds]

    @force_fp32(apply_to=('img', 'feats'))
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
        """Tracking forward function.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            model (nn.Module): MOT model.
            feats (tuple): Backbone features of the input image.
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            tuple: Tracking results.
        """
        if self.with_reid:
            if self.reid.get('img_norm_cfg', False):
                reid_img = imrenormalize(img, img_metas[0]['img_norm_cfg'],
                                         self.reid['img_norm_cfg'])
            else:
                reid_img = img.clone()

        valid_inds = bboxes[:, -1] > self.obj_score_thr
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]

        if self.empty:
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
            # motion
            if model.with_cmc:
                if model.with_linear_motion:
                    num_samples = model.linear_motion.num_samples
                else:
                    num_samples = 1
                self.tracks = model.cmc.track(self.last_img, img, self.tracks,
                                              num_samples, frame_id)

            if model.with_linear_motion:
                self.tracks = model.linear_motion.track(self.tracks, frame_id)

            # propagate tracks
            prop_bboxes, prop_labels, prop_ids = self.regress_tracks(
                feats, img_metas, model.detector, frame_id, rescale)

            # filter bboxes with propagated tracks
            ious = bbox_overlaps(bboxes[:, :4], prop_bboxes[:, :4])
            valid_inds = (ious < self.regression['match_iou_thr']).all(dim=1)
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)

            if self.with_reid:
                prop_embeds = model.reid.simple_test(
                    self.crop_imgs(reid_img, img_metas,
                                   prop_bboxes[:, :4].clone(), rescale))
                if bboxes.size(0) > 0:
                    embeds = model.reid.simple_test(
                        self.crop_imgs(reid_img, img_metas,
                                       bboxes[:, :4].clone(), rescale))
                else:
                    embeds = prop_embeds.new_zeros((0, prop_embeds.size(1)))
                # reid
                active_ids = [int(_) for _ in self.ids if _ not in prop_ids]
                if len(active_ids) > 0 and bboxes.size(0) > 0:
                    track_embeds = self.get(
                        'embeds',
                        active_ids,
                        self.reid.get('num_samples', None),
                        behavior='mean')
                    reid_dists = torch.cdist(track_embeds,
                                             embeds).cpu().numpy()

                    track_bboxes = self.get('bboxes', active_ids)
                    ious = bbox_overlaps(track_bboxes,
                                         bboxes[:, :4]).cpu().numpy()
                    iou_masks = ious < self.reid['match_iou_thr']
                    reid_dists[iou_masks] = 1e6

                    row, col = linear_sum_assignment(reid_dists)
                    for r, c in zip(row, col):
                        dist = reid_dists[r, c]
                        if dist <= self.reid['match_score_thr']:
                            ids[c] = active_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long)
            self.num_tracks += new_track_inds.sum()

            if bboxes.shape[1] == 4:
                bboxes = bboxes.new_zeros((0, 5))
            if prop_bboxes.shape[1] == 4:
                prop_bboxes = prop_bboxes.new_zeros((0, 5))

            bboxes = torch.cat((prop_bboxes, bboxes), dim=0)
            labels = torch.cat((prop_labels, labels), dim=0)
            ids = torch.cat((prop_ids, ids), dim=0)
            if self.with_reid:
                embeds = torch.cat((prop_embeds, embeds), dim=0)

        self.update(
            ids=ids,
            bboxes=bboxes[:, :4],
            scores=bboxes[:, -1],
            labels=labels,
            embeds=embeds if self.with_reid else None,
            frame_ids=frame_id)
        self.last_img = img
        return bboxes, labels, ids
