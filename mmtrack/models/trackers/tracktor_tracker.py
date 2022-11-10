# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmdet.models.layers import multiclass_nms
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
# TODO: unify the linear_assignment package for different trackers
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import OptConfigType, imrenormalize
from .base_tracker import BaseTracker


@MODELS.register_module()
class TracktorTracker(BaseTracker):
    """Tracker for Tracktor.

    Args:
        obj_score_thr (float, optional): Threshold to filter the objects.
            Defaults to 0.5.
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
                 obj_score_thr: float = 0.5,
                 regression: dict = dict(
                     obj_score_thr=0.5,
                     nms=dict(type='nms', iou_threshold=0.6),
                     match_iou_thr=0.3),
                 reid: dict = dict(
                     num_samples=10,
                     img_scale=(256, 128),
                     img_norm_cfg=None,
                     match_score_thr=2.0,
                     match_iou_thr=0.2),
                 **kwargs):
        super().__init__(**kwargs)
        self.obj_score_thr = obj_score_thr
        self.regression = regression
        self.reid = reid

    def regress_tracks(self,
                       x: List[Tensor],
                       metainfo: dict,
                       detector: nn.Module,
                       frame_id: int,
                       rescale: bool = False):
        """Regress the tracks to current frame."""
        memo = self.memo
        bboxes = memo.bboxes[memo.frame_ids == frame_id - 1]
        ids = memo.ids[memo.frame_ids == frame_id - 1]

        if rescale:
            factor_x, factor_y = metainfo['scale_factor']
            bboxes *= torch.tensor([factor_x, factor_y, factor_x,
                                    factor_y]).to(bboxes.device)

        if bboxes.size(0) == 0:
            return bboxes.new_zeros((0, 4)), bboxes.new_zeros(0), \
                   ids.new_zeros(0), ids.new_zeros(0),

        proposals = InstanceData(**dict(bboxes=bboxes))
        det_results = detector.roi_head.predict_bbox(x, [metainfo],
                                                     [proposals], None,
                                                     rescale)
        track_bboxes = det_results[0].bboxes
        track_scores = det_results[0].scores
        _track_bboxes, track_labels, valid_inds = multiclass_nms(
            track_bboxes,
            track_scores,
            0,
            self.regression['nms'],
            return_inds=True)
        ids = ids[valid_inds]

        track_bboxes = _track_bboxes[:, :-1].clone()
        track_scores = _track_bboxes[:, -1].clone()
        valid_inds = track_scores > self.regression['obj_score_thr']
        return track_bboxes[valid_inds], track_scores[
            valid_inds], track_labels[valid_inds], ids[valid_inds]

    def track(self,
              model: nn.Module,
              img: Tensor,
              feats: List[Tensor],
              data_sample: TrackDataSample,
              data_preprocessor: OptConfigType = None,
              rescale: bool = False,
              **kwargs) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                ByteTrack method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.
            data_preprocessor (dict or ConfigDict, optional): The pre-process
               config of :class:`TrackDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_det_instances.bboxes
        labels = data_sample.pred_det_instances.labels
        scores = data_sample.pred_det_instances.scores

        frame_id = metainfo.get('frame_id', -1)
        if frame_id == 0:
            self.reset()

        if self.with_reid:
            if self.reid.get('img_norm_cfg', False):
                img_norm_cfg = dict(
                    mean=data_preprocessor['mean'],
                    std=data_preprocessor['std'],
                    to_bgr=data_preprocessor['rgb_to_bgr'])
                reid_img = imrenormalize(img, img_norm_cfg,
                                         self.reid['img_norm_cfg'])
            else:
                reid_img = img.clone()

        valid_inds = scores > self.obj_score_thr
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]

        if self.empty:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += num_new_tracks
            if self.with_reid:
                crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                       rescale)
                if crops.size(0) > 0:
                    embeds = model.reid(crops, mode='tensor')
                else:
                    embeds = crops.new_zeros((0, model.reid.head.out_channels))
        else:
            # motion
            if model.with_cmc:
                if model.with_linear_motion:
                    num_samples = model.linear_motion.num_samples
                else:
                    num_samples = 1
                self.tracks = model.cmc.track(self.last_img, img, self.tracks,
                                              num_samples, frame_id, metainfo)

            if model.with_linear_motion:
                self.tracks = model.linear_motion.track(self.tracks, frame_id)

            # propagate tracks
            prop_bboxes, prop_scores, prop_labels, prop_ids = \
                self.regress_tracks(feats, metainfo,
                                    model.detector, frame_id, rescale)

            # filter bboxes with propagated tracks
            ious = bbox_overlaps(bboxes[:, :4], prop_bboxes[:, :4])
            valid_inds = (ious < self.regression['match_iou_thr']).all(dim=1)
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            scores = scores[valid_inds]
            ids = torch.full((bboxes.size(0), ), -1,
                             dtype=torch.long).to(bboxes.device)

            if self.with_reid:
                prop_crops = self.crop_imgs(reid_img, metainfo,
                                            prop_bboxes.clone(), rescale)
                if prop_crops.size(0) > 0:
                    prop_embeds = model.reid(prop_crops, mode='tensor')
                else:
                    prop_embeds = prop_crops.new_zeros(
                        (0, model.reid.head.out_channels))
                if bboxes.size(0) > 0:
                    embeds = model.reid(
                        self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                       rescale),
                        mode='tensor')
                else:
                    embeds = prop_embeds.\
                        new_zeros((0, model.reid.head.out_channels))
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
                    ious = bbox_overlaps(track_bboxes, bboxes).cpu().numpy()
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
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += new_track_inds.sum()

            bboxes = torch.cat((prop_bboxes, bboxes), dim=0)
            scores = torch.cat((prop_scores, scores), dim=0)
            labels = torch.cat((prop_labels, labels), dim=0)
            ids = torch.cat((prop_ids, ids), dim=0)
            if self.with_reid:
                embeds = torch.cat((prop_embeds, embeds), dim=0)

        self.update(
            ids=ids,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            embeds=embeds if self.with_reid else None,
            frame_ids=frame_id)
        self.last_img = img

        # update pred_track_instances
        pred_track_instances = InstanceData()
        pred_track_instances.bboxes = bboxes
        pred_track_instances.labels = labels
        pred_track_instances.scores = scores
        pred_track_instances.instances_id = ids

        return pred_track_instances
