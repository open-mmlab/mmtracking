# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from ..task_modules.track import embed_similarity
from .base_tracker import BaseTracker


@MODELS.register_module()
class QuasiDenseTAOTracker(BaseTracker):
    """Tracker for Quasi-Dense Tracking Method with TAO Dataset.

    Args:
        init_score_thr (float): The cls_score threshold to
            initialize a new tracklet. Defaults to 0.8.
        obj_score_thr (float): The cls_score threshold to
            update a tracked tracklet. Defaults to 0.5.
        match_score_thr (float): The match threshold. Defaults to 0.5.
        memo_frames (int): The most frames in a track memory.
            Defaults to 10.
        memo_momentum (float): The momentum value for embeds updating.
            Defaults to 0.8.
        momentum_obj_score (float): The momentum value for object score
            updating. Default to 0.5.
        obj_score_diff_thr (float): The threshold for object score
            difference for adjacent detection in the same track.
        nms_conf_thr (float): The nms threshold for confidence.
            Defaults to 0.5.
        distractor_nms_thr (float): The nms threshold for inter-class.
            Defaults to 0.3.
        distractor_score_thr (float): The threshold for distractor.
            Defaults to 0.5.
        match_metric (str): The match metric. Defaults to 'bisoftmax'.
        match_with_cosine (bool): If True, match score contains cosine
            similarity. Default to True.
    """

    def __init__(self,
                 init_score_thr: float = 0.0001,
                 obj_score_thr: float = 0.0001,
                 match_score_thr: float = 0.5,
                 memo_frames: int = 10,
                 memo_momentum: float = 0.8,
                 momentum_obj_score: float = 0.5,
                 obj_score_diff_thr: float = 1.0,
                 distractor_nms_thr: float = 0.3,
                 distractor_score_thr: float = 0.5,
                 match_metric: str = 'bisoftmax',
                 match_with_cosine: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr

        self.memo_frames = memo_frames
        self.memo_momentum = memo_momentum
        self.momentum_obj_score = momentum_obj_score
        self.obj_score_diff_thr = obj_score_diff_thr
        self.distractor_nms_thr = distractor_nms_thr
        self.distractor_score_thr = distractor_score_thr
        assert match_metric in ['bisoftmax', 'cosine']
        self.match_metric = match_metric
        self.match_with_cosine = match_with_cosine

        self.num_tracks = 0
        self.tracks = dict()

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()

    def update(self, ids: Tensor, bboxes: Tensor, labels: Tensor,
               embeds: Tensor, scores: Tensor, frame_id: int) -> None:
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            scores (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
        """
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, label, score in zip(ids[tracklet_inds],
                                                 bboxes[tracklet_inds],
                                                 embeds[tracklet_inds],
                                                 labels[tracklet_inds],
                                                 scores[tracklet_inds]):
            id = int(id)
            if id in self.tracks:
                self.tracks[id]['bboxes'].append(bbox)
                self.tracks[id]['labels'].append(label)
                self.tracks[id]['scores'].append(score)
                self.tracks[id]['embeds'] = (
                    1 - self.memo_momentum
                ) * self.tracks[id]['embeds'] + self.memo_momentum * embed
                self.tracks[id]['frame_ids'].append(frame_id)
            else:
                self.tracks[id] = dict(
                    bboxes=[bbox],
                    labels=[label],
                    scores=[score],
                    embeds=embed,
                    frame_ids=[frame_id])

        # pop memo
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v['frame_ids'][-1] >= self.memo_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    @property
    def memo(self) -> Tuple[Tensor, ...]:
        """Get tracks memory."""
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_scores = []
        memo_embeds = []
        for k, v in self.tracks.items():
            memo_ids.append(k)
            memo_bboxes.append(v['bboxes'][-1][None, :])
            memo_labels.append(v['labels'][-1].view(1, 1))
            memo_scores.append(v['scores'][-1].view(1, 1))
            memo_embeds.append(v['embeds'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_scores = torch.cat(memo_scores, dim=0).squeeze(1)
        return memo_bboxes, memo_labels, memo_scores, memo_embeds, memo_ids.\
            squeeze(0)

    def track(self,
              model: torch.nn.Module,
              img: torch.Tensor,
              feats: List[torch.Tensor],
              data_sample: TrackDataSample,
              temperature: int = -1,
              rescale=True,
              **kwargs) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                QDTrack method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.
            temperature (int): similarity temperature.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                True.

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
        # create pred_track_instances
        pred_track_instances = InstanceData()
        # return zero bboxes if there is no track targets
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            pred_track_instances = data_sample.pred_det_instances.clone()
            pred_track_instances.instances_id = ids
            return pred_track_instances

        # get track feats
        rescaled_bboxes = bboxes.clone()
        if rescale:
            scale_factor = rescaled_bboxes.new_tensor(
                metainfo['scale_factor']).repeat((1, 2))
            rescaled_bboxes = rescaled_bboxes * scale_factor
        track_feats = model.track_head.predict(feats, [rescaled_bboxes])

        # all objects is valid here
        valid_inds = labels > -1
        # inter-class nms
        low_inds = torch.nonzero(
            scores < self.distractor_score_thr, as_tuple=False).squeeze(1)
        cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
        ious = bbox_overlaps(bboxes[low_inds], bboxes)
        ious *= cat_same.to(ious.device)
        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > self.distractor_nms_thr).any():
                valid_inds[ind] = False
        bboxes = bboxes[valid_inds]
        scores = scores[valid_inds]
        labels = labels[valid_inds]
        embeds = track_feats[valid_inds]

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            memo_bboxes, memo_labels, memo_scores, memo_embeds, memo_ids \
                = self.memo

            if self.match_metric == 'bisoftmax':
                sims = embed_similarity(
                    embeds,
                    memo_embeds,
                    method='dot_product',
                    temperature=temperature)
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                exps = torch.exp(sims) * cat_same.to(sims.device)
                d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                cos_scores = embed_similarity(
                    embeds, memo_embeds, method='cosine')
                cos_scores *= cat_same.to(cos_scores.device)
                match_scores = (d2t_scores + t2d_scores) / 2
                if self.match_with_cosine:
                    match_scores = (match_scores + cos_scores) / 2
            elif self.match_metric == 'cosine':
                cos_scores = embed_similarity(
                    embeds, memo_embeds, method='cosine')
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                match_scores = cos_scores * cat_same.float().to(
                    cos_scores.device)
            else:
                raise NotImplementedError()

            # keep the object score consistency for detection of the same track
            obj_score_diffs = torch.abs(
                scores.view(-1, 1).expand_as(match_scores) -
                memo_scores.view(1, -1).expand_as(match_scores))

            num_objs = bboxes.size(0)
            ids = torch.full((num_objs, ), -1, dtype=torch.long)
            for i in range(num_objs):
                if scores[i] < self.obj_score_thr:
                    continue
                conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                obj_score_diff = obj_score_diffs[i, memo_ind]
                # update track and object score for matched detection
                if (conf > self.match_score_thr) and (obj_score_diff <
                                                      self.obj_score_diff_thr):
                    ids[i] = memo_ids[memo_ind]
                    match_scores[:i, memo_ind] = 0
                    match_scores[i + 1:, memo_ind] = 0
                    m = self.momentum_obj_score
                    scores[i] = m * scores[i] + (1 - m) * memo_scores[memo_ind]
        else:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
        # init tracklets
        new_inds = (ids == -1) & (scores > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long)
        self.num_tracks += num_news

        self.update(ids, bboxes, labels, embeds, scores, frame_id)

        tracklet_inds = ids > -1
        # update pred_track_instances
        pred_track_instances.bboxes = bboxes[tracklet_inds]
        pred_track_instances.labels = labels[tracklet_inds]
        pred_track_instances.scores = scores[tracklet_inds]
        pred_track_instances.instances_id = ids[tracklet_inds]

        return pred_track_instances
