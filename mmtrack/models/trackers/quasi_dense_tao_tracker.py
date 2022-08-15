# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox_overlaps

from mmtrack.core import embed_similarity
from ..builder import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
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
                 init_score_thr=0.0001,
                 obj_score_thr=0.0001,
                 match_score_thr=0.5,
                 memo_frames=10,
                 memo_momentum=0.8,
                 momentum_obj_score=0.5,
                 obj_score_diff_thr=1.0,
                 distractor_nms_thr=0.3,
                 distractor_score_thr=0.5,
                 match_metric='bisoftmax',
                 match_with_cosine=True,
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

    def update(self, ids, bboxes, labels, embeds, frame_id):
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
        """
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, label in zip(ids[tracklet_inds],
                                          bboxes[tracklet_inds],
                                          embeds[tracklet_inds],
                                          labels[tracklet_inds]):
            id = int(id)
            if id in self.tracks:
                self.tracks[id]['bboxes'].append(bbox)
                self.tracks[id]['labels'].append(label)
                self.tracks[id]['embeds'] = (
                    1 - self.memo_momentum
                ) * self.tracks[id]['embeds'] + self.memo_momentum * embed
                self.tracks[id]['frame_ids'].append(frame_id)
            else:
                self.tracks[id] = dict(
                    bboxes=[bbox],
                    labels=[label],
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
    def memo(self):
        """Get tracks memory."""
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_embeds = []
        for k, v in self.tracks.items():
            memo_ids.append(k)
            memo_bboxes.append(v['bboxes'][-1][None, :])
            memo_labels.append(v['labels'][-1].view(1, 1))
            memo_embeds.append(v['embeds'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        return memo_bboxes, memo_labels, memo_embeds, memo_ids.squeeze(0)

    def track(self,
              img_metas,
              feats,
              model,
              bboxes,
              labels,
              frame_id,
              temperature=-1,
              **kwargs):
        """Tracking forward function.

        Args:
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            feats (tuple): Backbone features of the input image.
            model (nn.Module): The forward model.
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            temperature (int): similarity temperature.

        Returns:
            list: Tracking results.
        """
        # return zero bboxes if there is no track targets
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            return bboxes, labels, ids
        # get track feats
        track_bboxes = bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(bboxes.device)
        track_feats = model.track_head.extract_bbox_feats(
            feats, [track_bboxes])
        # all objects is valid here
        valid_inds = labels > -1
        # inter-class nms
        low_inds = torch.nonzero(
            bboxes[:, -1] < self.distractor_score_thr,
            as_tuple=False).squeeze(1)
        cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
        ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
        ious *= cat_same.to(ious.device)
        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > self.distractor_nms_thr).any():
                valid_inds[ind] = False
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        embeds = track_feats[valid_inds]

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            memo_bboxes, memo_labels, memo_embeds, memo_ids = self.memo

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
                scores = (d2t_scores + t2d_scores) / 2
                if self.match_with_cosine:
                    scores = (scores + cos_scores) / 2
            elif self.match_metric == 'cosine':
                cos_scores = embed_similarity(
                    embeds, memo_embeds, method='cosine')
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                scores = cos_scores * cat_same.float().to(cos_scores.device)
            else:
                raise NotImplementedError()

            # keep the object score consistency for detection of the same track
            obj_score_diffs = torch.abs(
                bboxes[:, -1].view(-1, 1).expand_as(scores) -
                memo_bboxes[:, -1].view(1, -1).expand_as(scores))

            num_objs = bboxes.size(0)
            ids = torch.full((num_objs, ), -1, dtype=torch.long)
            for i in range(num_objs):
                if bboxes[i, -1] < self.obj_score_thr:
                    continue
                conf, memo_ind = torch.max(scores[i, :], dim=0)
                obj_score_diff = obj_score_diffs[i, memo_ind]
                # update track and object score for matched detection
                if (conf > self.match_score_thr) and (obj_score_diff <
                                                      self.obj_score_diff_thr):
                    ids[i] = memo_ids[memo_ind]
                    scores[:i, memo_ind] = 0
                    scores[i + 1:, memo_ind] = 0
                    m = self.momentum_obj_score
                    bboxes[i, -1] = m * bboxes[i, -1] + (
                        1 - m) * memo_bboxes[memo_ind, -1]
        else:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
        # init tracklets
        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long)
        self.num_tracks += num_news

        self.update(ids, bboxes, labels, embeds, frame_id)

        return bboxes, labels, ids
