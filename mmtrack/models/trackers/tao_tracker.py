import torch
from mmdet.core import bbox_overlaps

from mmtrack.core import embed_similarity
from ..builder import TRACKERS


@TRACKERS.register_module()
class TaoTracker(object):

    def __init__(self,
                 init_score_thr=0.8,
                 obj_score_thr=0.5,
                 match_score_thr=0.5,
                 memo_tracklet_frames=10,
                 memo_backdrop_frames=1,
                 memo_momentum=0.8,
                 nms_conf_thr=0.5,
                 nms_backdrop_iou_thr=0.3,
                 nms_class_iou_thr=0.7,
                 with_cats=True,
                 match_metric='bisoftmax'):
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        assert match_metric in ['bisoftmax', 'softmax', 'cosine']
        self.match_metric = match_metric

        self.reset()

    def reset(self):
        self.num_tracklets = 0
        self.tracklets = dict()
        self.backdrops = []

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, labels, embeds, frame_id):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, label in zip(ids[tracklet_inds],
                                          bboxes[tracklet_inds],
                                          embeds[tracklet_inds],
                                          labels[tracklet_inds]):
            id = int(id)
            if id in self.tracklets:
                self.tracklets[id]['bboxes'].append(bbox)
                self.tracklets[id]['labels'].append(label)
                self.tracklets[id]['embeds'] = (
                    1 - self.memo_momentum
                ) * self.tracklets[id]['embeds'] + self.memo_momentum * embed
                self.tracklets[id]['frame_ids'].append(frame_id)
            else:
                self.tracklets[id] = dict(
                    bboxes=[bbox],
                    labels=[label],
                    embeds=embed,
                    frame_ids=[frame_id])

        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_overlaps(bboxes[backdrop_inds, :-1], bboxes[:, :-1])
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds]))

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v['frame_ids'][-1] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_embeds = []
        for k, v in self.tracklets.items():
            memo_ids.append(k)
            memo_bboxes.append(v['bboxes'][-1][None, :])
            memo_labels.append(v['labels'][-1].view(1, 1))
            memo_embeds.append(v['embeds'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        for backdrop in self.backdrops:
            backdrop_ids = torch.full((1, backdrop['embeds'].size(0)),
                                      -1,
                                      dtype=torch.long)
            memo_ids = torch.cat([memo_ids, backdrop_ids], dim=1)
            memo_bboxes.append(backdrop['bboxes'])
            memo_labels.append(backdrop['labels'][:, None])
            memo_embeds.append(backdrop['embeds'])

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        return memo_bboxes, memo_labels, memo_embeds, memo_ids.squeeze(0)

    def nms_cross_classes(self, bboxes, labels, embeds):
        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.size(0)):
            thr = self.nms_backdrop_iou_thr if bboxes[
                i, -1] < self.obj_score_thr else self.nms_class_iou_thr
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        embeds = embeds[valids, :]
        return bboxes, labels, embeds

    def cal_similarity(self,
                       labels,
                       embeds,
                       memo_labels,
                       memo_embeds,
                       temperature=-1):
        if self.match_metric == 'bisoftmax':
            sims = embed_similarity(
                embeds,
                memo_embeds,
                method='dot_product',
                temperature=temperature,
                transpose=True)
            d2t_scores = sims.softmax(dim=1)
            t2d_scores = sims.softmax(dim=0)
            scores = (d2t_scores + t2d_scores) / 2
        elif self.match_metric == 'softmax':
            sims = embed_similarity(
                embeds,
                memo_embeds,
                method='dot_product',
                temperature=temperature,
                transpose=True)
            scores = sims.softmax(dim=1)
        elif self.match_metric == 'cosine':
            scores = embed_similarity(
                embeds, memo_embeds, method='cosine', transpose=True)
        else:
            raise NotImplementedError()

        if self.with_cats:
            cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
            scores *= cat_same.float()

        return scores

    def assign_ids(self, scores, obj_scores, memo_ids):
        num_objs = obj_scores.size(0)
        ids = torch.full((num_objs, ), -1, dtype=torch.long)
        for i in range(num_objs):
            conf, memo_ind = torch.max(scores[i, :], dim=0)
            id = memo_ids[memo_ind]
            if conf > self.match_score_thr:
                if id > -1:
                    if obj_scores[i] > self.obj_score_thr:
                        ids[i] = id
                        scores[:i, memo_ind] = 0
                        scores[i + 1:, memo_ind] = 0
                    else:
                        if conf > self.nms_conf_thr:
                            ids[i] = -2
        return ids

    def init_tracklets(self, ids, obj_scores):
        new_objs = (ids == -1) & (obj_scores > self.init_score_thr).cpu()
        num_new_objs = new_objs.sum()
        ids[new_objs] = torch.arange(
            self.num_tracklets,
            self.num_tracklets + num_new_objs,
            dtype=torch.long)
        self.num_tracklets += num_new_objs
        return ids

    def match(self, bboxes, labels, embeds, frame_id, temperature=-1):
        if embeds is None:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
            return bboxes, labels, ids
        # duplicate removal for potential backdrops and cross classes
        bboxes, labels, embeds = self.nms_cross_classes(bboxes, labels, embeds)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            memo_bboxes, memo_labels, memo_embeds, memo_ids = self.memo
            scores = self.cal_similarity(labels, embeds, memo_labels,
                                         memo_embeds)
            ids = self.assign_ids(scores, bboxes[:, -1], memo_ids)
        else:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)

        # init tracklets
        ids = self.init_tracklets(ids, bboxes[:, -1])

        self.update_memo(ids, bboxes, labels, embeds, frame_id)

        return bboxes, labels, ids
