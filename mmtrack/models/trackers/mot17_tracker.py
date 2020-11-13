import os
import random
from collections import defaultdict

import cv2
import mmcv
import numpy as np
import seaborn as sns
import torch
from mmcv.image import imread, imwrite
from mmcv.visualization import color_val, imshow
from mmdet.core import bbox_overlaps

from mmtrack.core import embed_similarity
from ..builder import TRACKERS


@TRACKERS.register_module()
class MOT17Tracker(object):

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
                 match_metric='bisoftmax',
                 cosine_factor=0):
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
        self.cosine_factor = cosine_factor

        self.reset()

    def reset(self):
        self.num_tracklets = 0
        self.tracklets = dict()
        self.backdrops = []
        self.pred_tracks = defaultdict(lambda: defaultdict(list))
        self.gt_tracks = defaultdict(lambda: defaultdict(list))

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

    def match(self,
              bboxes,
              labels,
              embeds,
              frame_id,
              temperature=-1,
              **kwargs):
        if embeds is None:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
            return bboxes, labels, ids
        # duplicate removal for potential backdrops and cross classes
        bboxes, labels, embeds = self.nms_cross_classes(bboxes, labels, embeds)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            memo_bboxes, memo_labels, memo_embeds, memo_ids = self.memo
            scores = self.cal_similarity(labels, embeds, memo_labels,
                                         memo_embeds, temperature)
            if self.cosine_factor > 0:
                cos_scores = embed_similarity(
                    embeds, memo_embeds, method='cosine', transpose=True)
                scores = (2 * scores + self.cosine_factor * cos_scores) / (
                    2 + self.cosine_factor)
            raw_scores = scores.clone()
            ids = self.assign_ids(scores, bboxes[:, -1], memo_ids)
        else:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)

        # init tracklets
        ids = self.init_tracklets(ids, bboxes[:, -1])

        self.update_memo(ids, bboxes, labels, embeds, frame_id)

        if 'metas' in kwargs and kwargs['metas'].analyze:
            valid_inds = ids > -1
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            ids = ids[valid_inds]
            metas = kwargs['metas']

            gt_bboxes, gt_labels, gt_ids = [
                metas['bboxes'], metas['labels'], metas['instance_ids']
            ]
            gt_bboxes = torch.cat(
                (gt_bboxes, torch.zeros(gt_bboxes.size(0), 1)), dim=1)

            fns = torch.ones(gt_bboxes.size(0), dtype=torch.long)
            fps = torch.ones(bboxes.size(0), dtype=torch.long)
            idsw = torch.zeros(bboxes.size(0), dtype=torch.long)

            ious = bbox_overlaps(bboxes[:, :4], gt_bboxes[:, :4])

            gt_inds = torch.full(ids.size(), -1, dtype=torch.long)
            for i, bbox in enumerate(bboxes):
                max_iou, j = ious[i].max(dim=0)
                if max_iou > 0.5:
                    fps[i], fns[j] = 0, 0
                    gt_inds[i] = j
                    ious[:, j] = -1

                    gt_id = int(gt_ids[j])
                    pred_id = int(ids[i])
                    if len(self.gt_tracks[gt_id]['ids']) > 0:
                        if pred_id != self.gt_tracks[gt_id]['ids'][-1]:
                            idsw[i] = 1
                    else:
                        if pred_id in self.pred_tracks:
                            idsw[i] = 1
                    self.gt_tracks[gt_id]['scores'].append(
                        float(f'{bbox[-1]:.3f}'))
                    self.gt_tracks[gt_id]['ids'].append(pred_id)
                    self.gt_tracks[gt_id]['frame_ids'].append(
                        metas.img_info['frame_id'])

            for i, id in enumerate(ids):
                id = int(id)

                self.pred_tracks[id]['scores'].append(
                    float(f'{bboxes[i, -1]:.3f}'))
                if metas.img_info['frame_id'] > 0:
                    memo_ind = torch.nonzero(
                        memo_ids == id, as_tuple=False).squeeze(1)
                else:
                    memo_ind = []
                if len(memo_ind) > 0:
                    self.pred_tracks[id]['match_scores'].append(
                        float(f'{raw_scores[i, memo_ind[0]]:.3f}'))
                else:
                    self.pred_tracks[id]['match_scores'].append(-1)
                if gt_inds[i] == -1:
                    self.pred_tracks[id]['ids'].append(-1)
                else:
                    self.pred_tracks[id]['ids'].append(int(gt_ids[gt_inds[i]]))
                self.pred_tracks[id]['frame_ids'].append(
                    metas.img_info['frame_id'])

            # if metas['img_info']['frame_id'] == 5:
            #     import pdb
            #     pdb.set_trace()

            fp_inds = fps == 1  # red
            fn_inds = fns == 1  # yellow
            idsw_inds = idsw == 1  # cyan
            # tp_inds = fps == 0  # green
            # tp_inds[idsw_inds] = 0

            os.makedirs(metas.out_file.rsplit('/', 1)[0], exist_ok=True)
            img = metas.img_name
            # black
            if idsw_inds.any():
                sw_ids = ids[idsw_inds]
                memo_inds = (memo_ids.view(-1, 1) == sw_ids.view(
                    1, -1)).sum(dim=1) > 0
                img = imshow_tracklets(
                    img,
                    memo_bboxes[memo_inds].numpy(),
                    memo_labels[memo_inds].numpy(),
                    memo_ids[memo_inds].numpy(),
                    color='magenta',
                    show=False)
            # img = imshow_tracklets(
            #     img,
            #     bboxes[tp_inds].numpy(),
            #     labels[tp_inds].numpy(),
            #     ids[tp_inds].numpy(),
            #     color='green',
            #     show=False)
            img = imshow_tracklets(
                img,
                bboxes[fp_inds].numpy(),
                labels[fp_inds].numpy(),
                ids[fp_inds].numpy(),
                color='red',
                show=False)
            img = imshow_tracklets(
                img,
                bboxes=gt_bboxes[fn_inds, :].numpy(),
                labels=gt_labels[fn_inds].numpy(),
                color='yellow',
                show=False)
            img = imshow_tracklets(
                img,
                bboxes[idsw_inds].numpy(),
                labels[idsw_inds].numpy(),
                ids[idsw_inds].numpy(),
                color='cyan',
                show=False,
                out_file=metas.out_file)

        return bboxes, labels, ids


def random_color(seed):
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color


def imshow_tracklets(img,
                     bboxes,
                     labels=None,
                     ids=None,
                     thickness=2,
                     font_scale=0.4,
                     show=False,
                     win_name='',
                     color=None,
                     out_file=None):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    # assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    if isinstance(img, str):
        img = imread(img)
    i = 0
    if bboxes.shape[0] == 0:
        if out_file is not None:
            imwrite(img, out_file)
        return img
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.numpy()
        labels = labels.numpy()
        ids = ids.numpy()
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2, _ = bbox.astype(np.int32)
        if ids is not None:
            if color is None:
                bbox_color = random_color(ids[i])
                bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
            else:
                bbox_color = mmcv.color_val(color)
            img[y1:y1 + 12, x1:x1 + 20, :] = bbox_color
            cv2.putText(
                img,
                str(ids[i]), (x1, y1 + 10),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                color=color_val('black'))
        else:
            if color is None:
                bbox_color = color_val('green')
            else:
                bbox_color = mmcv.color_val(color)

        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        if bbox[-1] < 0:
            bbox[-1] = np.nan
        # label_text = '{:.02f}'.format(bbox[-1])
        # img[y1 - 12:y1, x1:x1 + 30, :] = bbox_color
        # cv2.putText(
        #     img,
        #     label_text, (x1, y1 - 2),
        #     cv2.FONT_HERSHEY_COMPLEX,
        #     font_scale,
        #     color=color_val('black'))

        i += 1

    if show:
        imshow(img, win_name)
    if out_file is not None:
        imwrite(img, out_file)

    return img
