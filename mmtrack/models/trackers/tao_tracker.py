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
class TaoTracker(object):

    def __init__(self,
                 init_score_thr=0.0001,
                 obj_score_thr=0.0001,
                 match_score_thr=0.5,
                 memo_frames=10,
                 momentum_embed=0.8,
                 momentum_obj_score=0.5,
                 obj_score_diff_thr=1.0,
                 distractor_nms_thr=0.3,
                 distractor_score_thr=0.5,
                 match_metric='bisoftmax',
                 match_with_cosine=True):
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr

        self.memo_frames = memo_frames
        self.momentum_embed = momentum_embed
        self.momentum_obj_score = momentum_obj_score
        self.obj_score_diff_thr = obj_score_diff_thr
        self.distractor_nms_thr = distractor_nms_thr
        self.distractor_score_thr = distractor_score_thr
        assert match_metric in ['bisoftmax', 'cosine']
        self.match_metric = match_metric
        self.match_with_cosine = match_with_cosine

        self.reset()

    def reset(self):
        self.num_tracklets = 0
        self.tracklets = dict()
        # for analysis
        self.pred_tracks = defaultdict(lambda: defaultdict(list))
        self.gt_tracks = defaultdict(lambda: defaultdict(list))

    @property
    def valid_ids(self):
        valid_ids = []
        for k, v in self.gt_tracks.items():
            valid_ids.extend(v['ids'])
        return list(set(valid_ids))

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
                    1 - self.momentum_embed
                ) * self.tracklets[id]['embeds'] + self.momentum_embed * embed
                self.tracklets[id]['frame_ids'].append(frame_id)
            else:
                self.tracklets[id] = dict(
                    bboxes=[bbox],
                    labels=[label],
                    embeds=embed,
                    frame_ids=[frame_id])

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v['frame_ids'][-1] >= self.memo_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

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

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        return memo_bboxes, memo_labels, memo_embeds, memo_ids.squeeze(0)

    def init_tracklets(self, ids, obj_scores):
        new_objs = (ids == -1) & (obj_scores > self.init_score_thr).cpu()
        num_new_objs = new_objs.sum()
        ids[new_objs] = torch.arange(
            self.num_tracklets,
            self.num_tracklets + num_new_objs,
            dtype=torch.long)
        self.num_tracklets += num_new_objs
        return ids

    def track(self,
              bboxes,
              labels,
              track_feats,
              frame_id,
              temperature=-1,
              **kwargs):
        if track_feats is None:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
            return bboxes, labels, ids

        # all objects is valid here
        valid_inds = labels > -1
        # nms
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
            if 'metas' in kwargs:
                raw_scores = scores.clone()

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
        ids = self.init_tracklets(ids, bboxes[:, -1])
        self.update_memo(ids, bboxes, labels, embeds, frame_id)

        # ----------------
        if 'metas' in kwargs and kwargs['metas'].analyze:
            metas = kwargs['metas']
            gt_bboxes, gt_labels, gt_ids = [
                metas['bboxes'], metas['labels'], metas['instance_ids']
            ]
            gt_bboxes = torch.cat(
                (gt_bboxes, torch.zeros(gt_bboxes.size(0), 1)), dim=1)

            if bboxes.size(0) == 0 or gt_bboxes.size(0) == 0:
                return bboxes, labels, ids

            fns = torch.ones(gt_bboxes.size(0), dtype=torch.long)
            fps = torch.ones(bboxes.size(0), dtype=torch.long)
            sw_fps = torch.zeros(bboxes.size(0), dtype=torch.long)
            idsw = torch.zeros(bboxes.size(0), dtype=torch.long)

            ious = bbox_overlaps(bboxes[:, :4], gt_bboxes[:, :4])
            same_cat = labels.view(-1, 1) == gt_labels.view(1, -1)
            ious *= same_cat.float().to(ious.device)

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

                if fps[i]:
                    if id in self.valid_ids:
                        sw_fps[i] = 1
                    continue

            fp_inds = sw_fps == 1  # red
            fn_inds = fns == 1  # yellow
            idsw_inds = idsw == 1  # cyan
            tp_inds = fps == 0  # green
            tp_inds[idsw_inds] = 0

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
            img = imshow_tracklets(
                img,
                bboxes[tp_inds].numpy(),
                labels[tp_inds].numpy(),
                ids[tp_inds].numpy(),
                color='green',
                show=False)
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
