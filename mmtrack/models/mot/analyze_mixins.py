import os

import mmcv
import torch
from mmdet.core import bbox_overlaps


class AnalyzeMixin(object):

    def analyze(self,
                img_meta,
                bboxes,
                labels,
                ids,
                show=False,
                save=False,
                gt_cats=None):
        gt_bboxes, gt_labels, gt_ids, gt_ignores = self.loadGts(
            img_meta, gt_cats)
        track_inds = ids > -1
        track_bboxes = bboxes[track_inds]
        track_labels = labels[track_inds]
        track_ids = ids[track_inds]
        if len(gt_ignores) > 0:
            ignore_inds = (bbox_overlaps(
                bboxes[:, :4], gt_ignores, mode='iof') > 0.5).any(dim=1)
        if track_bboxes.size(0) == 0:
            self.counter.num_fn += gt_bboxes.size(0)
            return
        if gt_bboxes.size(0) == 0:
            self.counter.num_fp += track_bboxes.size(0)
            if gt_ignores.size(0) > 0:
                self.counter.num_fp -= ignore_inds[track_inds].sum()
            return
        # init
        # [N, 6]: [x1, y1, x2, y2, class, id]
        self.counter.num_gt += gt_bboxes.size(0)
        fps = torch.ones(bboxes.size(0), dtype=torch.long)
        fns = torch.ones(gt_bboxes.size(0), dtype=torch.long)
        # false negatives after tracking filter
        track_fns = torch.ones(gt_bboxes.size(0), dtype=torch.long)
        idsw = torch.zeros(track_ids.size(0), dtype=torch.long)

        # fp & fn for raw detection results
        ious = bbox_overlaps(bboxes[:, :4], gt_bboxes[:, :4])
        same_cat = labels.view(-1, 1) == gt_labels.view(1, -1)
        ious *= same_cat.float()
        max_ious, gt_inds = ious.max(dim=1)
        _, dt_inds = bboxes[:, -1].sort(descending=True)
        for dt_ind in dt_inds:
            iou, gt_ind = max_ious[dt_ind], gt_inds[dt_ind]
            if iou > 0.5 and fns[gt_ind] == 1:
                fns[gt_ind] = 0
                if ids[dt_ind] > -1:
                    track_fns[gt_ind] = 0
                gt_bboxes[gt_ind, 4] = bboxes[dt_ind, -1]
                fps[dt_ind] = 0
            else:
                if len(gt_ignores) > 0 and ignore_inds[dt_ind]:
                    fps[dt_ind] = 0
                    gt_inds[dt_ind] = -2
                else:
                    gt_inds[dt_ind] = -1

        track_gt_inds = gt_inds[track_inds]
        track_fps = fps[track_inds]

        for i, id in enumerate(track_ids):
            id = int(id)
            gt_ind = track_gt_inds[i]
            if gt_ind == -1 or gt_ind == -2:
                continue
            gt_id = int(gt_ids[gt_ind])
            if gt_id in self.id_maps.keys() and self.id_maps[gt_id] != id:
                idsw[i] = 1
            if gt_id not in self.id_maps.keys() and id in self.id_maps.values(
            ):
                idsw[i] = 1
            self.id_maps[gt_id] = id

        fp_inds = track_fps == 1
        fn_inds = track_fns == 1
        idsw_inds = idsw == 1
        self.counter.num_fp += fp_inds.sum()
        self.counter.num_fn += fn_inds.sum()
        self.counter.num_idsw += idsw_inds.sum()

        if show or save:
            vid_name, img_name = img_meta[0]['img_info']['file_name'].split(
                '/')
            img = os.path.join(self.data.img_prefix, vid_name, img_name)
            save_path = os.path.join(self.out, 'analysis', vid_name)
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, img_name) if save else None
            img = mmcv.imshow_tracklets(
                img,
                track_bboxes[fp_inds].numpy(),
                track_labels[fp_inds].numpy(),
                track_ids[fp_inds].numpy(),
                color='red',
                show=False)
            img = mmcv.imshow_tracklets(
                img,
                bboxes=gt_bboxes[fn_inds, :].numpy(),
                labels=gt_labels[fn_inds].numpy(),
                color='yellow',
                show=False)
            img = mmcv.imshow_tracklets(
                img,
                track_bboxes[idsw_inds].numpy(),
                track_labels[idsw_inds].numpy(),
                track_ids[idsw_inds].numpy(),
                color='cyan',
                show=show,
                out_file=save_file)

    def loadGts(self, img_meta, gt_cats=None):
        vid = self.dataset.vid
        img_id = img_meta[0]['img_info']['id']
        ann_ids = vid.getAnnIds(img_id)
        anns = vid.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_ignores = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            if gt_cats is not None and ann['category_id'] not in gt_cats:
                continue
            if ann['iscrowd'] or ann['ignore']:
                gt_ignores.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(ann['category_id'] - 1)
                gt_ids.append(ann['instance_id'])
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float)
        gt_bboxes = torch.cat((gt_bboxes, torch.zeros(gt_bboxes.size(0), 1)),
                              dim=1)
        gt_labels = torch.tensor(gt_labels, dtype=torch.long)
        gt_ids = torch.tensor(gt_ids, dtype=torch.long)
        gt_ignores = torch.tensor(gt_ignores, dtype=torch.float)
        return gt_bboxes, gt_labels, gt_ids, gt_ignores

    def save_pkl(self,
                 img_meta,
                 det_bboxes,
                 det_labels,
                 embeds,
                 bboxes=None,
                 cls_logits=None,
                 keep_inds=None):
        vid_name, img_name = img_meta[0]['img_info']['file_name'].split('/')
        save_path = os.path.join(self.out, 'pkls', vid_name)
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, '{}.pkl'.format(img_name))
        to_save = dict(
            det_bboxes=det_bboxes.cpu(),
            det_labels=det_labels.cpu(),
            bboxes=bboxes.cpu() if bboxes else None,
            embeds=embeds.cpu(),
            keep_inds=keep_inds.cpu() if keep_inds else None,
            cls_logits=cls_logits.cpu() if cls_logits else None)
        mmcv.dump(to_save, save_file)

    def show_tracklets(self, img_meta, track_bboxes, track_labels, track_ids):
        vid_name, img_name = img_meta[0]['img_info']['file_name'].split('/')
        save_path = os.path.join(self.out, 'shows', vid_name)
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, img_name)
        img = os.path.join(self.data.img_prefix, vid_name, img_name)
        img = mmcv.imshow_tracklets(
            img, track_bboxes, track_labels, track_ids, out_file=save_file)

    # def plt_tracklets(self, img_meta, track_bboxes, track_labels, track_ids):
    #     vid_name, img_name = img_meta[0]['img_info']['file_name'].split('/')
    #     save_path = os.path.join(self.out, 'shows', vid_name)
    #     os.makedirs(save_path, exist_ok=True)
    #     save_file = os.path.join(save_path, img_name.split('-')[-1])
    #     img = os.path.join(self.data.img_prefix, vid_name, img_name)
    #     # car_inds = track_labels == 2
    #     # img = imshow_bboxes_w_ids(
    #     #     img,
    #     #     track_bboxes[car_inds],
    #     #     track_ids[car_inds],
    #     #     out_file=save_file)
    #     img = imshow_bboxes_w_ids(
    #         img, track_bboxes, track_ids, out_file=save_file)
