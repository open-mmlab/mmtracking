import numpy as np
from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class MOT17Dataset(CocoVideoDataset):

    CLASSES = ('pedestrian')

    def __init__(self, visibility_thr=-1, *args, **kwargs):
        self.visibility_thr = visibility_thr
        super().__init__(*args, **kwargs)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_instance_ids = []

        for i, ann in enumerate(ann_info):
            if ann['detected']:
                if ann['bboxes']:
                    public_bboxes = np.array(ann['bboxes'], dtype=np.float32)
                else:
                    public_bboxes = np.zeros((0, 5), dtype=np.float32)
            else:
                if ann['visibility'] < self.visibility_thr:
                    continue
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                bbox = [x1, y1, x1 + w, y1 + h]
                if ann.get('iscrowd', False) or ann.get('ignore', False):
                    gt_bboxes_ignore.append(bbox)
                else:
                    gt_bboxes.append(bbox)
                    gt_labels.append(self.cat2label[ann['category_id']])
                    gt_instance_ids.append(ann['instance_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_instance_ids = np.array(gt_instance_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_instance_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            instance_ids=gt_instance_ids,
            public_bboxes=public_bboxes)

        return ann

    def format_track_results(self, results, **kwargs):
        pass
