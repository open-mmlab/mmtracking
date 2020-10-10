import mmcv
import numpy as np
from mmdet.datasets import DATASETS

from mmtrack.core import restore_result
from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class MOT17Dataset(CocoVideoDataset):

    CLASSES = ('pedestrian')

    def __init__(self,
                 visibility_thr=-1,
                 detection_file=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.visibility_thr = visibility_thr
        self.detections = self.load_detections(detection_file)

    def load_detections(self, detection_file=None):
        # support detections in three formats
        # 1. MMDet: [img_1, img_2, ...]
        # 2. MMTrack: dict(bbox_results=[img_1, img_2, ...])
        # 3. Public:
        #    1) dict(img1_name: [], img2_name: [], ...)
        #    2) dict(bbox_results=dict(img1_name: [], img2_name: [], ...))
        # return as a dict or a list
        if detection_file is not None:
            detections = mmcv.load(detection_file)
            if isinstance(detections, dict):
                # results from mmtrack
                if 'bbox_results' in detections:
                    detections = detections['bbox_results']
            else:
                # results from mmdet
                if not isinstance(detections, list):
                    raise TypeError('detections must be a dict or a list.')
            return detections
        else:
            return None

    def _parse_detections(self, img_info):
        dets = dict()
        if isinstance(self.detections, list):
            # return by indices of the dataloader
            raise NotImplementedError()
        else:
            assert isinstance(self.detections, dict)
            detections = self.detections[img_info['file_name']]
            public_bboxes, public_labels = restore_result(detections)
            dets['public_bboxes'] = public_bboxes
            dets['public_labels'] = public_labels
        return dets

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
            instance_ids=gt_instance_ids)

        return ann

    def format_track_results(self, results, outfile_prefix=None, **kwargs):
        assert isinstance(results, list), 'results must be a list.'
        pass
        # if outfile_prefix is None:
        #     tmp_dir = tempfile.TemporaryDirectory()
        #     outfile_prefix = osp.join(tmp_dir.name, 'results')
        # else:
        #     tmp_dir = None
        # outfile

    def evaluate(self,
                 results,
                 metric='track',
                 logger=None,
                 outfile_prefix=None,
                 iou_thr=0.5):
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['bbox', 'track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        for metric in metrics:
            if metric == 'bbox':
                results = results['bbox_results']
