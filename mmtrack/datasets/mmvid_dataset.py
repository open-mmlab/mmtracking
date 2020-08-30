import numpy as np
from torch.utils.data import Dataset

from mmdet.datasets import DATASETS, CustomDataset
from mmdet.datasets.pipelines import Compose
from .parsers import MmVID
from pycocotools.coco import COCO
from mmcv.utils import print_log
from mmtrack.utils import get_root_logger


# TODO: add classes filter
@DATASETS.register_module()
class MmVIDDataset(CustomDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_prefix=None,
                 sample_ref=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.logger = get_root_logger()
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.sample_ref = sample_ref
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.data_infos = self.load_annotations(self.ann_file)
        self.pipeline = Compose(pipeline)
        if not test_mode:
            self.data_infos = self._filter_imgs()
            self._set_group_flag()

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        data_infos = []
        for i, data_info in enumerate(self.data_infos):
            if min(data_info['width'], data_info['height']) >= min_size:
                data_infos.append(data_info)
        return data_infos

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        mode = 'TEST' if self.test_mode else 'TRAIN'

        if isinstance(ann_file, dict):
            data_infos = []
            self.VID = MmVID(ann_file['VID'])
            data_infos.extend(self.VID.images)

            self.DET = COCO(ann_file['DET'])
            self.cat_ids = self.DET.get_cat_ids(cat_names=self.CLASSES)
            self.cat2label = {
                cat_id: i
                for i, cat_id in enumerate(self.cat_ids)
            }
            img_ids = self.DET.get_img_ids()
            for img_id in img_ids:
                data_info = self.DET.load_imgs([img_id])[0]
                data_info['type'] = 'DET'
                data_info['filename'] = data_info['file_name']
                data_infos.append(data_info)
            num_vid_imgs = len(self.VID.images)
            num_det_imgs = len(img_ids)
            self.logger.info(
                (f"{mode}: Joint {num_vid_imgs} images from VID set "
                 f"and {num_det_imgs} images from DET set."))
            return data_infos
        else:
            self.VID = MmVID(ann_file)
            num_vid_imgs = len(self.VID.images)
            self.logger.info(f'{mode}: {num_vid_imgs} images from VID set.')
            return self.VID.images

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_info = self.data_infos[idx]
        if img_info['type'] == 'DET':
            img_id = img_info['id']
            ann_ids = self.DET.get_ann_ids(img_ids=[img_id])
            ann_info = self.DET.load_anns(ann_ids)
            return self._parse_ann_info(self.data_infos[idx], ann_info)
        elif img_info['type'] == 'VID':
            anns = self.VID.parse_anns(
                img_info['annotations'], ignore_keys=['ignore', 'crowd'])
            return anns
        else:
            raise ValueError('Type must be DET or VID. ')

    def _parse_ann_info(self, img_info, ann_info):
        # TODO: remove this def
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
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
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
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def matching(self, ann, ref_ann):
        if 'instance_ids' in ann.keys():
            gt_instances = list(ann['instance_ids'])
            ref_instances = list(ref_ann['instance_ids'])
            gt_pids = np.array([
                ref_instances.index(i) if i in ref_instances else -1
                for i in gt_instances
            ])
            ref_gt_pids = np.array([
                gt_instances.index(i) if i in gt_instances else -1
                for i in ref_instances
            ])
        else:
            gt_pids = np.arange(ann['bboxes'].shape[0], dtype=np.int64)
            ref_gt_pids = gt_pids.copy()
        return gt_pids, ref_gt_pids

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        for _results in results:
            _results['img_prefix'] = self.img_prefix[_results['img_info']
                                                     ['type']]
            _results['bbox_fields'] = []

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if img_info['type'] == 'VID':
            ref_img_info = self.VID.get_ref_img(idx, **self.sample_ref)
            ref_ann_info = self.VID.parse_anns(ref_img_info['annotations'])
            ref_results = dict(img_info=ref_img_info, ann_info=ref_ann_info)
        else:
            ref_results = results.copy()

        mids, ref_mids = self.matching(results['ann_info'],
                                       ref_results['ann_info'])
        if (mids == -1).all():
            return None
        else:
            results['ann_info']['mids'] = mids
            ref_results['ann_info']['mids'] = ref_mids
            self.pre_pipeline([results, ref_results])
        return self.pipeline([results, ref_results])

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        pass

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        pass
