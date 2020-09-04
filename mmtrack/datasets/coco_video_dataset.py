import random
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS, CocoDataset
from mmdet.datasets.pipelines import Compose
from pycocotools.coco import COCO

from mmtrack.utils import get_root_logger
from .parsers import CocoVID


@DATASETS.register_module()
class CocoVideoDataset(CocoDataset):

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_prefix=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 key_img_sampler=dict(interval=1),
                 ref_img_sampler=dict(scope=3)):
        self.logger = get_root_logger()
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.key_img_sampler = key_img_sampler
        self.ref_img_sampler = ref_img_sampler

        self.data_infos = self.load_annotations(self.ann_file)
        self.pipeline = Compose(pipeline)
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            self._set_group_flag()

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def _filter_imgs(self, min_size=32):
        valid_inds = []

        if self.num_vid_imgs > 0:
            vid_img_ids = set(_['image_id'] for _ in self.VID.anns.values())
            for i, img_info in enumerate(self.data_infos[:self.num_vid_imgs]):
                if self.filter_empty_gt and (
                        img_info['id'] not in vid_img_ids):
                    continue
                if min(img_info['width'], img_info['height']) >= min_size:
                    valid_inds.append(i)

        if self.num_det_imgs > 0:
            det_img_ids = set(_['image_id'] for _ in self.coco.anns.values())
            for i, img_info in enumerate(self.data_infos[self.num_vid_imgs:]):
                if self.filter_empty_gt and img_info['id'] not in det_img_ids:
                    continue
                if min(img_info['width'], img_info['height']) >= min_size:
                    valid_inds.append(i + self.num_vid_imgs)

        return valid_inds

    def key_img_sampling(self, vid_id, interval=1):
        img_ids = self.VID.get_img_ids_from_vid(vid_id)
        if not self.test_mode:
            img_ids = img_ids[::interval]
        return img_ids

    def ref_img_sampling(self, img_info, scope, num=1, method='uniform'):
        assert num == 1
        if scope > 0:
            vid_id = img_info['video_id']
            img_ids = self.VID.get_img_ids_from_vid(vid_id)
            frame_id = img_info['frame_id']
            if method == 'uniform':
                left = max(0, frame_id - scope)
                right = min(frame_id + scope, len(img_ids) - 1)
                valid_inds = img_ids[left:frame_id] + img_ids[frame_id +
                                                              1:right + 1]
                ref_img_id = random.choice(valid_inds)
            else:
                raise NotImplementedError(
                    'Only uniform sampling is supported now.')
            ref_img_info = self.VID.loadImgs([ref_img_id])[0]
            ref_img_info['filename'] = ref_img_info['file_name']
            ref_img_info['type'] = 'VID'
        else:
            ref_img_info = img_info.copy()
        return ref_img_info

    def load_video_anns(self, ann_file):
        data_infos = []

        self.VID = CocoVID(ann_file)

        self.cat_ids = self.VID.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.vid_ids = self.VID.get_vid_ids()
        for vid_id in self.vid_ids:
            img_ids = self.key_img_sampling(vid_id, **self.key_img_sampler)
            for img_id in img_ids:
                info = self.VID.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                info['type'] = 'VID'
                data_infos.append(info)

        return data_infos

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        mode = 'TEST' if self.test_mode else 'TRAIN'
        if isinstance(ann_file, str):
            print_log(
                f'Default loading {ann_file} as video dataset.',
                logger=self.logger)
            ann_file = dict(VID=ann_file)
        elif isinstance(ann_file, dict):
            for k in ann_file.keys():
                if k not in ['VID', 'DET']:
                    raise ValueError('Keys must be DET or VID.')
        else:
            raise TypeError('ann_file must be a str or dict.')

        data_infos = []

        if 'VID' in ann_file.keys():
            vid_data_infos = self.load_video_anns(ann_file['VID'])
            data_infos.extend(vid_data_infos)
        self.num_vid_imgs = len(data_infos)

        if 'DET' in ann_file.keys():
            det_data_infos = super().load_annotations(ann_file['DET'])
            for info in det_data_infos:
                info['type'] = 'DET'
            data_infos.extend(det_data_infos)
        self.num_det_imgs = len(data_infos) - self.num_vid_imgs

        print_log((f"{mode}: Load {self.num_vid_imgs} images from VID set "
                   f"and {self.num_det_imgs} images from DET set."),
                  logger=self.logger)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_info = self.data_infos[idx]
        img_id = img_info['id']
        api = self.coco if img_info['type'] == 'DET' else self.VID
        ann_ids = api.get_ann_ids(img_ids=[img_id])
        ann_info = api.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def match_gts(self, ann, ref_ann):
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
        gt_masks_ann = []
        if img_info['type'] == 'VID':
            gt_instance_ids = []

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
                if ann.get('segmentation', False):
                    gt_masks_ann.append(ann['segmentation'])
                if ann.get('instance_id', False):
                    gt_instance_ids.append(ann['instance_id'])

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

        if img_info['type'] == 'VID':
            ann['instance_ids'] = gt_instance_ids

        return ann

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
            ref_img_info = self.ref_img_sampling(img_info,
                                                 **self.ref_img_sampler)
            ref_ann_ids = self.VID.get_ann_ids(img_ids=[ref_img_info['id']])
            ref_ann_info = self.VID.load_anns(ref_ann_ids)
            ref_ann_info = self._parse_ann_info(ref_img_info, ref_ann_info)
            ref_results = dict(img_info=ref_img_info, ann_info=ref_ann_info)
        else:
            ref_results = results.copy()

        mids, ref_mids = self.match_gts(results['ann_info'],
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
