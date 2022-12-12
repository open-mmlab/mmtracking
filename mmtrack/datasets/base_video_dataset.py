# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import random
from typing import Any, List, Tuple

from mmdet.datasets.api_wrappers import COCO
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import FileClient
from mmengine.logging import MMLogger

from mmtrack.registry import DATASETS
from .api_wrappers import CocoVID


@DATASETS.register_module()
class BaseVideoDataset(BaseDataset):
    """Base video dataset for VID, MOT and VIS tasks, except for SOT tasks.

    Args:
        load_as_video (bool, optional): Load data as videos or images.
            Defaults to True.
        key_img_sampler (dict, optional): Configuration of sampling key images.
            Defaults to dict(interval=1).
        ref_img_sampler (dict, optional): Configuration of sampling
            reference images.
            - num_ref_imgs (int, optional): The number of sampled reference
                images. Defaults to 2.
            - frame_range (List(int) | int, optional): The sampling range of
                reference frames in the same video for key frame.
                Defaults to 9.
            - filter_key_img (bool, optional): If False, the key image will be
                in the sampling reference candidates, otherwise, it is exclude.
                Defaults to True.
            - method (str, optional): The sampling method. Options are
                'uniform', 'bilateral_uniform', 'test_with_adaptive_stride',
                'test_with_fix_stride'. Defaults to 'bilateral_uniform'.
    """
    META = dict(classes=None)

    def __init__(self,
                 load_as_video: bool = True,
                 key_img_sampler: dict = dict(interval=1),
                 ref_img_sampler: dict = dict(
                     num_ref_imgs=2,
                     frame_range=9,
                     filter_key_img=True,
                     method='bilateral_uniform'),
                 *args,
                 **kwargs):
        self.load_as_video = load_as_video
        self.key_img_sampler = key_img_sampler
        self.ref_img_sampler = ref_img_sampler
        super().__init__(*args, **kwargs)

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - serialize_data: Serialize ``self.data_list`` if
            ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # Load data information.
        # We use `self.valid_data_inds` to record the ids of `data_list` used
        # for training and testing.
        self.data_list, self.valid_data_indices = self.load_data_list()
        # Filter illegal data, such as data that has no annotations.
        self.valid_data_indices = self.filter_data()

        # Serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def load_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from an annotation file named as ``self.ann_file``.
        Specifically, if self.load_as_video is True, it loads from the video
        annotation file. Otherwise, from the image annotation file.

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        if self.load_as_video:
            data_list, valid_data_indices = self._load_video_data_list()
        else:
            data_list, valid_data_indices = self._load_image_data_list()

        return data_list, valid_data_indices

    def _load_video_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from a video annotation file named as
        ``self.ann_file``.

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        file_client = FileClient.infer_client(uri=self.ann_file)
        with file_client.get_local_path(self.ann_file) as local_path:
            coco = CocoVID(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the classes
        self.cat_ids = coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(coco.cat_img_map)

        data_list = []
        valid_data_indices = []
        data_id = 0
        vid_ids = coco.get_vid_ids()
        for vid_id in vid_ids:
            img_ids = coco.get_img_ids_from_vid(vid_id)
            for img_id in img_ids:
                # load img info
                raw_img_info = coco.load_imgs([img_id])[0]
                raw_img_info['img_id'] = img_id
                raw_img_info['video_length'] = len(img_ids)

                # load ann info
                ann_ids = coco.get_ann_ids(
                    img_ids=[img_id], cat_ids=self.cat_ids)
                raw_ann_info = coco.load_anns(ann_ids)

                if (self.key_img_sampler is not None) and (
                        raw_img_info['frame_id'] %
                        self.key_img_sampler.get('interval', 1) == 0):
                    valid_data_indices.append(data_id)
                # get data_info
                parsed_data_info = self.parse_data_info(
                    dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))
                data_list.append(parsed_data_info)
                data_id += 1

        return data_list, valid_data_indices

    def _load_image_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from an image annotation file named as
        ``self.ann_file``.

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        file_client = FileClient.infer_client(uri=self.ann_file)
        with file_client.get_local_path(self.ann_file) as local_path:
            coco = COCO(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the classes
        self.cat_ids = coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(coco.cat_img_map)

        data_list = []
        valid_data_indices = []
        data_id = 0
        img_ids = coco.get_img_ids()
        total_ann_ids = []
        for img_id in img_ids:
            # load img info
            raw_img_info = coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            # load ann info
            ann_ids = coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids)
            raw_ann_info = coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            # load images for training
            valid_data_indices.append(data_id)

            # get data_info
            parsed_data_info = self.parse_data_info(
                dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))
            data_list.append(parsed_data_info)
            data_id += 1

        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f"Annotation ids in '{self.ann_file}' are not unique!"

        return data_list, valid_data_indices

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            dict: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info = {}

        data_info.update(img_info)
        if self.data_prefix.get('img_path', None) is not None:
            img_path = osp.join(self.data_prefix['img_path'],
                                img_info['file_name'])
        else:
            img_path = img_info['file_name']
        data_info['img_path'] = img_path

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

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
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            if ann.get('instance_id', None):
                instance['instance_id'] = ann['instance_id']
            else:
                # image dataset usually has no `instance_id`.
                # Therefore, we set it to `i`.
                instance['instance_id'] = i
            if len(instance) > 0:
                instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[int]:
        """Filter annotations according to filter_cfg.

        Returns:
            list[int]: Filtered results.
        """
        if self.test_mode:
            return self.valid_data_indices

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list
                           if len(data_info['instances']) > 0)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_indices = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if self.filter_cfg is None:
                if img_id not in ids_in_cat:
                    continue
                if min(width, height) >= 32:
                    valid_data_indices.append(i)
            else:
                if self.filter_cfg.get('filter_empty_gt',
                                       True) and img_id not in ids_in_cat:
                    continue
                if min(width, height) >= self.filter_cfg.get('min_size', 32):
                    valid_data_indices.append(i)

        set_valid_data_indices = set(self.valid_data_indices)
        valid_data_indices = [
            id for id in valid_data_indices if id in set_valid_data_indices
        ]
        return valid_data_indices

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by the index of `self.valid_data_indices` and
        automatically call ``full_init`` if the dataset has not been fully
        initialized.

        Args:
            idx (int): The index of data in `self.valid_data_indices`.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        ori_idx = self.valid_data_indices[idx]
        data_info = super().get_data_info(ori_idx)
        # Reset the `sample_idx`
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        return data_info

    @force_full_init
    def _get_ori_data_info(self, ori_idx: int) -> dict:
        """Get annotation by the index of `self.data_list` and automatically
        call ``full_init`` if the dataset has not been fully initialized.

        Args:
            ori_idx (int): The index of data in `self.data_list`.

        Returns:
            dict: The ori_idx-th annotation of the `self.data_list``.
        """
        ori_data_info = super().get_data_info(ori_idx)
        # delete the `sample_idx` key
        ori_data_info.pop('sample_idx')
        return ori_data_info

    @force_full_init
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        return len(self.valid_data_indices)

    def prepare_data(self, idx: int) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        if self.ref_img_sampler is not None:
            data_infos = self.ref_img_sampling(idx, data_info,
                                               **self.ref_img_sampler)
            for _data in data_infos:
                if 'video_id' in data_infos[0]:
                    assert data_infos[0]['video_id'] == _data['video_id']
                _data['is_video_data'] = self.load_as_video
            final_data_info = data_infos[0].copy()
            # Collate data_list scatters (list of dict to dict of list)
            for key in final_data_info.keys():
                final_data_info[key] = [_data[key] for _data in data_infos]
        else:
            final_data_info = data_info.copy()
            final_data_info['is_video_data'] = self.load_as_video

        return self.pipeline(final_data_info)

    def ref_img_sampling(self,
                         idx: int,
                         data_info: dict,
                         frame_range: list,
                         stride: int = 1,
                         num_ref_imgs: int = 1,
                         filter_key_img: bool = True,
                         method: str = 'uniform') -> List[dict]:
        """Sampling reference frames in the same video for key frame.

        Args:
            idx (int): The index of `data_info`.
            data_info (dict): The information of key frame.
            frame_range (List(int) | int): The sampling range of reference
                frames in the same video for key frame.
            stride (int): The sampling frame stride when sampling reference
                images. Default: 1.
            num_ref_imgs (int): The number of sampled reference images.
                Default: 1.
            filter_key_img (bool): If False, the key image will be in the
                sampling reference candidates, otherwise, it is exclude.
                Default: True.
            method (str): The sampling method. Options are 'uniform',
                'bilateral_uniform', 'test_with_adaptive_stride',
                'test_with_fix_stride'. 'uniform' denotes reference images are
                randomly sampled from the nearby frames of key frame.
                'bilateral_uniform' denotes reference images are randomly
                sampled from the two sides of the nearby frames of key frame.
                'test_with_adaptive_stride' is only used in testing, and
                denotes the sampling frame stride is equal to (video length /
                the number of reference images). test_with_fix_stride is only
                used in testing with sampling frame stride equalling to
                `stride`. Default: 'uniform'.

        Returns:
            list[dict]: `data_info` and the reference images information.
        """
        assert isinstance(data_info, dict)
        if isinstance(frame_range, int):
            assert frame_range >= 0, 'frame_range can not be a negative value.'
            frame_range = [-frame_range, frame_range]
        elif isinstance(frame_range, list):
            assert len(frame_range) == 2, 'The length must be 2.'
            assert frame_range[0] <= 0 and frame_range[1] >= 0
            for i in frame_range:
                assert isinstance(i, int), 'Each element must be int.'
        else:
            raise TypeError('The type of frame_range must be int or list.')

        if 'test' in method and \
                (frame_range[1] - frame_range[0]) != num_ref_imgs:
            logger = MMLogger.get_current_instance()
            logger.info(
                'Warning:'
                "frame_range[1] - frame_range[0] isn't equal to num_ref_imgs."
                'Set num_ref_imgs to frame_range[1] - frame_range[0].')
            self.ref_img_sampler[
                'num_ref_imgs'] = frame_range[1] - frame_range[0]

        if (not self.load_as_video) or data_info.get('frame_id', -1) < 0 \
                or (frame_range[0] == 0 and frame_range[1] == 0):
            ref_data_infos = []
            for i in range(num_ref_imgs):
                ref_data_infos.append(data_info.copy())
        else:
            frame_id = data_info['frame_id']
            left = max(0, frame_id + frame_range[0])
            right = min(frame_id + frame_range[1],
                        data_info['video_length'] - 1)
            frame_ids = list(range(0, data_info['video_length']))

            ref_frame_ids = []
            if method == 'uniform':
                valid_ids = frame_ids[left:right + 1]
                if filter_key_img and frame_id in valid_ids:
                    valid_ids.remove(frame_id)
                num_samples = min(num_ref_imgs, len(valid_ids))
                ref_frame_ids.extend(random.sample(valid_ids, num_samples))
            elif method == 'bilateral_uniform':
                assert num_ref_imgs % 2 == 0, \
                    'only support load even number of ref_imgs.'
                for mode in ['left', 'right']:
                    if mode == 'left':
                        valid_ids = frame_ids[left:frame_id + 1]
                    else:
                        valid_ids = frame_ids[frame_id:right + 1]
                    if filter_key_img and frame_id in valid_ids:
                        valid_ids.remove(frame_id)
                    num_samples = min(num_ref_imgs // 2, len(valid_ids))
                    sampled_inds = random.sample(valid_ids, num_samples)
                    ref_frame_ids.extend(sampled_inds)
            elif method == 'test_with_adaptive_stride':
                if frame_id == 0:
                    stride = float(len(frame_ids) - 1) / (num_ref_imgs - 1)
                    for i in range(num_ref_imgs):
                        ref_id = round(i * stride)
                        ref_frame_ids.append(frame_ids[ref_id])
            elif method == 'test_with_fix_stride':
                if frame_id == 0:
                    for i in range(frame_range[0], 1):
                        ref_frame_ids.append(frame_ids[0])
                    for i in range(1, frame_range[1] + 1):
                        ref_id = min(round(i * stride), len(frame_ids) - 1)
                        ref_frame_ids.append(frame_ids[ref_id])
                elif frame_id % stride == 0:
                    ref_id = min(
                        round(frame_id + frame_range[1] * stride),
                        len(frame_ids) - 1)
                    ref_frame_ids.append(frame_ids[ref_id])
                data_info['num_left_ref_imgs'] = abs(frame_range[0])
                data_info['frame_stride'] = stride
            else:
                raise NotImplementedError

            ref_data_infos = []
            for ref_frame_id in ref_frame_ids:
                offset = ref_frame_id - frame_id
                ref_data_info = self._get_ori_data_info(
                    self.valid_data_indices[idx] + offset)

                # We need data_info and ref_data_info to have the same keys.
                for key in data_info.keys():
                    if key not in ref_data_info:
                        ref_data_info[key] = data_info[key]

                ref_data_infos.append(ref_data_info)

            ref_data_infos = sorted(
                ref_data_infos, key=lambda i: i['frame_id'])
        return [data_info, *ref_data_infos]
