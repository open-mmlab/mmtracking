# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

from mmdet.datasets.api_wrappers import COCO
from mmengine.fileio import FileClient

from mmtrack.registry import DATASETS
from .api_wrappers import CocoVID
from .base_video_dataset import BaseVideoDataset


@DATASETS.register_module()
class ImagenetVIDDataset(BaseVideoDataset):
    """ImageNet VID dataset for video object detection."""

    METAINFO = {
        'classes':
        ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
         'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
         'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle',
         'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train',
         'turtle', 'watercraft', 'whale', 'zebra')
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

                # load frames for training
                if self.test_mode:
                    assert not raw_img_info['is_vid_train_frame'], \
                        'is_vid_train_frame must be False in testing'
                    valid_data_indices.append(data_id)
                elif raw_img_info['is_vid_train_frame']:
                    valid_data_indices.append(data_id)

                # get data_info
                parsed_data_info = self.parse_data_info(
                    dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))
                data_list.append(parsed_data_info)
                data_id += 1
        assert len(
            valid_data_indices
        ) != 0, f"There is no frame for training in '{self.ann_file}'!"

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

        img_ids = coco.get_img_ids()
        data_id = 0
        valid_data_indices = []
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            # load images for training
            if raw_img_info['is_vid_train_frame']:
                valid_data_indices.append(data_id)

            parsed_data_info = self.parse_data_info(
                dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))
            data_list.append(parsed_data_info)
            data_id += 1
        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f"Annotation ids in '{self.ann_file}' are not unique!"

        return data_list, valid_data_indices
