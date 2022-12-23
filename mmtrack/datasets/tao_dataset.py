# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

from mmdet.datasets.api_wrappers import COCO
from mmengine.fileio import FileClient

from mmtrack.registry import DATASETS
from .api_wrappers import CocoVID
from .base_video_dataset import BaseVideoDataset


@DATASETS.register_module()
class TaoDataset(BaseVideoDataset):
    """Dataset for TAO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        if self.load_as_video:
            data_list, valid_data_indices = self._load_tao_data_list()
        else:
            data_list, valid_data_indices = self._load_lvis_data_list()

        return data_list, valid_data_indices

    def _load_tao_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        file_client = FileClient.infer_client(uri=self.ann_file)
        with file_client.get_local_path(self.ann_file) as local_path:
            coco = CocoVID(local_path)
        self._metainfo['categories'] = coco.cats
        # The order of returned `cat_ids` will not
        # change with the order of the classes
        self.cat_ids = coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(coco.cat_img_map)

        data_list = []
        vid_ids = coco.get_vid_ids()
        for vid_id in vid_ids:
            img_ids = coco.get_img_ids_from_vid(vid_id)
            for img_id in img_ids:
                # load img info
                raw_img_info = coco.load_imgs([img_id])[0]
                if raw_img_info['file_name'].startswith('COCO'):
                    # Convert form the COCO 2014 file naming convention of
                    # COCO_[train/val/test]2014_000000000000.jpg to the 2017
                    # naming convention of 000000000000.jpg
                    # (LVIS v1 will fix this naming issue)
                    raw_img_info['filename'] = raw_img_info['file_name'][-16:]
                else:
                    raw_img_info['filename'] = raw_img_info['file_name']
                raw_img_info['img_id'] = img_id
                raw_img_info['video_length'] = len(img_ids)

                # load ann info
                ann_ids = coco.get_ann_ids(
                    img_ids=[img_id], cat_ids=self.cat_ids)
                raw_ann_info = coco.load_anns(ann_ids)

                # get data_info
                parsed_data_info = self.parse_data_info(
                    dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))
                data_list.append(parsed_data_info)

        valid_data_indices = list(range(len(data_list)))
        return data_list, valid_data_indices

    def _load_lvis_data_list(self):
        """Load annotations from an annotation file named as ``self.ann_file``

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
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            if raw_img_info['file_name'].startswith('COCO'):
                # Convert form the COCO 2014 file naming convention of
                # COCO_[train/val/test]2014_000000000000.jpg to the 2017
                # naming convention of 000000000000.jpg
                # (LVIS v1 will fix this naming issue)
                raw_img_info['filename'] = raw_img_info['file_name'][-16:]
            else:
                raw_img_info['filename'] = raw_img_info['file_name']

            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))
            data_list.append(parsed_data_info)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f"Annotation ids in '{self.ann_file}' are not unique!"

        valid_data_indices = list(range(len(data_list)))
        return data_list, valid_data_indices
