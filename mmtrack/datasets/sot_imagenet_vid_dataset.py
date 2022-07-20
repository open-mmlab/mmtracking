# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import numpy as np
from mmengine.dataset import force_full_init
from mmengine.fileio.file_client import FileClient

from mmtrack.registry import DATASETS
from .api_wrappers import CocoVID
from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class SOTImageNetVIDDataset(BaseSOTDataset):
    """ImageNet VID dataset of single object tracking.

    The dataset only support training mode.
    """

    def __init__(self, *args, **kwargs):
        """Initialization of SOT dataset class."""
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``.

        Returns:
            list[dict]: The length of the list is the number of instances. The
                inner dict contains instance ID in CocoVID API.
        """
        file_client = FileClient.infer_client(uri=self.ann_file)
        with file_client.get_local_path(self.ann_file) as local_file:
            self.coco = CocoVID(local_file)
        data_infos = [
            dict(ins_id=ins_id) for ins_id in self.coco.instancesToImgs.keys()
        ]
        return data_infos

    def get_bboxes_from_video(self, video_idx: int) -> np.ndarray:
        """Get bbox annotation about one instance in a video. Considering
        `get_bboxes_from_video` in `SOTBaseDataset` is not compatible with
        `SOTImageNetVIDDataset`, we oveload this function though it's not
        called by `self.get_ann_infos_from_video`.

        Args:
            video_idx (int): The index of video. Here, each video_idx denotes
                an instance.

        Returns:
            ndarray: In [N, 4] shape. The bbox is in (x, y, w, h) format.
        """
        instance_id = self.get_data_info(video_idx)['ins_id']
        img_ids = self.coco.instancesToImgs[instance_id]
        bboxes = []
        for img_id in img_ids:
            for ann in self.coco.imgToAnns[img_id]:
                if ann['instance_id'] == instance_id:
                    bboxes.append(ann['bbox'])
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        return bboxes

    def get_img_infos_from_video(self, video_idx: int) -> dict:
        """Get image information about one instance in a video.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: {
                    'video_id': int,
                    'frame_ids': np.ndarray,
                    'img_paths': list[str],
                    'video_length': int
                  }
        """
        instance_id = self.get_data_info(video_idx)['ins_id']
        img_ids = self.coco.instancesToImgs[instance_id]
        frame_ids = []
        img_names = []
        # In ImageNetVID dataset, frame_ids may not be continuous.
        for img_id in img_ids:
            frame_ids.append(self.coco.imgs[img_id]['frame_id'])
            img_names.append(
                osp.join(self.data_prefix['img_path'],
                         self.coco.imgs[img_id]['file_name']))
        img_infos = dict(
            video_id=video_idx,
            frame_ids=frame_ids,
            img_paths=img_names,
            video_length=len(frame_ids))
        return img_infos

    def get_ann_infos_from_video(self, video_idx: int) -> dict:
        """Get annotation information about one instance in a video.
        Note: We overload this function for speed up loading video information.

        Args:
            video_idx (int): The index of video. Here, each video_idx denotes
                an instance.

        Returns:
            dict: {
                    'bboxes': np.ndarray in (N, 4) shape,
                    'bboxes_isvalid': np.ndarray,
                    'visible': np.ndarray
                  }.
                  The annotation information in some datasets may contain
                    'visible_ratio'. The bbox is in (x1, y1, x2, y2) format.
        """
        instance_id = self.get_data_info(video_idx)['ins_id']
        img_ids = self.coco.instancesToImgs[instance_id]
        bboxes = []
        visible = []
        for img_id in img_ids:
            for ann in self.coco.imgToAnns[img_id]:
                if ann['instance_id'] == instance_id:
                    bboxes.append(ann['bbox'])
                    visible.append(not ann.get('occluded', False))
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        bboxes_isvalid = (bboxes[:, 2] > self.bbox_min_size) & (
            bboxes[:, 3] > self.bbox_min_size)
        bboxes[:, 2:] += bboxes[:, :2]
        visible = np.array(visible, dtype=np.bool_) & bboxes_isvalid
        ann_infos = ann_infos = dict(
            bboxes=bboxes, bboxes_isvalid=bboxes_isvalid, visible=visible)
        return ann_infos

    def get_visibility_from_video(self, video_idx: int) -> dict:
        """Get the visible information about one instance in a video.
        Considering `get_visibility_from_video` in `SOTBaseDataset` is not
        compatible with `SOTImageNetVIDDataset`, we oveload this function
        though it's not called by `self.get_ann_infos_from_video`.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: The visibilities of each object in the video.
        """
        instance_id = self.get_data_info(video_idx)['ins_id']
        img_ids = self.coco.instancesToImgs[instance_id]
        visible = []
        for img_id in img_ids:
            for ann in self.coco.imgToAnns[img_id]:
                if ann['instance_id'] == instance_id:
                    visible.append(not ann.get('occluded', False))
        visible_info = dict(visible=np.array(visible, dtype=np.bool_))
        return visible_info

    @force_full_init
    def get_len_per_video(self, video_idx: int) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        instance_id = self.get_data_info(video_idx)['ins_id']
        return len(self.coco.instancesToImgs[instance_id])
