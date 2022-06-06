# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import numpy as np
from mmengine.dataset import force_full_init
from mmengine.fileio.file_client import FileClient
from pycocotools.coco import COCO

from mmtrack.registry import DATASETS
from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class SOTCocoDataset(BaseSOTDataset):
    """Coco dataset of single object tracking.

    The dataset only support training mode.
    """

    def __init__(self, *args, **kwargs):
        """Initialization of SOT dataset class."""
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``.
        Each instance is viewed as a video.

        Returns:
            list[dict]: The length of the list is the number of valid object
                annotations. The inner dict contains annotation ID in coco
                API.
        """
        file_client = FileClient.infer_client(uri=self.ann_file)
        with file_client.get_local_path(self.ann_file) as local_file:
            self.coco = COCO(local_file)
        ann_list = list(self.coco.anns.keys())
        data_infos = [
            dict(ann_id=ann) for ann in ann_list
            if self.coco.anns[ann].get('iscrowd', 0) == 0
        ]
        return data_infos

    def get_bboxes_from_video(self, video_idx: int) -> np.ndarray:
        """Get bbox annotation about one instance in an image.

        Args:
            video_idx (int): The index of video.

        Returns:
            ndarray: In [1, 4] shape. The bbox is in (x, y, w, h) format.
        """
        ann_id = self.get_data_info(video_idx)['ann_id']
        anno = self.coco.anns[ann_id]
        bboxes = np.array(anno['bbox'], dtype=np.float32).reshape(-1, 4)
        return bboxes

    def get_img_infos_from_video(self, video_idx: int) -> dict:
        """Get the image information about one instance in a image.

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
        ann_id = self.get_data_info(video_idx)['ann_id']
        imgs = self.coco.loadImgs([self.coco.anns[ann_id]['image_id']])
        img_names = [
            osp.join(self.data_prefix['img_path'], img['file_name'])
            for img in imgs
        ]
        frame_ids = np.arange(self.get_len_per_video(video_idx))
        img_infos = dict(
            video_id=video_idx,
            frame_ids=frame_ids,
            img_paths=img_names,
            video_length=1)
        return img_infos

    @force_full_init
    def get_len_per_video(self, video_idx: int) -> int:
        """Get the number of frames in a video. Here, it returns 1 since Coco
        is a image dataset.

        Args:
            video_idx (int): The index of video. Each video_idx denotes an
                instance.

        Returns:
            int: The length of video.
        """
        return 1
