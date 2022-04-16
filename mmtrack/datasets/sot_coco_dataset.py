# Copyright (c) OpenMMLab. All rights reserved.
import time

import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from pycocotools.coco import COCO

from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class SOTCocoDataset(BaseSOTDataset):
    """Coco dataset of single object tracking.

    The dataset only support training mode.
    """

    def __init__(self, ann_file, *args, **kwargs):
        """Initialization of SOT dataset class.

        Args:
            ann_file (str): The official coco annotation file. It will be
                loaded and parsed in the `self.load_data_infos` function.
        """
        file_client_args = kwargs.get('file_client_args', dict(backend='disk'))
        self.file_client = mmcv.FileClient(**file_client_args)
        with self.file_client.get_local_path(ann_file) as local_path:
            self.coco = COCO(local_path)
        super().__init__(*args, **kwargs)

    def load_data_infos(self, split='train'):
        """Load dataset information. Each instance is viewed as a video.

        Args:
            split (str, optional): The split of dataset. Defaults to 'train'.

        Returns:
            list[int]: The length of the list is the number of valid object
                annotations. The elemment in the list is annotation ID in coco
                API.
        """
        print('Loading Coco dataset...')
        start_time = time.time()
        ann_list = list(self.coco.anns.keys())
        videos_list = [
            ann for ann in ann_list
            if self.coco.anns[ann].get('iscrowd', 0) == 0
        ]
        print(f'Coco dataset loaded! ({time.time()-start_time:.2f} s)')
        return videos_list

    def get_bboxes_from_video(self, video_ind):
        """Get bbox annotation about the instance in an image.

        Args:
            video_ind (int): video index. Each video_ind denotes an instance.

        Returns:
            ndarray: in [1, 4] shape. The bbox is in (x, y, w, h) format.
        """
        ann_id = self.data_infos[video_ind]
        anno = self.coco.anns[ann_id]
        bboxes = np.array(anno['bbox']).reshape(-1, 4)
        return bboxes

    def get_img_infos_from_video(self, video_ind):
        """Get all frame paths in a video.

        Args:
            video_ind (int): video index. Each video_ind denotes an instance.

        Returns:
            list[str]: all image paths
        """
        ann_id = self.data_infos[video_ind]
        imgs = self.coco.loadImgs([self.coco.anns[ann_id]['image_id']])
        img_names = [img['file_name'] for img in imgs]
        frame_ids = np.arange(self.get_len_per_video(video_ind))
        img_infos = dict(
            filename=img_names, frame_ids=frame_ids, video_id=video_ind)
        return img_infos

    def get_len_per_video(self, video_ind):
        """Get the number of frames in a video."""
        return 1
