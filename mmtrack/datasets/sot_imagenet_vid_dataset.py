# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets import DATASETS

from mmtrack.datasets.parsers import CocoVID
from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class SOTImageNetVIDDataset(BaseSOTDataset):
    """ImageNet VID dataset of single object tracking.

    The dataset only support training mode.
    """

    def __init__(self, ann_file, *args, **kwargs):
        """Initialization of SOT dataset class.

        Args:
            ann_file (str): The coco-format annotation file of ImageNet VID
                Dataset. It will be loaded and parsed in the
                `self.load_data_infos` function.
        """
        file_client_args = kwargs.get('file_client_args', dict(backend='disk'))
        self.file_client = mmcv.FileClient(**file_client_args)
        with self.file_client.get_local_path(ann_file) as local_path:
            self.coco = CocoVID(local_path)
        super().__init__(*args, **kwargs)

    def load_data_infos(self, split='train'):
        """Load dataset information.

        Args:
            split (str, optional): The split of dataset. Defaults to 'train'.

        Returns:
            list[int]: The length of the list is the number of instances. The
                elemment in the list is instance ID in coco API.
        """
        data_infos = list(self.coco.instancesToImgs.keys())
        return data_infos

    def get_bboxes_from_video(self, video_ind):
        """Get bbox annotation about the instance in a video. Considering
        `get_bboxes_from_video` in `SOTBaseDataset` is not compatible with
        `SOTImageNetVIDDataset`, we oveload this function though it's not
        called by `self.get_ann_infos_from_video`.

        Args:
            video_ind (int): video index. Each video_ind denotes an instance.

        Returns:
            ndarray: in [N, 4] shape. The bbox is in (x, y, w, h) format.
        """
        instance_id = self.data_infos[video_ind]
        img_ids = self.coco.instancesToImgs[instance_id]
        bboxes = []
        for img_id in img_ids:
            for ann in self.coco.imgToAnns[img_id]:
                if ann['instance_id'] == instance_id:
                    bboxes.append(ann['bbox'])
        bboxes = np.array(bboxes).reshape(-1, 4)
        return bboxes

    def get_img_infos_from_video(self, video_ind):
        """Get image information in a video.

        Args:
            video_ind (int): video index

        Returns:
            dict: {'filename': list[str], 'frame_ids':ndarray, 'video_id':int}
        """
        instance_id = self.data_infos[video_ind]
        img_ids = self.coco.instancesToImgs[instance_id]
        frame_ids = []
        img_names = []
        # In ImageNetVID dataset, frame_ids may not be continuous.
        for img_id in img_ids:
            frame_ids.append(self.coco.imgs[img_id]['frame_id'])
            img_names.append(self.coco.imgs[img_id]['file_name'])
        img_infos = dict(
            filename=img_names, frame_ids=frame_ids, video_id=video_ind)
        return img_infos

    def get_ann_infos_from_video(self, video_ind):
        """Get annotation information in a video.
        Note: We overload this function for speed up loading video information.

        Args:
            video_ind (int): video index. Each video_ind denotes an instance.

        Returns:
            dict: {'bboxes': ndarray in (N, 4) shape, 'bboxes_isvalid':
                ndarray, 'visible':ndarray}. The bbox is in
                (x1, y1, x2, y2) format.
        """
        instance_id = self.data_infos[video_ind]
        img_ids = self.coco.instancesToImgs[instance_id]
        bboxes = []
        visible = []
        for img_id in img_ids:
            for ann in self.coco.imgToAnns[img_id]:
                if ann['instance_id'] == instance_id:
                    bboxes.append(ann['bbox'])
                    visible.append(not ann.get('occluded', False))
        bboxes = np.array(bboxes).reshape(-1, 4)
        bboxes_isvalid = (bboxes[:, 2] > self.bbox_min_size) & (
            bboxes[:, 3] > self.bbox_min_size)
        bboxes[:, 2:] += bboxes[:, :2]
        visible = np.array(visible, dtype=np.bool_) & bboxes_isvalid
        ann_infos = ann_infos = dict(
            bboxes=bboxes, bboxes_isvalid=bboxes_isvalid, visible=visible)
        return ann_infos

    def get_visibility_from_video(self, video_ind):
        """Get the visible information in a video.

        Considering `get_visibility_from_video` in `SOTBaseDataset` is not
        compatible with `SOTImageNetVIDDataset`, we oveload this function
        though it's not called by `self.get_ann_infos_from_video`.
        """
        instance_id = self.data_infos[video_ind]
        img_ids = self.coco.instancesToImgs[instance_id]
        visible = []
        for img_id in img_ids:
            for ann in self.coco.imgToAnns[img_id]:
                if ann['instance_id'] == instance_id:
                    visible.append(not ann.get('occluded', False))
        visible_info = dict(visible=np.array(visible, dtype=np.bool_))
        return visible_info

    def get_len_per_video(self, video_ind):
        """Get the number of frames in a video."""
        instance_id = self.data_infos[video_ind]
        return len(self.coco.instancesToImgs[instance_id])
