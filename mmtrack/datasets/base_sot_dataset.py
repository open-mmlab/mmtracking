# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
from abc import ABCMeta
from io import StringIO
from typing import Any, Optional, Sequence, Union

import numpy as np
from addict import Dict
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio.file_client import FileClient

from mmtrack.registry import DATASETS


@DATASETS.register_module()
class BaseSOTDataset(BaseDataset, metaclass=ABCMeta):
    """Base dataset for SOT task. The dataset can both support training and
    testing mode.

    Args:
        bbox_min_size (int, optional): Only bounding boxes whose sizes are
            larger than ``bbox_min_size`` can be regarded as valid.
            Default to 0.
        only_eval_visible (bool, optional): Whether to only evaluate frames
            where object are visible. Default to False.
    """

    META = dict(CLASSES=None)

    def __init__(self,
                 bbox_min_size: int = 0,
                 only_eval_visible: bool = False,
                 *args,
                 **kwargs):
        self.bbox_min_size = bbox_min_size
        self.only_eval_visible = only_eval_visible
        # ``self.load_as_video`` must be set to True in order to using
        # distributed video sampler to load dataset when testing.
        self.load_as_video = True
        super().__init__(*args, **kwargs)

        # used to record the video information at the beginning of the video
        # test. Thus, we can avoid reloading the files of video information
        # repeatedly in all frames of one video.
        self.test_memo = Dict()

    def _loadtxt(self,
                 filepath: str,
                 dtype=np.float32,
                 delimiter: Optional[str] = None,
                 skiprows: int = 0,
                 return_ndarray: bool = True) -> Union[np.ndarray, str]:
        """Load TEXT file.

        Args:
            filepath (str): The path of file.
            dtype (data-type, optional): Data-type of the resulting array.
                Defaults to np.float32.
            delimiter (str, optional): The string used to separate values.
                Defaults to None.
            skiprows (int, optional): Skip the first ``skiprows`` lines,
                including comments. Defaults to 0.
            return_ndarray (bool, optional): Whether to return the ``ndarray``
                type. Defaults to True.

        Returns:
            Union[np.ndarray, str]: Contents of the file.
        """
        file_client = FileClient.infer_client(uri=filepath)
        file_string = file_client.get_text(filepath)
        if return_ndarray:
            return np.loadtxt(
                StringIO(file_string),
                dtype=dtype,
                delimiter=delimiter,
                skiprows=skiprows)
        else:
            return file_string.strip()

    def get_bboxes_from_video(self, video_idx: int) -> np.ndarray:
        """Get bboxes annotation about the instance in a video.

        Args:
            video_idx (int): video index

        Returns:
            np.ndarray: In [N, 4] shape. The N is the number of bbox and
                the bbox is in (x, y, w, h) format.
        """
        meta_video_info = self.get_data_info(video_idx)
        bbox_path = osp.join(self.data_prefix['img_path'],
                             meta_video_info['ann_path'])
        bboxes = self._loadtxt(bbox_path, dtype=float, delimiter=',')
        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=0)

        end_frame_id = meta_video_info['end_frame_id']
        start_frame_id = meta_video_info['start_frame_id']

        if not self.test_mode:
            assert len(bboxes) == (
                end_frame_id - start_frame_id + 1
            ), f'{len(bboxes)} is not equal to {end_frame_id}-{start_frame_id}+1'  # noqa
        return bboxes

    def get_len_per_video(self, video_idx: int) -> int:
        """Get the number of frames in a video.

        Args:
            video_idx (int): The index of video.

        Returns:
            int: The length of the video.
        """
        return self.get_data_info(
            video_idx)['end_frame_id'] - self.get_data_info(
                video_idx)['start_frame_id'] + 1

    def get_visibility_from_video(self, video_idx: int) -> dict:
        """Get the visible information of instance in a video.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: The visibilities of each object in the video.
        """
        visible = np.array([True] * self.get_len_per_video(video_idx))
        return dict(visible=visible)

    def get_masks_from_video(self, video_idx: int) -> Any:
        """Get the mask information of instance in a video.

        Args:
            video_idx (int): The index of video.

        Returns:
            Any: Not implemented yet.
        """
        pass

    def get_img_infos_from_video(self, video_idx: int) -> dict:
        """Get the information of images in a video.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: {
                    'video_id': int,
                    'frame_ids': np.ndarray,
                    'img_paths': list[str]
                  }
        """
        img_paths = []
        meta_video_info = self.get_data_info(video_idx)
        start_frame_id = meta_video_info['start_frame_id']
        end_frame_id = meta_video_info['end_frame_id']
        framename_template = meta_video_info['framename_template']
        for frame_id in range(start_frame_id, end_frame_id + 1):
            img_paths.append(
                osp.join(self.data_prefix['img_path'],
                         meta_video_info['video_path'],
                         framename_template % frame_id))
        frame_ids = np.arange(self.get_len_per_video(video_idx))

        img_infos = dict(
            video_id=video_idx, frame_ids=frame_ids, img_paths=img_paths)
        return img_infos

    def get_ann_infos_from_video(self, video_idx: int) -> dict:
        """Get the information of annotations in a video.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: {
                    'bboxes': np.ndarray in (N, 4) shape,
                    'bboxes_isvalid': np.ndarray,
                    'visible': np.ndarray
                  }.
                  The annotation information in some datasets may contain
                    'visible_ratio'. The bbox is in (x1, y1, x2, y2) format.
        """
        bboxes = self.get_bboxes_from_video(video_idx)
        # The visible information in some datasets may contain
        # 'visible_ratio'.
        visible_info = self.get_visibility_from_video(video_idx)
        bboxes_isvalid = (bboxes[:, 2] > self.bbox_min_size) & (
            bboxes[:, 3] > self.bbox_min_size)
        visible_info['visible'] = visible_info['visible'] & bboxes_isvalid
        bboxes[:, 2:] += bboxes[:, :2]

        ann_infos = dict(
            bboxes=bboxes, bboxes_isvalid=bboxes_isvalid, **visible_info)
        return ann_infos

    def prepare_test_data(self, video_idx: int, frame_idx: int) -> dict:
        """Get testing data of one frame. We parse one video, get one frame
        from it and pass the frame information to the pipeline.

        Args:
            video_idx (int): The index of video.
            frame_idx (int): The index of frame.

        Returns:
            dict: Testing data of one frame.
        """
        # Avoid reloading the files of video information
        # repeatedly in all frames of one video.
        if self.test_memo.get('video_idx', None) != video_idx:
            self.test_memo.video_idx = video_idx
            ann_infos = self.get_ann_infos_from_video(video_idx)
            img_infos = self.get_img_infos_from_video(video_idx)
            self.test_memo.video_infos = dict(**img_infos, **ann_infos)
        assert 'video_idx' in self.test_memo and 'video_infos'\
            in self.test_memo

        results = {}
        results['img_path'] = self.test_memo.video_infos['img_paths'][
            frame_idx]
        results['frame_id'] = frame_idx

        results['instances'] = []
        instance = {}
        instance['bbox'] = self.test_memo.video_infos['bboxes'][frame_idx]
        instance['visible'] = self.test_memo.video_infos['visible'][frame_idx]
        instance['bbox_label'] = np.array([0], dtype=np.int32)
        instance['ignore_flag'] = False
        results['instances'].append(instance)

        results = self.pipeline(results)
        return results

    def prepare_train_data(self, video_idx: int) -> dict:
        """Get training data sampled from some videos. We firstly sample two
        videos from the dataset and then parse the data information in the
        subsequent pipeline. The first operation in the training pipeline must
        be frames sampling.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: Training data pairs, triplets or groups.
        """
        video_idxes = random.choices(list(range(len(self))), k=2)
        pair_video_infos = []
        for video_idx in video_idxes:
            ann_infos = self.get_ann_infos_from_video(video_idx)
            img_infos = self.get_img_infos_from_video(video_idx)
            video_infos = dict(**img_infos, **ann_infos)
            pair_video_infos.append(video_infos)

        results = self.pipeline(pair_video_infos)
        return results

    def prepare_data(self, idx: Union[Sequence[int], int]) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        if self.test_mode:
            assert isinstance(idx, Sequence) and len(idx) == 2
            # the first element in the ``Sequence`` is the video index and the
            # second element in the ``Sequence`` is the frame index
            return self.prepare_test_data(idx[0], idx[1])
        else:
            assert isinstance(idx, int)
            return self.prepare_train_data(idx)

    @property
    def num_videos(self) -> int:
        """Get the number of videos in the dataset.

        Returns:
            int: The number of videos.
        """
        num_videos = len(self.data_address) if self.serialize_data else len(
            self.data_list)
        return num_videos

    @force_full_init
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.test_mode:
            return sum(
                self.get_len_per_video(idx) for idx in range(self.num_videos))
        else:
            return self.num_videos
