# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
from typing import List, Union

import numpy as np

from mmtrack.registry import DATASETS
from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class TrackingNetDataset(BaseSOTDataset):
    """TrackingNet dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self,
                 chunks_list: List[Union[int, str]] = ['all'],
                 *args,
                 **kwargs):
        """Initialization of SOT dataset class.

        Args:
            chunks_list (list[Union[int, str]], optional): The training chunks.
                The optional values in this list are: 0, 1, 2, ..., 10, 11 and
                'all'. Some methods may only use part of the dataset. Default
                to all chunks, namely ['all'].
        """
        if isinstance(chunks_list, (str, int)):
            chunks_list = [chunks_list]
        assert set(chunks_list).issubset(set(range(12)) | {'all'})
        if 'all' in chunks_list:
            self.chunks_list = [f'TRAIN_{i}' for i in range(12)]
        else:
            self.chunks_list = [f'TRAIN_{chunk}' for chunk in chunks_list]
        super(TrackingNetDataset, self).__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``.

        Args:
            split (str, optional): the split of dataset. Defaults to 'train'.

        Returns:
            list[dict]: the length of the list is the number of videos. The
                inner dict is in the following format:
                    {
                        'video_path': the video path
                        'ann_path': the annotation path
                        'start_frame_id': the starting frame ID number
                            contained in the image name
                        'end_frame_id': the ending frame ID number contained in
                            the image name
                        'framename_template': the template of image name
                    }
        """
        print('Loading TrackingNet dataset...')
        start_time = time.time()
        chunks = set(self.chunks_list)
        data_infos = []
        data_infos_str = self._loadtxt(
            self.ann_file, return_ndarray=False).split('\n')
        # the first line of annotation file is a dataset comment.
        for line in data_infos_str[1:]:
            # compatible with different OS.
            line = line.strip().replace('/', os.sep).split(',')
            chunk = line[0].split(os.sep)[0]
            if chunk == 'TEST' or chunk in chunks:
                data_info = dict(
                    video_path=line[0],
                    ann_path=line[1],
                    start_frame_id=int(line[2]),
                    end_frame_id=int(line[3]),
                    framename_template='%d.jpg')
                data_infos.append(data_info)
        print(f'TrackingNet dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def prepare_test_data(self, video_idx: int, frame_idx: int) -> dict:
        """Get testing data of one frame. We parse one video, get one frame
        from it and pass the frame information to the pipeline.

        Args:
            video_idx (int): The index of video.
            frame_idx (int): The index of frame.

        Returns:
            dict: Testing data of one frame.
        """
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
        results['video_id'] = video_idx
        results['video_length'] = self.test_memo.video_infos['video_length']

        instance = {}
        if frame_idx == 0:
            ann_infos = self.get_ann_infos_from_video(video_idx)
            instance['bbox'] = ann_infos['bboxes'][frame_idx]

        results['instances'] = []
        instance['visible'] = True
        instance['bbox_label'] = np.array([0], dtype=np.int32)
        results['instances'].append(instance)
        results = self.pipeline(results)
        return results
