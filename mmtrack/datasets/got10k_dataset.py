# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time
from typing import List

import numpy as np

from mmtrack.registry import DATASETS
from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class GOT10kDataset(BaseSOTDataset):
    """GOT10k Dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self, *args, **kwargs):
        """Initialization of SOT dataset class."""
        super(GOT10kDataset, self).__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``.

        Returns:
            list[dict]: the length of the list is the number of videos. The
                inner dict is in the following format:
                    {
                        'video_path': the video path
                        'ann_path': the annotation path
                        'start_frame_id': the starting frame number contained
                            in the image name
                        'end_frame_id': the ending frame number contained in
                            the image name
                        'framename_template': the template of image name
                    }
        """
        print('Loading GOT10k dataset...')
        start_time = time.time()
        data_infos = []
        data_infos_str = self._loadtxt(
            self.ann_file, return_ndarray=False).split('\n')
        # the first line of annotation file is a dataset comment.
        for line in data_infos_str[1:]:
            # compatible with different OS.
            line = line.strip().replace('/', os.sep).split(',')
            data_info = dict(
                video_path=line[0],
                ann_path=line[1],
                start_frame_id=int(line[2]),
                end_frame_id=int(line[3]),
                framename_template='%08d.jpg')
            data_infos.append(data_info)
        print(f'GOT10k dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def get_visibility_from_video(self, video_idx: int) -> dict:
        """Get the visible information of instance in a video.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: The visibilities of each object in the video.
        """
        if not self.test_mode:
            video_path = self.get_data_info(video_idx)['video_path']
            absense_info_path = osp.join(self.data_prefix['img_path'],
                                         video_path, 'absence.label')
            cover_info_path = osp.join(self.data_prefix['img_path'],
                                       video_path, 'cover.label')
            absense_info = self._loadtxt(absense_info_path, dtype=bool)
            # The values of key 'cover' are
            # int numbers in range [0,8], which correspond to
            # ranges of object visible ratios: 0%, (0%, 15%],
            # (15%~30%], (30%, 45%], (45%, 60%],(60%, 75%],
            # (75%, 90%], (90%, 100%) and 100% respectively
            cover_info = self._loadtxt(cover_info_path, dtype=int)
            visible = ~absense_info & (cover_info > 0)
            visible_ratio = cover_info / 8.
            return dict(visible=visible, visible_ratio=visible_ratio)
        else:
            return super(GOT10kDataset,
                         self).get_visibility_from_video(video_idx)

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
