# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
from typing import List

import numpy as np

from mmtrack.registry import DATASETS
from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class VOTDataset(BaseSOTDataset):
    """VOT dataset of single object tracking. The dataset is only used to test.

    Args:
        dataset_type (str, optional): The type of VOT challenge. The
            optional values are in ['vot2018', 'vot2018_lt',
            'vot2019', 'vot2019_lt']
    """

    def __init__(self, dataset_type: str = 'vot2018', *args, **kwargs):
        """Initialization of SOT dataset class."""
        assert dataset_type in [
            'vot2018', 'vot2018_lt', 'vot2019', 'vot2019_lt'
        ], 'We only support VOT-[2018~2019] chanllenges'
        self.dataset_type = dataset_type
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load dataset information.

        Returns:
            list[dict]: The length of the list is the number of videos. The
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
        print('Loading VOT dataset...')
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
        print(f'VOT dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def get_ann_infos_from_video(self, video_ind: int) -> np.ndarray:
        """Get bboxes annotation about the instance in a video.

        Args:
            video_ind (int): video index

        Returns:
            np.ndarray: in [N, 8] shape. The N is the bbox number and the bbox
                is in (x1, y1, x2, y2, x3, y3, x4, y4) format.
        """
        bboxes = self.get_bboxes_from_video(video_ind)
        if bboxes.shape[1] == 4:
            x1, y1 = bboxes[:, 0], bboxes[:, 1],
            x2, y2 = bboxes[:, 0] + bboxes[:, 2], bboxes[:, 1],
            x3, y3 = bboxes[:, 0] + bboxes[:, 2], bboxes[:, 1] + bboxes[:, 3]
            x4, y4 = bboxes[:, 0], bboxes[:, 1] + bboxes[:, 3],
            bboxes = np.stack((x1, y1, x2, y2, x3, y3, x4, y4), axis=-1)

        visible_info = self.get_visibility_from_video(video_ind)
        # bboxes in VOT datasets are all valid
        bboxes_isvalid = np.array([True] * len(bboxes), dtype=np.bool_)
        ann_infos = dict(
            bboxes=bboxes, bboxes_isvalid=bboxes_isvalid, **visible_info)
        return ann_infos
