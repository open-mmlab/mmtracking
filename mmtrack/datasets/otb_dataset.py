# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
import time

import numpy as np
from mmdet.datasets import DATASETS

from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class OTB100Dataset(BaseSOTDataset):
    """OTB100 dataset of single object tracking.

    The dataset is only used to test.
    """

    def __init__(self, *args, **kwargs):
        """Initialization of SOT dataset class."""
        super().__init__(*args, **kwargs)

    def load_data_infos(self, split='test'):
        """Load dataset information.

        Args:
            split (str, optional): Dataset split. Defaults to 'test'.

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
                        'init_skip_num': (optional) the number of skipped
                            frames when initializing tracker
                    }
        """
        print('Loading OTB100 dataset...')
        start_time = time.time()
        data_infos = []
        data_infos_str = self.loadtxt(
            self.ann_file, return_array=False).split('\n')
        # the first line of annotation file is a dataset comment.
        for line in data_infos_str[1:]:
            # compatible with different OS.
            line = line.strip().replace('/', os.sep).split(',')
            if line[0].split(os.sep)[1] == 'Board':
                framename_template = '%05d.jpg'
            else:
                framename_template = '%04d.jpg'
            data_info = dict(
                video_path=line[0],
                ann_path=line[1],
                start_frame_id=int(line[2]),
                end_frame_id=int(line[3]),
                framename_template=framename_template)
            # Tracker initializatioin in `Tiger1` video will skip the first
            # 5 frames. Details can be seen in the official file
            # `tracker_benchmark_v1.0/initOmit/tiger1.txt`.
            # Annotation loading will refer to this information.
            if line[0].split(os.sep)[1] == 'Tiger1':
                data_info['init_skip_num'] = 5
            data_infos.append(data_info)
        print(f'OTB100 dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def get_bboxes_from_video(self, video_ind):
        """Get bboxes annotation about the instance in a video.

        Args:
            video_ind (int): video index

        Returns:
            ndarray: in [N, 4] shape. The N is the bbox number and the bbox
                is in (x, y, w, h) format.
        """
        bboxes_file = osp.join(self.img_prefix,
                               self.data_infos[video_ind]['ann_path'])
        bboxes = []
        bboxes_info = self.loadtxt(bboxes_file, return_array=False).split('\n')
        for bbox in bboxes_info:
            bbox = list(map(int, re.findall(r'-?\d+', bbox)))
            bboxes.append(bbox)
        bboxes = np.array(bboxes, dtype=float)

        if 'init_skip_num' in self.data_infos[video_ind]:
            init_skip_num = self.data_infos[video_ind]['init_skip_num']
            bboxes = bboxes[init_skip_num:]

        end_frame_id = self.data_infos[video_ind]['end_frame_id']
        start_frame_id = self.data_infos[video_ind]['start_frame_id']
        assert len(bboxes) == (
            end_frame_id - start_frame_id + 1
        ), f'{len(bboxes)} is not equal to {end_frame_id}-{start_frame_id}+1'
        assert bboxes.shape[1] == 4
        return bboxes
