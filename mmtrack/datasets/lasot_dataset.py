# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time
from typing import List

from mmtrack.registry import DATASETS
from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class LaSOTDataset(BaseSOTDataset):
    """LaSOT dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self, *args, **kwargs):
        """Initialization of SOT dataset class."""
        super(LaSOTDataset, self).__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``.

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
        print('Loading LaSOT dataset...')
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
        print(f'LaSOT dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def get_visibility_from_video(self, video_idx: int) -> dict:
        """Get the visible information of instance in a video.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: The visibilities of each object in the video.
        """
        video_path = osp.dirname(self.get_data_info(video_idx)['video_path'])
        full_occlusion_file = osp.join(self.data_prefix['img_path'],
                                       video_path, 'full_occlusion.txt')
        out_of_view_file = osp.join(self.data_prefix['img_path'], video_path,
                                    'out_of_view.txt')
        full_occlusion = self._loadtxt(
            full_occlusion_file, dtype=bool, delimiter=',')
        out_of_view = self._loadtxt(
            out_of_view_file, dtype=bool, delimiter=',')
        visible = ~(full_occlusion | out_of_view)
        return dict(visible=visible)
