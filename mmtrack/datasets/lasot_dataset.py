# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import time

import numpy as np
from mmdet.datasets import DATASETS

from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class LaSOTDataset(BaseSOTDataset):
    """LaSOT dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self, ann_file, *args, **kwargs):
        """Initialization of SOT dataset class.

        Args:
            ann_file (str): The file contains testing video names. It will be
                loaded in the `self.load_data_infos` function.
        """
        self.ann_file = ann_file
        super(LaSOTDataset, self).__init__(*args, **kwargs)

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
                    }
        """
        print('Loading LaSOT dataset...')
        start_time = time.time()
        assert split in ['train', 'test']
        data_infos = []

        test_videos_list = np.loadtxt(self.ann_file, dtype=np.str_)
        if self.test_mode:
            videos_list = test_videos_list.tolist()
        else:
            all_videos_list = glob.glob(self.img_prefix + '/*/*-[1-20]')
            test_videos = set(test_videos_list)
            videos_list = []
            for x in all_videos_list:
                x = osp.basename(x)
                if x not in test_videos:
                    videos_list.append(x)

        videos_list = sorted(videos_list)
        for video_name in videos_list:
            video_name = osp.join(video_name.split('-')[0], video_name)
            video_path = osp.join(video_name, 'img')
            ann_path = osp.join(video_name, 'groundtruth.txt')
            img_names = glob.glob(
                osp.join(self.img_prefix, video_name, 'img', '*.jpg'))
            end_frame_name = max(
                img_names, key=lambda x: int(osp.basename(x).split('.')[0]))
            end_frame_id = int(osp.basename(end_frame_name).split('.')[0])
            data_infos.append(
                dict(
                    video_path=video_path,
                    ann_path=ann_path,
                    start_frame_id=1,
                    end_frame_id=end_frame_id,
                    framename_template='%08d.jpg'))
        print(f'LaSOT dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def get_visibility_from_video(self, video_ind):
        """Get the visible information of instance in a video."""
        video_path = osp.dirname(self.data_infos[video_ind]['video_path'])
        full_occlusion_file = osp.join(self.img_prefix, video_path,
                                       'full_occlusion.txt')
        out_of_view_file = osp.join(self.img_prefix, video_path,
                                    'out_of_view.txt')
        full_occlusion = np.loadtxt(
            full_occlusion_file, dtype=bool, delimiter=',')
        out_of_view = np.loadtxt(out_of_view_file, dtype=bool, delimiter=',')
        visible = ~(full_occlusion | out_of_view)
        return dict(visible=visible)
