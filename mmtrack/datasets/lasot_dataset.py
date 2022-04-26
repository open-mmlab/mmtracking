# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time

from mmdet.datasets import DATASETS

from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class LaSOTDataset(BaseSOTDataset):
    """LaSOT dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self, *args, **kwargs):
        """Initialization of SOT dataset class."""
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
        data_infos_str = self.loadtxt(
            self.ann_file, return_array=False).split('\n')
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

    def get_visibility_from_video(self, video_ind):
        """Get the visible information of instance in a video."""
        video_path = osp.dirname(self.data_infos[video_ind]['video_path'])
        full_occlusion_file = osp.join(self.img_prefix, video_path,
                                       'full_occlusion.txt')
        out_of_view_file = osp.join(self.img_prefix, video_path,
                                    'out_of_view.txt')
        full_occlusion = self.loadtxt(
            full_occlusion_file, dtype=bool, delimiter=',')
        out_of_view = self.loadtxt(out_of_view_file, dtype=bool, delimiter=',')
        visible = ~(full_occlusion | out_of_view)
        return dict(visible=visible)
