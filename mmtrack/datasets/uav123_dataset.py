# Copyright (c) OpenMMLab. All rights reserved.
import time

from mmdet.datasets import DATASETS

from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class UAV123Dataset(BaseSOTDataset):
    """UAV123 dataset of single object tracking.

    The dataset is only used to test.
    """

    def __init__(self, ann_file, *args, **kwargs):
        """Initialization of SOT dataset class.

        Args:
            ann_file (str): The file contains data information. It will be
                loaded and parsed in the `self.load_data_infos` function.
        """
        self.ann_file = ann_file
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
                    }
        """
        print('Loading UAV123 dataset...')
        start_time = time.time()
        data_infos = []
        with open(self.ann_file, 'r') as f:
            # the first line of annotation file is dataset comment.
            for line in f.readlines()[1:]:
                line = line.strip().split(',')
                data_info = dict(
                    video_path=line[0],
                    ann_path=line[1],
                    start_frame_id=int(line[2]),
                    end_frame_id=int(line[3]),
                    framename_template='%06d.jpg')
                data_infos.append(data_info)
        print(f'UAV123 dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos
