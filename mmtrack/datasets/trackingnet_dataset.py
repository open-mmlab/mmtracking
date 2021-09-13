# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import defaultdict

from mmdet.datasets import DATASETS

from .sot_test_dataset import SOTTestDataset


@DATASETS.register_module()
class TrackingNetTestDataset(SOTTestDataset):
    """TrackingNet dataset for the testing of single object tracking.

    The dataset doesn't support training mode.
    """

    def format_results(self, results, resfile_path=None):
        """Format the results to txts (standard format for TrackingNet
        Challenge).

        Args:
            results (dict(list[ndarray])): Testing results of the dataset.
            resfile_path (str): Path to save the formatted results.
                Defaults to None.
        """
        # prepare saved dir
        assert resfile_path is not None, 'Please give key-value pair \
            like resfile_path=xxx in argparse'

        if not osp.isdir(resfile_path):
            os.makedirs(resfile_path, exist_ok=True)

        # transform tracking results format
        # from [bbox_1, bbox_2, ...] to {'video_1':[bbox_1, bbox_2, ...], ...}
        results = results['track_results']
        print('-------- There are total {} images --------'.format(
            len(results)))

        video_info = self.coco.videos
        format_results = defaultdict(list)
        for img_id, res in enumerate(results):
            img_info = self.data_infos[img_id]
            assert img_info['id'] == img_id + 1, 'img id is not matched'
            video_name = video_info[img_info['video_id']]['name']
            format_results[video_name].append(res[:4])

        assert len(video_info) == len(
            format_results
        ), 'The number of video is not matched! There are {} videos in the \
            dataset and {} videos in the testing results'.format(
            len(video_info), len(format_results))

        # writing submitted results
        print('writing submitted results to {}'.format(resfile_path))
        for video_name, bboxes in format_results.items():
            video_txt = osp.join(resfile_path, '{}.txt'.format(video_name))
            with open(video_txt, 'w') as f:
                for bbox in bboxes:
                    bbox = [
                        str(f'{bbox[0]:.3f}'),
                        str(f'{bbox[1]:.3f}'),
                        str(f'{(bbox[2] - bbox[0]):.3f}'),
                        str(f'{(bbox[3] - bbox[1]):.3f}')
                    ]
                    line = ','.join(bbox) + '\n'
                    f.writelines(line)
