# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
from collections import defaultdict

from mmdet.datasets import DATASETS

from .sot_test_dataset import SOTTestDataset


@DATASETS.register_module()
class GOT10kDataset(SOTTestDataset):
    """GOT10k dataset for the testing of single object tracking.

    The dataset doesn't support training mode.
    """

    def format_results(self, results, resfile_path=None):
        """Format the results to txts (standard format for GOT10k Challenge).

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
        track_bboxes = results['track_bboxes']
        print('-------- There are total {} images --------'.format(
            len(track_bboxes)))

        video_info = self.coco.videos
        format_results = defaultdict(list)
        for img_id, track_bbox in enumerate(track_bboxes):
            img_info = self.data_infos[img_id]
            assert img_info['id'] == img_id + 1, 'img id is not matched'
            video_name = video_info[img_info['video_id']]['name']
            format_results[video_name].append(track_bbox[:4])

        assert len(video_info) == len(
            format_results
        ), 'The number of video is not matched! There are {} videos in the \
            dataset and {} videos in the testing results'.format(
            len(video_info), len(format_results))

        # writing submitted results
        # TODO record test time
        print('writing submitted results to {}'.format(resfile_path))
        for video_name, bboxes in format_results.items():
            video_file_path = osp.join(resfile_path, video_name)
            if not osp.isdir(video_file_path):
                os.makedirs(video_file_path, exist_ok=True)
            video_txt = osp.join(video_file_path,
                                 '{}_001.txt'.format(video_name))
            with open(video_txt, 'w') as f:
                for bbox in bboxes:
                    bbox = [
                        str(f'{bbox[0]:.4f}'),
                        str(f'{bbox[1]:.4f}'),
                        str(f'{(bbox[2] - bbox[0]):.4f}'),
                        str(f'{(bbox[3] - bbox[1]):.4f}')
                    ]
                    line = ','.join(bbox) + '\n'
                    f.writelines(line)
        shutil.make_archive(resfile_path, 'zip', resfile_path)
        shutil.rmtree(resfile_path)
