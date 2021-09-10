# Copyright (c) OpenMMLab. All rights reserved.
import json
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
        with open(self.ann_file, 'r') as f:
            info = json.load(f)
            video_info = info['videos']
            imgs_info = info['images']
        print('-------- Image Number: {} --------'.format(len(results)))

        format_results = defaultdict(list)
        for img_id, res in enumerate(results):
            img_info = imgs_info[img_id]
            assert img_info['id'] == img_id + 1, 'img id is not matched'
            video_name = video_info['name']
            format_results[video_name].append(res[:4])

        assert len(video_info) == len(
            format_results), 'video number is not right {}--{}'.format(
                len(video_info), len(format_results))

        # writing submitted results
        print('writing submitted results to {}'.format(resfile_path))
        for v_name, bboxes in format_results.items():
            vid_txt = osp.join(resfile_path, '{}.txt'.format(v_name))
            with open(vid_txt, 'w') as f:
                for i, bbox in enumerate(bboxes):
                    bbox = [
                        str(bbox[0]),
                        str(bbox[1]),
                        str(bbox[2] - bbox[0]),
                        str(bbox[3] - bbox[1])
                    ]
                    line = ','.join(bbox) + '\n'
                    f.writelines(line)
