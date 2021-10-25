# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np

from mmtrack.datasets import DATASETS as DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../../data')
LASOT_ANN_PATH = f'{PREFIX}/demo_sot_data/lasot'


def test_format_results():
    dataset_class = DATASETS.get('TrackingNetDataset')
    dataset = dataset_class(
        ann_file=osp.join(LASOT_ANN_PATH, 'lasot_test_dummy.json'),
        pipeline=[])

    results = []
    for video_name in ['airplane-1', 'airplane-2']:
        results.extend(
            mmcv.list_from_file(
                osp.join(LASOT_ANN_PATH, video_name, 'track_results.txt')))

    track_results = []
    for result in results:
        x1, y1, x2, y2 = result.split(',')
        track_results.append(
            np.array([float(x1),
                      float(y1),
                      float(x2),
                      float(y2), 0.]))

    track_results = dict(track_results=track_results)

    tmp_dir = tempfile.TemporaryDirectory()
    dataset.format_results(track_results, resfile_path=tmp_dir.name)
    tmp_dir.cleanup()
