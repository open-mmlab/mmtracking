# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np

from mmtrack.datasets import DATASETS as DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../../data')

DEMO_ANN_FILE = f'{PREFIX}/demo_vis_data/ann.json'
DEMO_RES_FILE = f'{PREFIX}/demo_vis_data/results.json'


def test_vis_evaluation():
    dataset_class = DATASETS.get('YouTubeVISDataset')
    dataset_object = dataset_class(
        '2019', ann_file=DEMO_ANN_FILE, pipeline=[], test_mode=True)
    results_json = mmcv.load(DEMO_RES_FILE)

    results = defaultdict(list)
    track_bboxes_numpy = []
    for frame_bboxes in results_json['track_bboxes']:
        tmp = []
        for bbox in frame_bboxes:
            tmp.append(np.array(bbox).reshape(-1, 6))
        track_bboxes_numpy.append(tmp)
    results['track_bboxes'] = track_bboxes_numpy
    results['track_masks'] = results_json['track_masks']

    eval_results = dataset_object.evaluate(results, metric=['track_segm'])
    assert eval_results['segm_mAP_50'] == 1.0
    assert eval_results['segm_mAP'] == 1.0
