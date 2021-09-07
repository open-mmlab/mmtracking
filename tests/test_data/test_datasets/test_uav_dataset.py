# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest

from mmtrack.datasets import DATASETS as DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../../data')
UAV_ANN_PATH = f'{PREFIX}/demo_sot_data/uav'


@pytest.mark.parametrize('dataset', ['UAVDataset'])
def test_uav_dataset_parse_ann_info(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset = dataset_class(
        ann_file=osp.join(UAV_ANN_PATH, 'uav_test_dummy.json'), pipeline=[])

    # image 5 has 1 objects
    img_id = 5
    img_info = dataset.coco.load_imgs([img_id])[0]
    ann_ids = dataset.coco.get_ann_ids([img_id])
    ann_info = dataset.coco.loadAnns(ann_ids)
    ann = dataset._parse_ann_info(img_info, ann_info)
    assert ann['bboxes'].shape == (4, )
    assert ann['labels'] == 0


def test_uav_evaluation():
    dataset_class = DATASETS.get('UAVDataset')
    dataset = dataset_class(
        ann_file=osp.join(UAV_ANN_PATH, 'uav_test_dummy.json'), pipeline=[])

    results = []
    for video_name in ['ball1', 'drone1']:
        results.extend(
            mmcv.list_from_file(
                osp.join(UAV_ANN_PATH, video_name, 'track_results.txt')))
    track_results = []
    for result in results:
        x1, y1, x2, y2 = result.split(',')
        track_results.append(
            np.array([float(x1),
                      float(y1),
                      float(x2),
                      float(y2), 0.]))

    track_results = dict(track_results=track_results)
    eval_results = dataset.evaluate(track_results, metric=['track'])
    assert eval_results['success'] == 67.524
    assert eval_results['norm_precision'] == 70.0
    assert eval_results['precision'] == 50.0
