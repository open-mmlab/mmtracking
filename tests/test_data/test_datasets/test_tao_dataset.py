# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmtrack.datasets import DATASETS as DATASETS
from .utils import _create_coco_gt_results

PREFIX = osp.join(osp.dirname(__file__), '../../data')
DEMO_ANN_FILE = f'{PREFIX}/demo_tao_data/ann.json'
DEMO_TAO_DATA = f'{PREFIX}/demo_tao_data/'


def test_load_annotation():
    dataset_class = DATASETS.get('TaoDataset')
    dataset_object = dataset_class(
        ann_file=DEMO_ANN_FILE, classes=['serving_dish', 'baby'], pipeline=[])

    dataset_object.load_as_video = True
    data_infos = dataset_object.load_lvis_anns(DEMO_ANN_FILE)
    assert isinstance(data_infos, list)
    assert len(data_infos) == 2

    dataset_object.load_as_video = False
    data_infos = dataset_object.load_tao_anns(DEMO_ANN_FILE)
    assert isinstance(data_infos, list)
    assert len(data_infos) == 2
    assert len(dataset_object.vid_ids) == 1


def test_tao_evaluation():
    dataset_class = DATASETS.get('TaoDataset')
    dataset_object = dataset_class(
        ann_file=DEMO_ANN_FILE, classes=['serving_dish', 'baby'], pipeline=[])
    results = _create_coco_gt_results(dataset_object)
    eval_results = dataset_object.evaluate(results, metric=['track', 'bbox'])
    assert eval_results['bbox_AP'] == 1
    assert eval_results['track_AP'] == 1
